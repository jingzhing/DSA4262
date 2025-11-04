import argparse
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    RandomizedSearchCV,
)
from sklearn.utils.class_weight import compute_sample_weight

# you already have this in your project
from json_parse_plus import parse_enhanced


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_labels(labels_csv: str) -> pd.DataFrame:
    """
    Load the provided label CSV and ensure we have the required columns.
    We also coerce transcript_position to int and label to int.
    """
    df = pd.read_csv(labels_csv)
    need = {"gene_id", "transcript_id", "transcript_position", "label"}
    if not need.issubset(df.columns):
        raise ValueError(f"labels CSV must contain: {need}")
    df = df[["gene_id", "transcript_id", "transcript_position", "label"]].copy()
    df["transcript_position"] = df["transcript_position"].astype(int)
    df["label"] = df["label"].astype(int)
    return df


def build_xy_keys_enhanced(json_path: str, labels: pd.DataFrame):
    """
    Parse full dataset with enhanced features, then add transcript-normalized
    features. Output final feature matrix (X_full), labels y, and group IDs.

    Returns:
        keys        : list[(transcript_id, position)]
        X_full      : np.ndarray shape (N, F_final)
        y           : np.ndarray shape (N,)
        groups      : np.ndarray shape (N,) -> gene_id (used for group splits)
    """
    # Only parse sites for which we actually have labels
    keyset = {(t, int(p)) for t, p in zip(labels.transcript_id, labels.transcript_position)}

    # parse_enhanced should return:
    #   keys: [(transcript_id, pos), ...]
    #   X_base: np.array (N, B) main engineered numeric features per site
    #   X_motif: np.array (N, M) DRACH-like motif one-hot features
    keys, X_base, X_motif = parse_enhanced(json_path, restrict_keys=keyset)

    if len(keys) == 0:
        raise SystemExit("No overlap between JSON sites and labels.")

    # map each (transcript_id, pos) -> (label, gene_id)
    lab_map = {
        (r.transcript_id, int(r.transcript_position)): (int(r.label), r.gene_id)
        for r in labels.itertuples(index=False)
    }

    y = np.array([lab_map[k][0] for k in keys], dtype=int)        # (N,)
    groups = np.array([lab_map[k][1] for k in keys])              # (N,)
    transcript_ids = np.array([k[0] for k in keys])               # (N,)

    # ----- transcript-level centering -----
    # For each transcript, compute mean feature vector of X_base,
    # then subtract it from each site's vector to capture "how weird is this site
    # compared to its own transcript's baseline".
    unique_t = np.unique(transcript_ids)
    t_means = {}
    for t_id in unique_t:
        mask = (transcript_ids == t_id)
        t_means[t_id] = X_base[mask].mean(axis=0)  # mean over rows for that transcript

    norm_rows = np.vstack([
        X_base[i] - t_means[transcript_ids[i]]
        for i in range(len(transcript_ids))
    ]).astype(np.float32)

    # final feature vector:
    #   [ raw enhanced stats | transcript-centered deltas | motif one-hot ]
    # dims: base_dim + base_dim + motif_dim
    X_full = np.concatenate([X_base, norm_rows, X_motif], axis=1).astype(np.float32)

    return keys, X_full, y, groups


def _roc_hmean(y_true, y_pred_bin):
    """
    Helper for GHOST fallback: harmonic mean of TPR and TNR.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    tpr = tp / (tp + fn + 1e-12)  # recall / sensitivity
    tnr = tn / (tn + fp + 1e-12)  # specificity
    denom = (tpr + tnr)
    if denom == 0:
        return 0.0
    return 2 * (tpr * tnr) / denom


def ghost_threshold_tuning_probs(y_true, proba, thresholds=np.arange(0.05, 0.96, 0.05)):
    """
    Pick an operating threshold for convenience reporting.

    Strategy:
    1. Choose the threshold that maximizes Cohen's kappa between y_true and (proba>=thr).
    2. If all kappa are NaN or weird, fallback to ROC harmonic mean balance.
    """
    kappas = []
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        kappas.append(cohen_kappa_score(y_true, pred))
    kappas = np.array(kappas, dtype=float)

    if np.isfinite(kappas).any():
        best_idx = int(np.nanargmax(kappas))
        return float(thresholds[best_idx]), "kappa"

    h_scores = []
    for thr in thresholds:
        pred = (proba >= thr).astype(int)
        h_scores.append(_roc_hmean(y_true, pred))
    h_scores = np.array(h_scores, dtype=float)
    best_idx = int(np.argmax(h_scores))
    return float(thresholds[best_idx]), "roc_hmean"


# ---------------------------------------------------------------------
# Main training / tuning routine using RandomizedSearchCV
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="RandomizedSearch HGB trainer w/ transcript norm + PR-AUC scoring + GHOST eval"
    )
    ap.add_argument("--json", required=True,
                    help="Path to dataset0 json / json.gz (features live here)")
    ap.add_argument("--labels", required=True,
                    help="Path to data.info.labelled CSV (has gene_id, labels)")
    ap.add_argument("--out_model", default="hgb_plus_best_train.joblib",
                    help="Output .joblib bundle")
    ap.add_argument("--tuning_log", default="tuning_log_hgb_plus.csv",
                    help="CSV log of sampled hyperparams and CV scores")
    ap.add_argument("--test_preds", default="test_predictions_hgb_plus.csv",
                    help="CSV of per-site predictions on held-out test split")
    ap.add_argument("--cv_folds", type=int, default=3,
                    help="StratifiedGroupKFold folds for RandomizedSearchCV")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_iter", type=int, default=50,
                    help="How many random hyperparam samples to try")
    args = ap.parse_args()

    np.random.seed(args.random_state)

    # -------------------------------------------------
    # 1. Load + feature engineering
    # -------------------------------------------------
    labels = load_labels(args.labels)
    keys, X, y, groups = build_xy_keys_enhanced(args.json, labels)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"[info] total samples={len(y)} | pos={n_pos} neg={n_neg}")
    if n_pos == 0 or n_neg == 0:
        raise SystemExit(f"Need both classes; got pos={n_pos}, neg={n_neg}.")

    # -------------------------------------------------
    # 2. Holdout split (80/20) by gene_id
    #    => no gene leaks between train/test
    # -------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=args.random_state)
    (tr_idx, te_idx), = gss.split(X, y, groups=groups)

    X_tr, y_tr, groups_tr = X[tr_idx], y[tr_idx], groups[tr_idx]
    X_te, y_te, groups_te = X[te_idx], y[te_idx], groups[te_idx]
    keys_te = [keys[i] for i in te_idx]

    print(
        f"[split] train={len(y_tr)} (pos={int((y_tr==1).sum())})  "
        f"test={len(y_te)} (pos={int((y_te==1).sum())})"
    )

    # -------------------------------------------------
    # 3. RandomizedSearchCV over HistGradientBoostingClassifier
    #    Scoring metric: PR-AUC (Average Precision)
    #    CV: StratifiedGroupKFold on TRAIN ONLY
    #    We'll also pass sample_weight='balanced' to fight imbalance.
    # -------------------------------------------------
    sgkf = StratifiedGroupKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state
    )

    base_model = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,   # keep deterministic / same #iters per trial
        random_state=args.random_state,
    )

    # Hyperparam distributions:
    # (randomly sample combos from these lists)
    param_dist = {
        "learning_rate":      [0.03, 0.05, 0.1],
        "max_leaf_nodes":     [31, 63, 127],
        "min_samples_leaf":   [5, 10, 20],
        "max_iter":           [100, 200, 300],
        "l2_regularization":  [0.0, 1e-3, 1e-2],
    }

    # We'll optimize PR-AUC
    pr_scorer = make_scorer(average_precision_score, needs_proba=True)

    # Class weights (balanced) for TRAIN split
    w_tr = compute_sample_weight(class_weight="balanced", y=y_tr)

    rand_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring=pr_scorer,
        cv=sgkf,
        # n_jobs = -1 lets it parallelize candidates across CPUs
        n_jobs=-1,
        verbose=2,
        refit=True,  # refit best model on full TRAIN after search
        random_state=args.random_state,
    )

    t0_rs = time.time()
    rand_search.fit(
        X_tr,
        y_tr,
        groups=groups_tr,
        sample_weight=w_tr  # passed to each fold fit()
    )
    t1_rs = time.time()

    print(f"[tuning done] time={t1_rs - t0_rs:.1f}s")

    best_params = rand_search.best_params_
    best_cv_pr = float(rand_search.best_score_)
    print(f"[best params] {best_params}")
    print(f"[best mean CV PR-AUC] {best_cv_pr:.4f}")

    # Save full CV results to CSV so you can inspect later
    cv_df = pd.DataFrame(rand_search.cv_results_)
    # sort by best score (higher PR-AUC is better)
    cv_df = cv_df.sort_values("mean_test_score", ascending=False)
    cv_df.to_csv(args.tuning_log, index=False)
    print(f"[tuning log] wrote → {args.tuning_log}")

    # After refit=True, best_estimator_ is already re-trained on all X_tr,y_tr
    best_model = rand_search.best_estimator_

    # -------------------------------------------------
    # 4. Evaluate on untouched 20% TEST split
    # -------------------------------------------------
    p_tr = best_model.predict_proba(X_tr)[:, 1]
    p_te = best_model.predict_proba(X_te)[:, 1]

    pr_auc_te = average_precision_score(y_te, p_te)
    roc_auc_te = roc_auc_score(y_te, p_te)
    print(f"[test] PR-AUC={pr_auc_te:.4f} | ROC-AUC={roc_auc_te:.4f}")

    # GHOST threshold tuning on TRAIN ONLY,
    # then apply that threshold to TEST to inspect confusion matrix style behavior.
    ghost_thr, ghost_metric = ghost_threshold_tuning_probs(y_tr, p_tr)
    print(f"[ghost] chosen threshold={ghost_thr:.2f} via {ghost_metric}")

    y_te_pred_ghost = (p_te >= ghost_thr).astype(int)

    print("\nClassification report @GHOST threshold (inspection only):")
    print(classification_report(y_te, y_te_pred_ghost, digits=3))

    # -------------------------------------------------
    # 5. Save per-site predictions on TEST split
    # -------------------------------------------------
    df_test = pd.DataFrame({
        "transcript_id": [k[0] for k in keys_te],
        "transcript_position": [k[1] for k in keys_te],
        "true_label": y_te,
        "predicted_label_ghost": y_te_pred_ghost,
        "score": p_te
    }).sort_values(["transcript_id", "transcript_position"])

    df_test.to_csv(args.test_preds, index=False)
    print(f"[predictions] wrote → {args.test_preds}")

    # -------------------------------------------------
    # 6. Save bundle (model + metadata)
    # -------------------------------------------------
    bundle = {
        "model": best_model,
        "feature_dim": int(X.shape[1]),
        "best_params": best_params,
        "cv_mean_pr_auc": best_cv_pr,
        "test_pr_auc": float(pr_auc_te),
        "test_roc_auc": float(roc_auc_te),
        "ghost_threshold": float(ghost_thr),
        "ghost_metric": ghost_metric,
        "random_state": args.random_state,
        "feature_notes": {
            "final_dim": int(X.shape[1]),
            "includes_transcript_centering": True,
            "comment": "X_full = [base_features | transcript_centered | motif_onehot]"
        }
    }

    joblib.dump(bundle, args.out_model)
    print(f"[model saved] → {args.out_model}")


if __name__ == "__main__":
    main()