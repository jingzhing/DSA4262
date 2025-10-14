# run_script.py
# Split (by gene), optional tune HGB+calibration on an inner split, retrain on full train,
# predict on test, evaluate, and save artifacts.
# run_script.py
# Outer split (by gene), optional inner tuning (PR-AUC), oversampling, class_weight,
# retrain on full train, optional bagging ensemble, evaluate on test, save artifacts.
# run_script.py
# Outer split (by gene), optional inner tuning (PR-AUC), class_weight, optional oversampling,
# optional cost-sensitive sample_weight, train HGB and/or XGB (optionally both), ensemble, blend, evaluate, save.
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit

from json_parse import parse as parse_json

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k): return x


def load_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    need = {"gene_id", "transcript_id", "transcript_position", "label"}
    if not need.issubset(df.columns):
        raise ValueError(f"labels CSV must contain: {need}")
    df = df[["gene_id", "transcript_id", "transcript_position", "label"]].copy()
    df["transcript_position"] = df["transcript_position"].astype(int)
    df["label"] = df["label"].astype(int)
    return df


def map_indices(df: pd.DataFrame, key_to_feat_idx: dict, desc: str):
    feat_idx, row_idx = [], []
    for i, r in tqdm(enumerate(df.itertuples(index=False)), total=len(df), desc=desc, unit="rows"):
        k = (r.transcript_id, int(r.transcript_position))
        j = key_to_feat_idx.get(k)
        if j is not None:
            feat_idx.append(j)
            row_idx.append(i)
    return np.array(feat_idx, dtype=int), np.array(row_idx, dtype=int)


def build_hgb(lr: float, nodes: int, mleaf: int, cv_folds: int, seed: int, class_weight: str | None):
    # You can expand with max_iter/l2/max_bins later if desired.
    hgb = HistGradientBoostingClassifier(
        learning_rate=lr,
        max_leaf_nodes=nodes,
        min_samples_leaf=mleaf,
        validation_fraction=None,
        random_state=seed,
        class_weight=class_weight,  # "balanced" or None
    )
    return CalibratedClassifierCV(hgb, method="isotonic", cv=cv_folds)


def build_xgb(cv_folds: int, seed: int, scale_pos_weight: float, params: dict):
    try:
        from xgboost import XGBClassifier
    except Exception:
        raise SystemExit("xgboost is not installed. `pip install xgboost` or run without --blend_xgb/--use_xgb.")
    xgb = XGBClassifier(
        n_estimators=params.get("n_estimators", 500),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 8),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        random_state=seed,
        n_jobs=-1,
        tree_method=params.get("tree_method", "hist"),  # set "gpu_hist" if you want GPU
        scale_pos_weight=params.get("scale_pos_weight", scale_pos_weight),
        eval_metric="logloss",
    )
    return CalibratedClassifierCV(xgb, method="isotonic", cv=cv_folds)


def tune_hgb_inner(X_tr, y_tr, groups_tr, cv_folds, seed, grid, class_weight):
    print("[TUNE] Creating inner train/valid split by gene_id…")
    gss_inner = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed + 1)
    (inner_tr_idx, inner_va_idx), = gss_inner.split(X_tr, y_tr, groups=groups_tr)
    X_in_tr, y_in_tr = X_tr[inner_tr_idx], y_tr[inner_tr_idx]
    X_in_va, y_in_va = X_tr[inner_va_idx], y_tr[inner_va_idx]

    combos = [
        (lr, nodes, mleaf)
        for lr in grid["learning_rate"]
        for nodes in grid["max_leaf_nodes"]
        for mleaf in grid["min_samples_leaf"]
    ]
    best = {"pr_auc": -1.0, "roc_auc": -1.0, "params": None}

    print(f"[TUNE] Searching {len(combos)} configs:")
    for (lr, nodes, mleaf) in tqdm(combos, desc="Tuning HGB", unit="cfg"):
        clf = build_hgb(lr, nodes, mleaf, cv_folds, seed, class_weight)
        clf.fit(X_in_tr, y_in_tr)
        p_va = clf.predict_proba(X_in_va)[:, 1]
        pr_auc = average_precision_score(y_in_va, p_va)
        roc_auc = roc_auc_score(y_in_va, p_va)
        print(f"       lr={lr:<5} nodes={nodes:<4} min_leaf={mleaf:<3} → ROC_AUC={roc_auc:.4f}, PR_AUC={pr_auc:.4f}")
        if pr_auc > best["pr_auc"]:
            best.update({"pr_auc": pr_auc, "roc_auc": roc_auc, "params": (lr, nodes, mleaf)})

    print(f"[TUNE] Best params: lr={best['params'][0]}, nodes={best['params'][1]}, "
          f"min_leaf={best['params'][2]} (PR-AUC={best['pr_auc']:.4f}, ROC-AUC={best['roc_auc']:.4f})")
    return best["params"]


def simple_oversample(X, y, factor: int):
    """Duplicate positive class 'factor' times (factor>=1 means add 'factor-1' copies)."""
    if factor <= 1:
        return X, y
    pos_idx = np.where(y == 1)[0]
    if len(pos_idx) == 0:
        return X, y
    add_idx = np.random.choice(pos_idx, size=(factor - 1) * len(pos_idx), replace=True)
    X_bal = np.concatenate([X, X[add_idx]], axis=0)
    y_bal = np.concatenate([y, y[add_idx]], axis=0)
    return X_bal, y_bal


def fit_models_for_kind(kind, n_models, seed0, X_fit, y_fit, cv_folds, class_weight, hgb_params, xgb_params, pos_weight):
    """Train n_models of a given kind ('hgb' or 'xgb'), return list of calibrated classifiers."""
    models = []
    # Sample weights (cost-sensitive), applied identically to all models of this kind
    sample_weight = None
    if pos_weight and pos_weight > 0:
        sample_weight = np.where(y_fit == 1, float(pos_weight), 1.0).astype(np.float32)

    # For XGB, auto scale_pos_weight if not provided
    neg = int((y_fit == 0).sum())
    pos = int((y_fit == 1).sum())
    default_spw = float(neg) / float(max(pos, 1))

    for i in range(n_models):
        seed_i = seed0 + i
        if kind == "hgb":
            lr, nodes, mleaf = hgb_params["lr"], hgb_params["nodes"], hgb_params["mleaf"]
            clf = build_hgb(lr, nodes, mleaf, cv_folds=cv_folds, seed=seed_i, class_weight=class_weight)
        elif kind == "xgb":
            # Ensure scale_pos_weight present
            xgbp = dict(xgb_params) if xgb_params else {}
            xgbp.setdefault("scale_pos_weight", default_spw)
            clf = build_xgb(cv_folds=cv_folds, seed=seed_i, scale_pos_weight=xgbp["scale_pos_weight"], params=xgbp)
        else:
            raise ValueError(f"Unknown model kind: {kind}")

        if sample_weight is not None:
            clf.fit(X_fit, y_fit, sample_weight=sample_weight)
        else:
            clf.fit(X_fit, y_fit)
        models.append(clf)
    return models


def main():
    ap = argparse.ArgumentParser(description="Split/train/eval with tuning, cost-sensitive weights, optional oversampling, HGB and/or XGB, ensembling, blending")
    ap.add_argument("--json", required=True, help="Path to data.json(.gz)")
    ap.add_argument("--labels", required=True, help="Path to data.info.labelled CSV")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--train_size", type=float, default=0.8, help="Train fraction by gene_id for outer split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=5, help="Folds for isotonic calibration")

    # Imbalance controls
    ap.add_argument("--class_weight", type=str, default="balanced", choices=["balanced", "none"])
    ap.add_argument("--oversample_factor", type=int, default=1, help="Duplicate positives this many times (>=1)")
    ap.add_argument("--pos_weight", type=float, default=0.0, help="Cost-sensitive weight for positives in fit(); 0 disables")

    # HGB tuning
    ap.add_argument("--tune", action="store_true", help="Enable HGB hyperparameter tuning on inner split")
    ap.add_argument("--grid_lrs", type=str, default="0.02,0.03,0.06,0.1")
    ap.add_argument("--grid_nodes", type=str, default="31,63,127,255")
    ap.add_argument("--grid_minleaf", type=str, default="5,10,20,40")

    # Model selection
    ap.add_argument("--use_xgb", action="store_true", help="Use only XGBoost (instead of HGB)")
    ap.add_argument("--blend_xgb", action="store_true", help="Train BOTH HGB and XGBoost, blend their probabilities")
    ap.add_argument("--xgb_params", type=str, default="", help='JSON string of XGB params (e.g. {"n_estimators":800,"tree_method":"gpu_hist"})')

    # Ensembling
    ap.add_argument("--ensemble_n", type=int, default=1, help="Number of models per kind (seeds = seed..seed+N-1)")

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/9] Loading labels and outer split by gene_id…")
    labels = load_labels(args.labels)
    gss_outer = GroupShuffleSplit(n_splits=1, train_size=args.train_size, random_state=args.seed)
    (tr_idx, te_idx), = gss_outer.split(labels, groups=labels["gene_id"])
    train_labels = labels.iloc[tr_idx].reset_index(drop=True)
    test_labels  = labels.iloc[te_idx].reset_index(drop=True)
    (outdir / "train_labels.csv").write_text(train_labels.to_csv(index=False))
    (outdir / "test_labels.csv").write_text(test_labels.to_csv(index=False))
    print(f"[SPLIT] Train={len(train_labels)}  Test={len(test_labels)}")

    print("[2/9] Parsing JSON/features… (this may take a while)")
    all_keys, X_all = parse_json(args.json)
    key_to_feat_idx = {k: i for i, k in enumerate(all_keys)}
    print(f"[PARSE] Total feature rows: {len(all_keys)}")

    print("[3/9] Matching labels to features…")
    tr_feat_idx, tr_row_idx = map_indices(train_labels, key_to_feat_idx, "Map train")
    te_feat_idx, te_row_idx = map_indices(test_labels,  key_to_feat_idx, "Map test")
    if len(tr_feat_idx) == 0 or len(te_feat_idx) == 0:
        raise SystemExit("No overlap between JSON sites and split labels. Check inputs.")

    X_tr_all, y_tr_all = X_all[tr_feat_idx], train_labels.iloc[tr_row_idx]["label"].to_numpy()
    X_te, y_te = X_all[te_feat_idx], test_labels.iloc[te_row_idx]["label"].to_numpy()
    groups_tr_all = train_labels.iloc[tr_row_idx]["gene_id"].to_numpy()
    print(f"[MATCH] Train feats: {len(X_tr_all)} | Test feats: {len(X_te)}")

    # Class weight & pos_weight
    class_weight = None if args.class_weight == "none" else "balanced"

    # Compute default scale_pos_weight for XGB (neg/pos)
    pos = (y_tr_all == 1).sum()
    neg = (y_tr_all == 0).sum()
    default_spw = float(neg) / float(max(pos, 1))

    # HGB tuning (if HGB is involved)
    use_hgb = not args.use_xgb or args.blend_xgb  # True if HGB is used (alone or blended)
    if args.tune and use_hgb:
        grid = {
            "learning_rate": [float(x) for x in args.grid_lrs.split(",") if x.strip()],
            "max_leaf_nodes": [int(x) for x in args.grid_nodes.split(",") if x.strip()],
            "min_samples_leaf": [int(x) for x in args.grid_minleaf.split(",") if x.strip()],
        }
        print(f"[4/9] Hyperparameter tuning (HGB) with grid: {grid}")
        best_lr, best_nodes, best_mleaf = tune_hgb_inner(
            X_tr=X_tr_all, y_tr=y_tr_all, groups_tr=groups_tr_all,
            cv_folds=args.cv, seed=args.seed, grid=grid, class_weight=class_weight,
        )
    else:
        best_lr, best_nodes, best_mleaf = 0.06, 127, 10
        print(f"[4/9] HGB tuning skipped or HGB disabled. Using defaults: lr={best_lr}, nodes={best_nodes}, min_leaf={best_mleaf}")

    # Oversampling (optionally); for clean pos_weight comparison, set oversample_factor=1
    print("[5/9] Applying simple oversampling (if any)…")
    X_tr_bal, y_tr_bal = simple_oversample(X_tr_all, y_tr_all, args.oversample_factor)

    # Prepare params
    hgb_params = {"lr": best_lr, "nodes": best_nodes, "mleaf": best_mleaf}
    xgb_params = {}
    if args.xgb_params.strip():
        try:
            xgb_params = json.loads(args.xgb_params)
        except Exception:
            print("[WARN] Could not parse --xgb_params JSON. Using defaults.")
    xgb_params.setdefault("scale_pos_weight", default_spw)

    # Decide which kinds to train
    kinds = []
    if args.use_xgb and not args.blend_xgb:
        kinds = ["xgb"]
    elif args.blend_xgb:
        kinds = ["hgb", "xgb"]
    else:
        kinds = ["hgb"]

    # Train
    print("[6/9] Training model(s)…")
    models = []
    for kind in kinds:
        mods = fit_models_for_kind(
            kind=kind,
            n_models=args.ensemble_n,
            seed0=args.seed,
            X_fit=X_tr_bal,
            y_fit=y_tr_bal,
            cv_folds=args.cv,
            class_weight=class_weight,
            hgb_params=hgb_params,
            xgb_params=xgb_params,
            pos_weight=args.pos_weight,
        )
        models.extend(mods)

    # Evaluate with blended ensemble
    print("[7/9] Evaluating on held-out test…")
    if len(models) == 1:
        p_tr = models[0].predict_proba(X_tr_all)[:, 1]
        p_te = models[0].predict_proba(X_te)[:, 1]
    else:
        p_tr = np.mean([m.predict_proba(X_tr_all)[:, 1] for m in models], axis=0)
        p_te = np.mean([m.predict_proba(X_te)[:, 1] for m in models], axis=0)

    roc = roc_auc_score(y_te, p_te)
    pr  = average_precision_score(y_te, p_te)

    prec, rec, thr = precision_recall_curve(y_tr_all, p_tr)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best_t = thr[np.argmax(f1[:-1])] if len(thr) else 0.5
    yhat_te = (p_te >= best_t).astype(int)

    print(f"[TEST] ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")
    print(f"[THRESH] F1-opt (from train) = {best_t:.4f}")
    print(classification_report(y_te, yhat_te, digits=3))

    # Save
    print("[8/9] Saving artifacts & outputs…")
    if len(models) == 1:
        joblib.dump(models[0], outdir / "hgb_model.joblib")
        bundle_type = "single"
    else:
        bundle = {"type": "ensemble", "models": models}
        joblib.dump(bundle, outdir / "hgb_model.joblib")
        bundle_type = "ensemble"

    pd.DataFrame({
        "transcript_id": [all_keys[i][0] for i in te_feat_idx],
        "transcript_position": [all_keys[i][1] for i in te_feat_idx],
        "score": p_te,
        "label": y_te,
    }).to_csv(outdir / "test_predictions.csv", index=False)

    meta = {
        "kinds_trained": kinds,
        "model_bundle": bundle_type,
        "tuned_hgb": bool(args.tune and ("hgb" in kinds)),
        "hgb_params": {"learning_rate": best_lr, "max_leaf_nodes": best_nodes, "min_samples_leaf": best_mleaf} if "hgb" in kinds else None,
        "xgb_params": xgb_params if "xgb" in kinds else None,
        "outer_train_size": float(args.train_size),
        "cv_folds_calibration": int(args.cv),
        "seed": int(args.seed),
        "class_weight": args.class_weight,
        "oversample_factor": int(args.oversample_factor),
        "pos_weight": float(args.pos_weight),
        "ensemble_n": int(args.ensemble_n),
        "metrics": {"test_roc_auc": float(roc), "test_pr_auc": float(pr), "f1_opt_threshold_from_train": float(best_t)},
    }
    with open(outdir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(outdir / "metrics.txt", "w") as f:
        f.write(f"TEST ROC-AUC: {roc:.6f}\nTEST PR-AUC: {pr:.6f}\n")
        f.write(f"THRESH (F1-opt from train): {best_t:.6f}\n")
        if "hgb" in kinds:
            f.write(f"HGB PARAMS USED: lr={best_lr}, nodes={best_nodes}, min_leaf={best_mleaf}\n")
        if "xgb" in kinds:
            f.write(f"XGB PARAMS USED: {json.dumps(xgb_params)}\n")

    print(f"[SAVED] {outdir}/hgb_model.joblib, test_predictions.csv, metrics.txt, run_meta.json, train/test labels")
    print("[9/9] Done.")


if __name__ == "__main__":
    main()

#python run_script.py --json datasets\dataset0.json --labels datasets\data.info.labelled --outdir runs\exp_xgb_gpu_blend --blend_xgb --tune --class_weight balanced --oversample_factor 1 --pos_weight 4 --ensemble_n 5 --xgb_params "{\"tree_method\":\"gpu_hist\",\"n_estimators\":800,\"max_depth\":8,\"learning_rate\":0.05,\"subsample\":0.8,\"colsample_bytree\":0.8}" --grid_lrs 0.02,0.03,0.06,0.1 --grid_nodes 31,63,127,255 --grid_minleaf 5,10,20,40
