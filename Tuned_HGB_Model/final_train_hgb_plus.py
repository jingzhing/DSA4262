import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from json_parse_plus import parse_enhanced

# ---------- helpers ----------

def load_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    need = {'gene_id','transcript_id','transcript_position','label'}
    if not need.issubset(df.columns):
        raise ValueError(f"labels CSV must contain: {need}")
    df = df[['gene_id','transcript_id','transcript_position','label']].copy()
    df['transcript_position'] = df['transcript_position'].astype(int)
    df['label'] = df['label'].astype(int)
    return df

def build_xy_keys_enhanced(json_path: str, labels: pd.DataFrame):
    """
    Same feature construction logic as train_eval_hgb_plus.py:
    - enhanced per-site stats (76 dims)
    - transcript-centered deltas (76 dims)
    - motif one-hot (18 dims)
    -> concat to (N, 170)
    """
    # match keys to labeled sites
    keyset = {(t, int(p)) for t, p in zip(labels.transcript_id, labels.transcript_position)}
    keys, X_base, X_motif = parse_enhanced(json_path, restrict_keys=keyset)

    if len(keys) == 0:
        raise SystemExit("No overlap between JSON sites and labels.")

    # map label + gene_id
    lab_map = {
        (r.transcript_id, int(r.transcript_position)): (int(r.label), r.gene_id)
        for r in labels.itertuples(index=False)
    }

    y = np.array([lab_map[k][0] for k in keys], dtype=int)
    transcript_ids = np.array([k[0] for k in keys])

    # transcript-level centering
    unique_t = np.unique(transcript_ids)
    t_means = {}
    for t_id in unique_t:
        mask = (transcript_ids == t_id)
        t_means[t_id] = X_base[mask].mean(axis=0)  # (76,)

    X_norm = np.vstack([
        X_base[i] - t_means[transcript_ids[i]]
        for i in range(len(transcript_ids))
    ]).astype(np.float32)  # (N,76)

    X_full = np.concatenate([X_base, X_norm, X_motif], axis=1).astype(np.float32)
    return keys, X_full, y

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Final production HGB (enhanced features) trained on ALL labeled data"
    )
    ap.add_argument("--json", required=True,
                    help="Path to dataset0.json(.gz)")
    ap.add_argument("--labels", required=True,
                    help="Path to data.info.labelled CSV")
    ap.add_argument("--out_model", default="hgb_plus_final.joblib",
                    help="Output .joblib bundle")

    # best hyperparams from train_eval_hgb_plus.py:
    ap.add_argument("--learning_rate", type=float, required=True)
    ap.add_argument("--max_leaf_nodes", type=int, required=True)
    ap.add_argument("--min_samples_leaf", type=int, required=True)
    ap.add_argument("--max_iter", type=int, required=True)
    ap.add_argument("--l2_regularization", type=float, required=True)
    ap.add_argument("--pos_boost", type=float, required=True,
                    help="Extra weight multiplier for positive class, from tuning")

    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.random_state)

    # ----- load ALL labeled data -----
    labels = load_labels(args.labels)
    keys_all, X_all, y_all = build_xy_keys_enhanced(args.json, labels)

    n_pos = int((y_all == 1).sum())
    n_neg = int((y_all == 0).sum())
    print(f"[final-train] total labeled sites={len(y_all)} (pos={n_pos}, neg={n_neg})")

    # ----- build sample weights using the tuned pos_boost -----
    w_all = compute_sample_weight(class_weight="balanced", y=y_all)
    w_all[y_all == 1] *= args.pos_boost

    # ----- train final model on ALL data -----
    final_model = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,
        random_state=args.random_state,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        max_iter=args.max_iter,
        l2_regularization=args.l2_regularization,
    )
    final_model.fit(X_all, y_all, sample_weight=w_all)

    # ----- save bundle -----
    bundle = {
        "model": final_model,
        "feature_dim": int(X_all.shape[1]),
        "hyperparams": {
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "min_samples_leaf": args.min_samples_leaf,
            "max_iter": args.max_iter,
            "l2_regularization": args.l2_regularization,
            "pos_boost": args.pos_boost,
            "random_state": args.random_state
        },
        "feature_notes": {
            "enhanced_stats_dim": 76,
            "transcript_centered_dim": 76,
            "motif_dim": 18,
            "total_dim": int(X_all.shape[1]),
            "uses_transcript_normalization": True,
            "uses_enhanced_heterogeneity_features": True
        }
    }

    joblib.dump(bundle, args.out_model)
    print(f"[done] saved final production model → {args.out_model}")


if __name__ == "__main__":
    main()