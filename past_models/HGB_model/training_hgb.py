import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from json_parse import parse as parse_json  

def load_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    need = {'gene_id','transcript_id','transcript_position','label'}
    if not need.issubset(df.columns):
        raise ValueError(f"labels CSV must contain: {need}")
    df = df[['gene_id','transcript_id','transcript_position','label']].copy()
    df['transcript_position'] = df['transcript_position'].astype(int)
    df['label'] = df['label'].astype(int)
    return df

def main():
    ap = argparse.ArgumentParser(description="Train calibrated HistGradientBoosting (Part 1)")
    ap.add_argument('--json', required=True, help='Path to data.json / .jsonl(.gz)')
    ap.add_argument('--labels', required=True, help='Path to data.info.labelled CSV')
    ap.add_argument('--out_model', default='hgb_model.joblib', help='Output model path')
    ap.add_argument('--out_scaler', default='scaler.joblib', help='Output scaler path (kept for predict.py compatibility)')
    ap.add_argument('--learning_rate', type=float, default=0.06)
    ap.add_argument('--max_leaf_nodes', type=int, default=63)
    ap.add_argument('--min_samples_leaf', type=int, default=20)
    ap.add_argument('--cv', type=int, default=5, help='Folds for isotonic calibration')
    args = ap.parse_args()

    # Load labels and restrict to labeled (tx, pos)
    labels = load_labels(args.labels)
    restrict = {(r.transcript_id, int(r.transcript_position)) for r in labels.itertuples(index=False)}
    y_map = {(r.transcript_id, int(r.transcript_position)): (int(r.label), r.gene_id)
             for r in labels.itertuples(index=False)}

    # Pull features via parser (already aggregated)
    keys, X = parse_json(args.json, restrict_keys=restrict)
    if len(keys) == 0:
        raise SystemExit("No overlap between JSON sites and labels.")

    y = np.array([y_map[k][0] for k in keys], dtype=int)
    groups = np.array([y_map[k][1] for k in keys])

    # Grouped split by gene_id → avoids leakage
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    (tr_idx, va_idx), = gss.split(X, y, groups=groups)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    # Base HGB (tabular boosting) + isotonic calibration for reliable probabilities
    hgb = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        validation_fraction=None
    )
    clf = CalibratedClassifierCV(hgb, method="isotonic", cv=args.cv)

    # Fit
    clf.fit(X_tr, y_tr)

    # Evaluate (threshold-free + 0.5 threshold report)
    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_va = clf.predict_proba(X_va)[:, 1]
    print(f"[train] ROC-AUC={roc_auc_score(y_tr, p_tr):.4f}  PR-AUC={average_precision_score(y_tr, p_tr):.4f}")
    print(f"[valid] ROC-AUC={roc_auc_score(y_va, p_va):.4f}  PR-AUC={average_precision_score(y_va, p_va):.4f}")
    print("Validation report @0.5 threshold:")
    print(classification_report(y_va, (p_va >= 0.5).astype(int), digits=3))

    # Save artifacts
    joblib.dump(clf, args.out_model)
    # Save a scaler only for compatibility with existing predict.py scripts (HGB itself doesn't need it)
    scaler = StandardScaler().fit(X_tr)
    joblib.dump(scaler, args.out_scaler)
    # Convenience: full pipeline
    joblib.dump(Pipeline([("scaler", scaler), ("clf", clf)]), "hgb_pipeline.joblib")

    print(f"Saved: {args.out_model}, {args.out_scaler}, and hgb_pipeline.joblib")

if __name__ == "__main__":
    main()
