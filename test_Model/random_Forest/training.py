import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from json_parse import parse

def main():
    ap = argparse.ArgumentParser(description="Train m6A site classifier with grouped 5-fold calibration")
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_model", default="model.joblib")
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--min_samples_leaf", type=int, default=2)
    ap.add_argument("--cv_folds", type=int, default=5)
    args = ap.parse_args()

    lab = pd.read_csv(args.labels_csv)
    lab["key"] = list(zip(lab["transcript_id"], lab["transcript_position"].astype(int)))
    keyset = set(lab["key"].tolist())

    keys, X = parse(args.data_json, restrict_keys=keyset)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    lab = lab[lab["key"].isin(key_to_idx)].reset_index(drop=True)
    rows = [key_to_idx[k] for k in lab["key"]]
    X = X[rows]
    y = lab["label"].astype(int).to_numpy()
    groups = lab["transcript_id"].to_numpy()

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_transcripts = int(np.unique(groups).size)
    print(f"[info] Samples: {len(y)} | Pos: {n_pos} | Neg: {n_neg} | Transcripts: {n_transcripts}")

    sw = compute_sample_weight(class_weight="balanced", y=y)

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=-1,
        random_state=42
    )

    sgkf = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    cv_splits = list(sgkf.split(X, y, groups))

    clf = CalibratedClassifierCV(rf, method="sigmoid", cv=cv_splits)
    clf.fit(X, y, sample_weight=sw)

    joblib.dump({"model": clf, "feature_dim": X.shape[1]}, args.out_model)
    print(f"[done] Saved model → {args.out_model}")

if __name__ == "__main__":
    main()
