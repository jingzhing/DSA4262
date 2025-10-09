import argparse
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from json_parse import parse

def main():
    ap = argparse.ArgumentParser(description="Train XGBoost m6A site classifier")
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_model", default="xgb_model.joblib")
    args = ap.parse_args()

    # Load Labels
    lab = pd.read_csv(args.labels_csv)
    lab["key"] = list(zip(lab["transcript_id"], lab["transcript_position"].astype(int)))
    keyset = set(lab["key"].tolist())

    # Parse JSON -> features
    keys, X = parse(args.data_json, restrict_keys=keyset)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    lab = lab[lab["key"].isin(key_to_idx)].reset_index(drop=True)
    rows = [key_to_idx[k] for k in lab["key"]]
    X = X[rows]
    y = lab["label"].astype(int).to_numpy()

    # Imbalance handling
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"[info] Samples: {len(y)} | Pos: {n_pos} | Neg: {n_neg}")

    # Train XGBoost
    model = XGBClassifier(
        n_estimators = 400,
        max_depth = 6,
        learning_rate = 0.1,
        subsample = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos_weight,
        use_label_encoder = False,
        eval_metric = "logloss",
        random_state = 42,
        n_jobs = -1
    )

    sample_weights = compute_sample_weight(class_weight="balanced", y=y)
    model.fit(X, y, sample_weight=sample_weights)

    #Save Model
    joblib.dump({"model": model, "feature_dim": X.shape[1]}, args.out_model)
    print(f"[done] Saved model -> {args.out_model}")

if __name__ == "__main__":
    main()