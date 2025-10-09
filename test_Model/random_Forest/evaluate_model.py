import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from json_parse import parse
from sklearn.feature_selection import f_classif

def main():
    ap = argparse.ArgumentParser(description="Train/test split evaluation for m6A classifier")
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_model", default="model.joblib")
    ap.add_argument("--cv_folds", type=int, default=5)
    args = ap.parse_args()

    lab = pd.read_csv(args.labels_csv)
    lab["key"] = list(zip(lab["transcript_id"], lab["transcript_position"].astype(int)))
    keyset = set(lab["key"].tolist())
    keys, X = parse(args.data_json, restrict_keys=keyset)
    key_to_idx = {k:i for i,k in enumerate(keys)}
    lab = lab[lab["key"].isin(key_to_idx)].reset_index(drop=True)
    rows = [key_to_idx[k] for k in lab["key"]]
    X = X[rows]
    y = lab["label"].astype(int).to_numpy()
    groups = lab["transcript_id"].to_numpy()

    print(f"[info] Total samples: {len(y)}, positives: {(y==1).sum()}, negatives: {(y==0).sum()}")

    sgkf = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    print(f"[info] Train: {len(y_train)} | Test: {len(y_test)}")

    sw = compute_sample_weight(class_weight="balanced", y=y_train)

    rf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )

    clf = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
    clf.fit(X_train, y_train, sample_weight=sw)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n[Metrics]")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print("Confusion Matrix:\n", cm)

    f_vals, p_vals = f_classif(X, y)
    feature_df = pd.DataFrame({
        "Feature_Index": np.arange(X.shape[1]),
        "F_Value": f_vals,
        "P_Value": p_vals
    }).sort_values("F_Value", ascending=False)
    print("\n[Top 10 features by F-value]")
    print(feature_df.head(10).to_string(index=False))

    joblib.dump({"model": clf, "feature_dim": X.shape[1]}, args.out_model)
    print(f"\n[done] Saved trained model → {args.out_model}")

if __name__ == "__main__":
    main()
