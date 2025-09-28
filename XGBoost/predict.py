import argparse
import joblib
import pandas as pd
from json_parse import parse

def main():
    ap = argparse.ArgumentParser(description="Predict m6A modification probabilities (XGBoost)")
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # Load model
    bundle = joblib.load(args.model)
    model = bundle["model"]
    feat_dim = bundle["feature_dim"]

    # Parse features
    keys, X = parse(args.data_json)
    if X.shape[0] == 0:
        pd.DataFrame(columns=["transcript_id","transcript_position","score"]).to_csv(args.out_csv, index=False)
        print(f"Empty input → wrote header to {args.out_csv}")
        return
    if X.shape[1] != feat_dim:
        raise ValueError(f"Feature dim mismatch: model={feat_dim}, data={X.shape[1]}")

    # Predict probabilities
    scores = model.predict_proba(X)[:,1]
    df = pd.DataFrame({
        "transcript_id": [k[0] for k in keys],
        "transcript_position": [k[1] for k in keys],
        "score": scores
    }).sort_values(["transcript_id","transcript_position"])

    # Save CSV
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} predictions → {args.out_csv}")

if __name__ == "__main__":
    main()
