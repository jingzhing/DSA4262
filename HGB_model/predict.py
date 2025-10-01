# predict.py
import numpy as np
import pandas as pd
import joblib
from json_parse import parse
import argparse


# Argument parser
parser = argparse.ArgumentParser(description="Predict m6A modification probabilities")
parser.add_argument("--json", required=True, help="Input JSON.gz file")
parser.add_argument("--model", default="logreg_model.joblib", help="Trained model file")
parser.add_argument("--scaler", default="scaler.joblib", help="Feature scaler file")
parser.add_argument("--output", default="predictions.csv", help="Output CSV file")
args = parser.parse_args()

# Load model and scaler
clf = joblib.load(args.model)
scaler = joblib.load(args.scaler)

# Parse JSON features
keys, X = parse(args.json)
X = np.array(X)

print(f"Loaded {len(keys)} sites from {args.json}")

# Scale features
X_scaled = scaler.transform(X)

y_proba = clf.predict_proba(X_scaled)[:, 1]

# Prepare output
df_out = pd.DataFrame({
    "transcript_id": [k[0] for k in keys],
    "transcript_position": [k[1] for k in keys],
    "score": y_proba
})

# Ensure CSV format matches requirements
df_out.to_csv(args.output, index=False)
print(f"✅ Predictions saved to {args.output}")