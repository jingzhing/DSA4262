# run_both_presets.py
# Train + evaluate two presets (A & B) back-to-back using run_script.py
# - Preset A: PR-AUC focused (no oversampling, pos_weight=3)
# - Preset B: Higher recall (oversample x2, pos_weight=3)
#
# Both produce: predictions.csv (submission format) and test_predictions_with_labels.csv (diagnostics)
# Requires: run_script.py (your updated one), json_parse.py, xgboost installed if blending, CUDA optional.

import argparse, json, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Run presets A and B in one go")
    ap.add_argument("--json", required=True, help="Path to data.json(.gz)")
    ap.add_argument("--labels", required=True, help="Path to data.info.labelled CSV")
    ap.add_argument("--outroot", required=True, help="Output root directory; subfolders will be created for each preset")
    ap.add_argument("--ensemble_n", type=int, default=5, help="Ensemble size per model kind")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--tune", action="store_true", help="Enable HGB tuning (recommended)")
    ap.add_argument("--device", default="cuda", help="XGBoost device (cuda or cpu)")
    args = ap.parse_args()

    outroot = Path(args.outroot)
    outA = outroot / "presetA_prauc"
    outB = outroot / "presetB_recall"
    outA.mkdir(parents=True, exist_ok=True)
    outB.mkdir(parents=True, exist_ok=True)

    # Shared XGB params (GPU by default)
    xgb_params = {
        "device": args.device,               # "cuda" (XGBoost >= 2.0) or "cpu"
        "n_estimators": 1200,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    xgb_json = json.dumps(xgb_params)

    # Shared HGB tuning grids
    grid_lrs = "0.02,0.03,0.06,0.1"
    grid_nodes = "31,63,127,255"
    grid_minleaf = "5,10,20,40"

    # ---------- Preset A: PR-AUC focused ----------
    cmdA = [
        sys.executable, "HGB_model/run_script.py",
        "--json", args.json,
        "--labels", args.labels,
        "--outdir", str(outA),
        "--class_weight", "balanced",
        "--oversample_factor", "1",
        "--pos_weight", "3",
        "--ensemble_n", str(args.ensemble_n),
        "--blend_xgb",
        "--xgb_params", xgb_json,
        "--seed", str(args.seed),
        "--cv", str(args.cv),
        "--grid_lrs", grid_lrs,
        "--grid_nodes", grid_nodes,
        "--grid_minleaf", grid_minleaf,
    ]
    if args.tune:
        cmdA.append("--tune")

    print("\n=== Running PRESET A (PR-AUC) ===")
    print(" ".join(cmdA))
    subprocess.run(cmdA, check=True)

    # ---------- Preset B: Higher recall ----------
    cmdB = [
        sys.executable, "HGB_model/run_script.py",
        "--json", args.json,
        "--labels", args.labels,
        "--outdir", str(outB),
        "--class_weight", "balanced",
        "--oversample_factor", "2",   # <-- oversample x2
        "--pos_weight", "3",
        "--ensemble_n", str(args.ensemble_n),
        "--blend_xgb",
        "--xgb_params", xgb_json,
        "--seed", str(args.seed),
        "--cv", str(args.cv),
        "--grid_lrs", grid_lrs,
        "--grid_nodes", grid_nodes,
        "--grid_minleaf", grid_minleaf,
    ]
    if args.tune:
        cmdB.append("--tune")

    print("\n=== Running PRESET B (Recall) ===")
    print(" ".join(cmdB))
    subprocess.run(cmdB, check=True)

    print("\nDone.\nOutputs:")
    print(f"  - Preset A folder: {outA}")
    print(f"      • predictions.csv (submission format)")
    print(f"      • test_predictions_with_labels.csv (diagnostics)")
    print(f"      • hgb_model.joblib (ensemble bundle)")
    print(f"  - Preset B folder: {outB}")
    print(f"      • predictions.csv (submission format)")
    print(f"      • test_predictions_with_labels.csv (diagnostics)")
    print(f"      • hgb_model.joblib (ensemble bundle)")

if __name__ == "__main__":
    main()
