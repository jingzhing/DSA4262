# Final_HGBmodel_wGHOST.py
#20% run python Final_HGBmodel_wGHOST/Final_HGBmodel_wGHOST.py train ^  --json datasets\dataset0.json.gz ^  --labels datasets\data.info.labelled ^  --out_model runs\hgb_ens5_tuned_ghost.joblib ^  --preset prauc ^  --ensemble_n 5 ^  --cv 5 ^  --tune
#100% run python Final_HGBmodel_wGHOST\Final_HGBmodel_wGHOST.py train ^  --json datasets\dataset0.json.gz ^  --labels datasets\data.info.labelled ^  --out_model Final_HGBmodel_wGHOST\model\model_tuned.joblib ^  --preset prauc ^  --ensemble_n 5 ^  --cv 5 ^  --thr_mode ghost ^  --train_full ^  --tune

import argparse, json, gzip, time
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score, classification_report, cohen_kappa_score,
    confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import make_scorer

DRACH = [
    "AAACA","AAACC","AAACT","AAGCA","AAGCC","AAGCT","ATACA","ATACC","ATACT",
    "GAACA","GAACC","GAACT","GAGCA","GAGCC","GAGCT","GTACA","GTACC","GTACT"
]

def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")

def _middle_5mer(seven):
    s = seven.upper()
    return s[1:6] if len(s) >= 6 else s[:5].ljust(5, "N")

def _one_hot_drach(m5):
    v = np.zeros(len(DRACH), dtype=np.float32)
    if m5 in DRACH:
        v[DRACH.index(m5)] = 1.0
    return v

def _aggregate_enhanced(reads):
    if not reads:
        return np.zeros(76, dtype=np.float32)
    X = np.asarray(reads, dtype=np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    q25, q50, q75 = np.quantile(X, [0.25,0.5,0.75], axis=0)
    iqr = q75 - q25
    rng = X.max(axis=0) - X.min(axis=0)
    cv = std / (mean + 1e-6)
    cov_count = np.array([len(X)], dtype=np.float32)
    ctx_mean = X[:,3:6].mean(axis=0) if X.shape[1] >= 6 else np.zeros(3, np.float32)
    return np.concatenate([mean, std, q25, q50, q75, iqr, rng, cv, cov_count, ctx_mean], axis=0).astype(np.float32)

def _stream_json(json_path):
    with _open(json_path) as f:
        pbar = tqdm(total=None, unit="lines", desc="read-json")
        for line in f:
            pbar.update(1)
            if not line.strip():
                continue
            obj = json.loads(line)
            t_id = next(iter(obj))
            pos_dict = obj[t_id]
            pos_key = next(iter(pos_dict))
            pos = int(pos_key)
            inner = pos_dict[str(pos)]
            seven = next(iter(inner))
            reads = inner[seven]
            yield t_id, pos, seven, reads
        pbar.close()

def parse_enhanced(json_path, restrict_keys=None):
    keys, base_list, motif_list, tids = [], [], [], []
    for t_id, pos, seven, reads in _stream_json(json_path):
        key = (t_id, pos)
        if restrict_keys and key not in restrict_keys:
            continue
        base_feats = _aggregate_enhanced(reads)
        motif = _one_hot_drach(_middle_5mer(seven))
        keys.append(key)
        tids.append(t_id)
        base_list.append(base_feats)
        motif_list.append(motif)
    if len(base_list) == 0:
        X_base = np.zeros((0, 76), dtype=np.float32)
        X_motif = np.zeros((0, 18), dtype=np.float32)
        tids = np.array([], dtype=object)
    else:
        X_base = np.stack(base_list).astype(np.float32)
        X_motif = np.stack(motif_list).astype(np.float32)
        tids = np.array(tids)
    return keys, X_base, X_motif, tids

def load_labels(path):
    df = pd.read_csv(path)
    need = {"gene_id","transcript_id","transcript_position","label"}
    if not need.issubset(df.columns):
        raise SystemExit(f"labels CSV must contain: {need}")
    df = df[["gene_id","transcript_id","transcript_position","label"]].copy()
    df["transcript_position"] = df["transcript_position"].astype(int)
    df["label"] = df["label"].astype(int)
    return df

def build_xy_enhanced(json_path, labels):
    print("[1/6] build-keys")
    keyset = {(t, int(p)) for t, p in zip(labels.transcript_id, labels.transcript_position)}
    print("[2/6] parse-features")
    keys, X_base, X_motif, tids = parse_enhanced(json_path, restrict_keys=keyset)
    if len(keys) == 0:
        raise SystemExit("no overlap between JSON sites and labels")
    print("[3/6] transcript-centering")
    uniq = np.unique(tids)
    t_means = {}
    for t in tqdm(uniq, desc="t-means"):
        m = (tids == t)
        t_means[t] = X_base[m].mean(axis=0)
    X_norm = np.vstack([X_base[i] - t_means[tids[i]] for i in range(len(tids))]).astype(np.float32)
    print("[4/6] concat")
    X = np.concatenate([X_base, X_norm, X_motif], axis=1).astype(np.float32)
    print("[5/6] labels-groups")
    lab_map = {(r.transcript_id, int(r.transcript_position)):(int(r.label), r.gene_id) for r in labels.itertuples(index=False)}
    y = np.array([lab_map[k][0] for k in keys], dtype=int)
    groups = np.array([lab_map[k][1] for k in keys])
    print("[6/6] done")
    return keys, X, y, groups

def roc_hmean(y_true, y_bin):
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    s = tpr + tnr
    return 0.0 if s == 0 else 2 * (tpr * tnr) / s

def ghost_threshold(y_true, proba, thr_grid):
    best_val = -1.0
    best_thr = 0.5
    best_metric = "kappa"
    for t in tqdm(thr_grid, desc="ghost-kappa"):
        k = cohen_kappa_score(y_true, (proba >= t).astype(int))
        v = k if np.isfinite(k) else -np.inf
        if v > best_val:
            best_val = v
            best_thr = float(t)
            best_metric = "kappa"
    if not np.isfinite(best_val) or best_val < -1e10:
        best_val = -1.0
        for t in tqdm(thr_grid, desc="ghost-hmean"):
            h = roc_hmean(y_true, (proba >= t).astype(int))
            if h > best_val:
                best_val = h
                best_thr = float(t)
                best_metric = "roc_hmean"
    return best_thr, best_metric

def f1_opt_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.nanargmax(f1[:-1])]) if len(t) else 0.5

def select_threshold(y_true, proba, mode, thr_grid=None, thr_value=None):
    if mode == "ghost":
        return ghost_threshold(y_true, proba, thr_grid)
    if mode == "f1":
        return f1_opt_threshold(y_true, proba), "f1"
    if mode == "custom":
        return float(thr_value), "custom"
    return ghost_threshold(y_true, proba, thr_grid)

def fit_one_calibrated_hgb(seed, X, y, w, h):
    base = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,
        random_state=seed,
        learning_rate=h["lr"],
        max_leaf_nodes=h["nodes"],
        min_samples_leaf=h["leaf"],
        max_iter=h["iters"],
        l2_regularization=h["l2"]
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=h["cv"])
    clf.fit(X, y, sample_weight=w)
    return clf

def proba_of(clfs, X):
    if isinstance(clfs, list):
        return np.mean([c.predict_proba(X)[:,1] for c in clfs], axis=0)
    return clfs.predict_proba(X)[:,1]

def predict_proba_bundle(bundle, X):
    if bundle.get("clfs") is not None:
        return np.mean([c.predict_proba(X)[:,1] for c in bundle["clfs"]], axis=0)
    return bundle["clf"].predict_proba(X)[:,1]

def tune_hgb(X_tr, y_tr, groups_tr, w_tr, cv_folds, random_state):
    print("[tune] RandomizedSearchCV")
    base = HistGradientBoostingClassifier(loss="log_loss", early_stopping=False, random_state=random_state)
    param_dist = {
        "learning_rate":      [0.03, 0.05, 0.06, 0.08, 0.1],
        "max_leaf_nodes":     [31, 63, 127, 255],
        "min_samples_leaf":   [5, 10, 20, 40],
        "max_iter":           [200, 300, 400],
        "l2_regularization":  [0.0, 1e-3, 1e-2]
    }
    pr_scorer = make_scorer(average_precision_score, needs_proba=True)
    sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=40,
        scoring=pr_scorer,
        cv=sgkf,
        n_jobs=-1,
        verbose=2,
        refit=True,
        random_state=random_state
    )
    rs.fit(X_tr, y_tr, groups=groups_tr, sample_weight=w_tr)
    best = rs.best_params_
    best["cv"] = cv_folds
    print(f"[tune] best PR-AUC={rs.best_score_:.4f} params={best}")
    return best

def train(args):
    print("=== TRAIN: enhanced features + calibrated HGB ensemble ===")
    t0 = time.time()
    labels = load_labels(args.labels)
    keys, X, y, groups = build_xy_enhanced(args.json, labels)
    if len(keys) == 0:
        raise SystemExit("no training sites parsed")
    if args.train_full:
        X_tr, y_tr, groups_tr = X, y, groups
        X_va, y_va = X[:0], y[:0]
        print(f"[split] train_full={len(y_tr)}")
    else:
        gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=args.random_state)
        (tr, va), = gss.split(X, y, groups=groups)
        X_tr, y_tr, groups_tr = X[tr], y[tr], groups[tr]
        X_va, y_va = X[va], y[va]
        print(f"[split] train={len(y_tr)} test={len(y_va)} pos_tr={int((y_tr==1).sum())} pos_te={int((y_va==1).sum())}")

    presets = {
        "prauc":   {"lr":0.06,"nodes":127,"leaf":10,"iters":300,"l2":1e-3,"pos_boost":1.5},
        "balanced":{"lr":0.06,"nodes":63, "leaf":10,"iters":200,"l2":1e-3,"pos_boost":2.0},
        "recall":  {"lr":0.05,"nodes":127,"leaf":5, "iters":300,"l2":0.0,"pos_boost":3.0}
    }
    if args.tune:
        w_tmp = compute_sample_weight(class_weight="balanced", y=y_tr)
        w_tmp[y_tr==1] *= presets[args.preset]["pos_boost"]
        tuned = tune_hgb(X_tr, y_tr, groups_tr, w_tmp, args.cv, args.random_state)
        h = {"lr":tuned["learning_rate"],"nodes":tuned["max_leaf_nodes"],"leaf":tuned["min_samples_leaf"],"iters":tuned["max_iter"],"l2":tuned["l2_regularization"],"cv":args.cv}
    else:
        p = presets[args.preset]
        h = {"lr":p["lr"],"nodes":p["nodes"],"leaf":p["leaf"],"iters":p["iters"],"l2":p["l2"],"cv":args.cv}

    print("[weights] balanced + pos_boost")
    w_tr = compute_sample_weight(class_weight="balanced", y=y_tr)
    pos_boost = presets[args.preset]["pos_boost"]
    w_tr[y_tr==1] *= pos_boost

    print(f"[fit] calibrated HGB x{args.ensemble_n}")
    clfs = []
    for i in range(args.ensemble_n):
        seed = args.random_state + i
        clfs.append(fit_one_calibrated_hgb(seed, X_tr, y_tr, w_tr, h))

    print("[proba] compute (train)")
    p_tr = proba_of(clfs if args.ensemble_n>1 else clfs[0], X_tr)

    thr_grid = np.arange(args.thr_start, args.thr_stop+1e-9, args.thr_step)
    print(f"[threshold] mode={args.thr_mode}")
    thr, metric = select_threshold(y_tr, p_tr, mode=args.thr_mode, thr_grid=thr_grid, thr_value=args.thr_value)

    bundle = {
        "type":"hgb_ghost_enhanced_ens" if args.ensemble_n>1 else "hgb_ghost_enhanced",
        "clfs": clfs if args.ensemble_n>1 else None,
        "clf":  None if args.ensemble_n>1 else clfs[0],
        "ghost_threshold":float(thr),
        "ghost_metric":metric,
        "preset":args.preset,
        "feature_dim":int(X.shape[1]),
        "params":{"learning_rate":h["lr"],"max_leaf_nodes":h["nodes"],"min_samples_leaf":h["leaf"],"max_iter":h["iters"],"l2_regularization":h["l2"],"pos_boost":pos_boost},
        "cv":args.cv,
        "random_state":args.random_state,
        "ensemble_n":int(args.ensemble_n),
        "thr_mode":args.thr_mode,
        "tuned":bool(args.tune)
    }

    if len(y_va):
        p_va = proba_of(clfs if args.ensemble_n>1 else clfs[0], X_va)
        y_va_pred = (p_va >= thr).astype(int)
        pr_va = average_precision_score(y_va, p_va)
        roc_va = roc_auc_score(y_va, p_va)
        print(f"[valid] PR-AUC={pr_va:.4f} ROC-AUC={roc_va:.4f}")
        print("Validation @threshold")
        print(classification_report(y_va, y_va_pred, digits=3))
    else:
        print("[valid] skipped (train_full)")

    pr_tr = average_precision_score(y_tr, p_tr)
    roc_tr = roc_auc_score(y_tr, p_tr)
    print(f"[train] PR-AUC={pr_tr:.4f} ROC-AUC={roc_tr:.4f}")
    print(f"[threshold] value={thr:.4f} via {metric}")

    joblib.dump(bundle, args.out_model)
    t1 = time.time()
    print(f"[save] {args.out_model}")
    print(f"[done] {t1-t0:.1f}s")

def predict(args):
    print("=== PREDICT: enhanced features ===")
    t0 = time.time()
    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or bundle.get("type") not in {"hgb_ghost_enhanced","hgb_ghost_enhanced_ens"}:
        raise SystemExit("model bundle type mismatch")
    print("[parse] features")
    keys, X_base, X_motif, tids = parse_enhanced(args.json, restrict_keys=None)
    if len(keys) == 0:
        pd.DataFrame(columns=["transcript_id","transcript_position","score"]).to_csv(args.output, index=False)
        print(f"[save] {args.output}")
        return
    print("[center] transcript")
    uniq = np.unique(tids)
    t_means = {}
    for t in tqdm(uniq, desc="t-means"):
        m = (tids == t)
        t_means[t] = X_base[m].mean(axis=0)
    X_norm = np.vstack([X_base[i] - t_means[tids[i]] for i in range(len(tids))]).astype(np.float32)
    X = np.concatenate([X_base, X_norm, X_motif], axis=1).astype(np.float32)
    if X.shape[1] != int(bundle.get("feature_dim", X.shape[1])):
        raise SystemExit(f"feature dim mismatch: expected {bundle.get('feature_dim')}, got {X.shape[1]}")
    print("[infer] probabilities")
    proba = predict_proba_bundle(bundle, X)
    df = pd.DataFrame({"transcript_id":[k[0] for k in keys],"transcript_position":[k[1] for k in keys],"score":proba})
    if args.emit_labels:
        thr = float(bundle.get("ghost_threshold", 0.5))
        df["label_ghost"] = (proba >= thr).astype(int)
    df.to_csv(args.output, index=False)
    t1 = time.time()
    print(f"[save] {args.output}")
    print(f"[done] {t1-t0:.1f}s")

def evaluate(args):
    print("=== EVAL: metrics and reports ===")
    t0 = time.time()
    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or bundle.get("type") not in {"hgb_ghost_enhanced","hgb_ghost_enhanced_ens"}:
        raise SystemExit("model bundle type mismatch")
    labels = load_labels(args.labels)
    keyset = {(t, int(p)) for t, p in zip(labels.transcript_id, labels.transcript_position)}
    print("[parse] features")
    keys, X_base, X_motif, tids = parse_enhanced(args.json, restrict_keys=keyset)
    if len(keys) == 0:
        raise SystemExit("no overlap between JSON sites and labels")
    print("[center] transcript")
    uniq = np.unique(tids)
    t_means = {}
    for t in tqdm(uniq, desc="t-means"):
        m = (tids == t)
        t_means[t] = X_base[m].mean(axis=0)
    X_norm = np.vstack([X_base[i] - t_means[tids[i]] for i in range(len(tids))]).astype(np.float32)
    X = np.concatenate([X_base, X_norm, X_motif], axis=1).astype(np.float32)
    if X.shape[1] != int(bundle.get("feature_dim", X.shape[1])):
        raise SystemExit(f"feature dim mismatch: expected {bundle.get('feature_dim')}, got {X.shape[1]}")
    lab_map = {(r.transcript_id, int(r.transcript_position)):(int(r.label), r.gene_id) for r in labels.itertuples(index=False)}
    y = np.array([lab_map[k][0] for k in keys], dtype=int)
    print("[infer] probabilities")
    proba = predict_proba_bundle(bundle, X)
    thr = float(args.thr_override) if args.thr_override is not None else float(bundle.get("ghost_threshold", 0.5))
    y_hat = (proba >= thr).astype(int)
    pr_auc = float(average_precision_score(y, proba))
    roc_auc = float(roc_auc_score(y, proba))
    kappa = float(cohen_kappa_score(y, y_hat))
    rhm = float(roc_hmean(y, y_hat))
    prec = float(precision_score(y, y_hat, zero_division=0))
    rec = float(recall_score(y, y_hat, zero_division=0))
    f1 = float(f1_score(y, y_hat, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    print(f"[metrics] PR-AUC={pr_auc:.4f} ROC-AUC={roc_auc:.4f} Thr={thr:.3f} κ={kappa:.4f} Hmean={rhm:.4f}")
    print("Report @threshold")
    print(classification_report(y, y_hat, digits=3))
    if args.preds_out:
        dfp = pd.DataFrame({"transcript_id":[k[0] for k in keys],"transcript_position":[k[1] for k in keys],"true_label":y,"score":proba,"pred_label":y_hat})
        dfp.to_csv(args.preds_out, index=False)
        print(f"[save] {args.preds_out}")
    if args.curves_out_prefix:
        p, r, th = precision_recall_curve(y, proba)
        fpr, tpr, roc_th = roc_curve(y, proba)
        pd.DataFrame({"precision":p,"recall":r,"threshold":[np.nan]+list(th)}).to_csv(args.curves_out_prefix+"_pr.csv", index=False)
        pd.DataFrame({"fpr":fpr,"tpr":tpr,"threshold":roc_th}).to_csv(args.curves_out_prefix+"_roc.csv", index=False)
        print(f"[save] {args.curves_out_prefix}_pr.csv")
        print(f"[save] {args.curves_out_prefix}_roc.csv")
    if args.report_out:
        rep = {"pr_auc":pr_auc,"roc_auc":roc_auc,"threshold":thr,"kappa":kappa,"roc_hmean":rhm,"precision":prec,"recall":rec,"f1":f1,"confusion":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},"preset":bundle.get("preset"),"ensemble_n":bundle.get("ensemble_n",1),"thr_mode":bundle.get("thr_mode")}
        pd.Series(rep, dtype=object).to_json(args.report_out, orient="index")
        print(f"[save] {args.report_out}")
    t1 = time.time()
    print(f"[done] {t1-t0:.1f}s")

def main():
    ap = argparse.ArgumentParser(description="Enhanced features + Calibrated HGB + Ensemble + Thresholding")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--json", required=True)
    tr.add_argument("--labels", required=True)
    tr.add_argument("--out_model", default="hgb_ghost_enhanced.joblib")
    tr.add_argument("--preset", choices=["prauc","balanced","recall"], default="prauc")
    tr.add_argument("--cv", type=int, default=5)
    tr.add_argument("--random_state", type=int, default=42)
    tr.add_argument("--train_full", action="store_true")
    tr.add_argument("--ensemble_n", type=int, default=5)
    tr.add_argument("--thr_mode", choices=["ghost","f1","custom"], default="ghost")
    tr.add_argument("--thr_value", type=float, default=None)
    tr.add_argument("--thr_start", type=float, default=0.05)
    tr.add_argument("--thr_stop", type=float, default=0.95)
    tr.add_argument("--thr_step", type=float, default=0.05)
    tr.add_argument("--tune", action="store_true")
    tr.set_defaults(func=train)

    pr = sub.add_parser("predict")
    pr.add_argument("--json", required=True)
    pr.add_argument("--model", default="/opt/model/model.joblib")
    pr.add_argument("--output", default="predictions.csv")
    pr.add_argument("--emit_labels", action="store_true")
    pr.set_defaults(func=predict)

    ev = sub.add_parser("eval")
    ev.add_argument("--json", required=True)
    ev.add_argument("--labels", required=True)
    ev.add_argument("--model", default="/opt/model/model.joblib")
    ev.add_argument("--thr_override", type=float, default=None)
    ev.add_argument("--preds_out", default=None)
    ev.add_argument("--curves_out_prefix", default=None)
    ev.add_argument("--report_out", default=None)
    ev.set_defaults(func=evaluate)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    import sys
    main()