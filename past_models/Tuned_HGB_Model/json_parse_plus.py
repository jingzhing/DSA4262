import json, gzip
import numpy as np

# DRACH-like motifs from your original code
DRACH = [
    "AAACA","AAACC","AAACT","AAGCA","AAGCC","AAGCT","ATACA","ATACC","ATACT",
    "GAACA","GAACC","GAACT","GAGCA","GAGCC","GAGCT","GTACA","GTACC","GTACT"
]

def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")

def _middle_5mer(seven: str) -> str:
    s = seven.upper()
    return s[1:6] if len(s) >= 6 else s[:5].ljust(5, "N")

def _one_hot_drach(m5: str) -> np.ndarray:
    v = np.zeros(len(DRACH), dtype=np.float32)
    if m5 in DRACH:
        v[DRACH.index(m5)] = 1.0
    return v

def _aggregate_enhanced(reads) -> np.ndarray:
    """
    Turn variable-length per-read signals (N_reads x 9 feats)
    into a fixed vector with richer stats:
    - mean, std, q25,q50,q75, IQR, range, coeff var (9 each = 72)
    - coverage (#reads) (1)
    - local context mean of cols 3:6 (3)
    Total = 76 dims.
    """
    if not reads:
        return np.zeros(76, dtype=np.float32)

    X = np.asarray(reads, dtype=np.float32)  # (N_reads, 9)

    mean = X.mean(axis=0)                           # (9,)
    std  = X.std(axis=0)                            # (9,)
    q25, q50, q75 = np.quantile(X, [0.25,0.5,0.75], axis=0)  # each (9,)

    iqr = q75 - q25                                 # (9,)
    rng = X.max(axis=0) - X.min(axis=0)             # (9,)
    cv  = std / (mean + 1e-6)                       # (9,) coeff of variation

    cov_count = np.array([len(X)], dtype=np.float32)      # (1,)
    ctx_mean  = X[:,3:6].mean(axis=0) if X.shape[1] >= 6 else np.zeros(3, np.float32)  # (3,)

    feats = np.concatenate([
        mean, std, q25, q50, q75,
        iqr, rng, cv,
        cov_count,
        ctx_mean
    ], axis=0).astype(np.float32)  # (76,)

    return feats

def _stream_json(json_path):
    """
    Iterator over the JSON-lines style file.
    Yields: transcript_id, pos, heptamer, reads(list-of-9feat)
    """
    with _open(json_path) as f:
        for line in f:
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

def parse_enhanced(json_path: str, restrict_keys=None):
    """
    Returns:
      keys: list[(transcript_id, position)]
      X_base: np.ndarray (N, 76)  # rich site-level stats
      X_motif: np.ndarray (N, 18) # one-hot motif
    We do NOT yet add transcript-normalization. That happens in training.
    """
    keys = []
    base_list = []
    motif_list = []

    for t_id, pos, seven, reads in _stream_json(json_path):
        key = (t_id, pos)
        if restrict_keys and key not in restrict_keys:
            continue

        base_feats = _aggregate_enhanced(reads)         # (76,)
        m5 = _middle_5mer(seven)
        motif_oh = _one_hot_drach(m5)                   # (18,)

        keys.append(key)
        base_list.append(base_feats)
        motif_list.append(motif_oh)

    if len(base_list) == 0:
        X_base = np.zeros((0, 76), dtype=np.float32)
        X_motif = np.zeros((0, 18), dtype=np.float32)
    else:
        X_base = np.stack(base_list).astype(np.float32)   # (N,76)
        X_motif = np.stack(motif_list).astype(np.float32) # (N,18)

    return keys, X_base, X_motif