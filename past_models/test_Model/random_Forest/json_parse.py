# json_parse.py
import json, gzip
import numpy as np

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

def _aggregate(reads) -> np.ndarray:
    if not reads:
        return np.zeros(49, dtype=np.float32)
    X = np.asarray(reads, dtype=np.float32)  # (N, 9)
    q25, q50, q75 = np.quantile(X, [0.25, 0.5, 0.75], axis=0)
    return np.concatenate([
        X.mean(0), X.std(0), q25, q50, q75,
        np.array([len(X)], np.float32),
        X[:, 3:6].mean(0)
    ]).astype(np.float32)

def _stream_json(json_path):
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

def parse(json_path: str, restrict_keys=None):
    """Main entry point."""
    keys, feats = [], []
    for t_id, pos, seven, reads in _stream_json(json_path):
        key = (t_id, pos)
        if restrict_keys and key not in restrict_keys:
            continue
        m5 = _middle_5mer(seven)
        site = _aggregate(reads)            # 49
        onehot = _one_hot_drach(m5)         # 18
        keys.append(key)
        feats.append(np.concatenate([site, onehot], 0))  # 67
    X = np.vstack(feats).astype(np.float32) if feats else np.zeros((0, 67), np.float32)
    return keys, X
