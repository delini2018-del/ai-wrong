"""Shared helpers: data loading, confidence metrics, temperature scaling."""
import gzip
import numpy as np

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def _read(path, offset):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=offset)


def load_fashion(data_dir="data", seed=0, n_val=5000):
    Xtr = _read(f"{data_dir}/train-images-idx3-ubyte.gz", 16).reshape(-1, 784)
    ytr = _read(f"{data_dir}/train-labels-idx1-ubyte.gz", 8)
    Xte = _read(f"{data_dir}/t10k-images-idx3-ubyte.gz", 16).reshape(-1, 784)
    yte = _read(f"{data_dir}/t10k-labels-idx1-ubyte.gz", 8)
    Xtr, Xte = Xtr.astype(np.float32) / 255.0, Xte.astype(np.float32) / 255.0
    perm = np.random.default_rng(seed).permutation(len(Xtr))
    va, tr = perm[:n_val], perm[n_val:]
    return Xtr[tr], ytr[tr], Xtr[va], ytr[va], Xte, yte


def ece(conf, correct, n_bins=15):
    """Expected Calibration Error: average gap between confidence and accuracy."""
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def reliability(conf, correct, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append((float((lo + hi) / 2), int(m.sum()),
                     float(correct[m].mean()) if m.any() else None,
                     float(conf[m].mean()) if m.any() else None))
    return rows


def auroc(score, positive):
    """Rank-based AUROC: P(score of a positive > score of a negative), ties = 0.5."""
    score, positive = np.asarray(score), np.asarray(positive, bool)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):                       # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def selective_curve(conf, correct, coverages=np.linspace(0.1, 1.0, 19)):
    """Answer only the most confident fraction of inputs; report accuracy there."""
    order = np.argsort(-conf)
    out = []
    for c in coverages:
        k = max(1, int(round(c * len(conf))))
        out.append((float(c), float(correct[order[:k]].mean())))
    return out


def summarize(proba, y):
    conf, pred = proba.max(1), proba.argmax(1)
    correct = (pred == y)
    nll = float(-np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1)).mean())
    return dict(accuracy=float(correct.mean()), mean_conf=float(conf.mean()),
                gap=float(conf.mean() - correct.mean()), ece=ece(conf, correct),
                nll=nll, auroc=auroc(conf, correct),
                conf_when_right=float(conf[correct].mean()),
                conf_when_wrong=float(conf[~correct].mean()) if (~correct).any() else None)


def temper(proba, T):
    z = np.log(np.clip(proba, 1e-12, 1)) / T     # log-probs are logits up to a constant
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(proba_val, y_val, grid=np.linspace(0.5, 5.0, 91)):
    """Pick the single temperature that minimises validation NLL (Guo et al., 2017)."""
    best = min(grid, key=lambda T: -np.log(np.clip(
        temper(proba_val, T)[np.arange(len(y_val)), y_val], 1e-12, 1)).mean())
    return float(best)
