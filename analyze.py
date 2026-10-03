"""Evaluate trained checkpoints. Run after train_stage.py (see README).
  E1 confidence vs correctness (clean)   E2 drift over training (from history)
  E3 model size                          E4 distribution shift
  E5 foreign inputs (digits, noise)      E6 temperature scaling"""
import json, pickle
import numpy as np
from scipy.ndimage import rotate, zoom
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from confidence_lab import (load_fashion, summarize, reliability, selective_curve,
                            fit_temperature, temper, auroc)

Xtr, ytr, Xva, yva, Xte, yte = load_fashion(seed=0)
load = lambda n: pickle.load(open(f"results/models/{n}.pkl", "rb"))
main = load("main")
R = {"E2_training": json.load(open("results/models/main_hist.json"))}

# E1
P = main.predict_proba(Xte); conf, correct = P.max(1), P.argmax(1) == yte
buckets = []
for lo, hi in [(0, .6), (.6, .8), (.8, .9), (.9, .99), (.99, .999), (.999, 1.01)]:
    m = (conf >= lo) & (conf < hi)
    buckets.append(dict(lo=lo, hi=min(hi, 1), n=int(m.sum()), share=float(m.mean()),
                        accuracy=float(correct[m].mean()) if m.any() else None))
wrong = ~correct
R["E1_clean"] = dict(summary=summarize(P, yte), buckets=buckets,
    selective=selective_curve(conf, correct), reliability=reliability(conf, correct),
    n_wrong=int(wrong.sum()), wrong_over_99=int((wrong & (conf > .99)).sum()),
    wrong_over_999=int((wrong & (conf > .999)).sum()))
np.save("results/test_probs.npy", P)

# E6 temperature, fit on validation only
T = fit_temperature(main.predict_proba(Xva), yva); PT = temper(P, T)
R["E6_temperature"] = dict(T=T, before=summarize(P, yte), after=summarize(PT, yte),
                           reliability_after=reliability(PT.max(1), PT.argmax(1) == yte))

# E4 shift
rng = np.random.default_rng(0); imgs = Xte.reshape(-1, 28, 28); shift = []
for kind, levels in [("noise", [0, .1, .2, .3, .4, .6]), ("rotation", [0, 15, 30, 45, 60, 90])]:
    for lv in levels:
        if kind == "noise": X = np.clip(imgs + rng.normal(0, lv, imgs.shape), 0, 1)
        else: X = rotate(imgs, lv, axes=(1, 2), reshape=False, order=1) if lv else imgs
        Ps = main.predict_proba(X.reshape(-1, 784).astype(np.float32))
        s, st = summarize(Ps, yte), summarize(temper(Ps, T), yte)
        shift.append(dict(kind=kind, level=lv, **s, ece_temp=st["ece"], mean_conf_temp=st["mean_conf"]))
R["E4_shift"] = shift

# E5 foreign inputs
d = load_digits()
digits = np.clip(np.stack([zoom(im, 28 / 8, order=1) for im in d.images]) / 16.0, 0, 1)
digits = digits.reshape(-1, 784).astype(np.float32)
noise = rng.random((2000, 784)).astype(np.float32)
ood = {}
for name, X in [("handwritten_digits", digits), ("random_noise", noise)]:
    Po = main.predict_proba(X); c, cT = Po.max(1), temper(Po, T).max(1)
    lab = np.bincount(Po.argmax(1), minlength=10)
    ood[name] = dict(n=len(X), mean_conf=float(c.mean()), median_conf=float(np.median(c)),
        frac_over_90=float((c > .9).mean()), frac_over_99=float((c > .99).mean()),
        mean_conf_temp=float(cT.mean()), frac_over_90_temp=float((cT > .9).mean()),
        top_label=int(lab.argmax()), top_label_share=float(lab.max() / len(X)))
    np.save(f"results/{name}_probs.npy", Po)
cd = main.predict_proba(digits).max(1)
ood["auroc_clothes_vs_digits"] = auroc(np.r_[conf, cd], np.r_[np.ones(len(conf)), np.zeros(len(cd))])
ood["clean_frac_over_90"] = float((conf > .9).mean())
R["E5_ood"] = ood

# E3 size
lr = LogisticRegression(max_iter=200).fit(Xtr, ytr)
sizes = [dict(model="Logistic regression", params=7850, **summarize(lr.predict_proba(Xte), yte))]
for n, label in [("w16", "MLP 16"), ("w64", "MLP 64"), ("w256", "MLP 256"),
                 ("main", "MLP 512-256"), ("w1024", "MLP 1024-512")]:
    m = load(n); p = int(sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_))
    sizes.append(dict(model=label, params=p, **summarize(m.predict_proba(Xte), yte)))
R["E3_size"] = sizes
json.dump(R, open("results/results.json", "w"), indent=2)
print(json.dumps({k: v for k, v in R.items() if k != "E2_training"}, indent=1)[:6000])
