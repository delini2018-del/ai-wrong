"""Resumable training (lets long runs be split into chunks). Usage:
   python train_stage.py NAME "512,256" TOTAL_EPOCHS CHUNK [track]"""
import json, os, pickle, sys, time
import numpy as np
from sklearn.neural_network import MLPClassifier
from confidence_lab import load_fashion, summarize

name, hidden, total, chunk = sys.argv[1], tuple(int(h) for h in sys.argv[2].split(",")), int(sys.argv[3]), int(sys.argv[4])
track = len(sys.argv) > 5
Xtr, ytr, Xva, yva, Xte, yte = load_fashion(seed=0)
os.makedirs("results/models", exist_ok=True)
pk, hk = f"results/models/{name}.pkl", f"results/models/{name}_hist.json"
if os.path.exists(pk):
    m, hist = pickle.load(open(pk, "rb")), json.load(open(hk))
else:
    m = MLPClassifier(hidden_layer_sizes=hidden, alpha=0.0, batch_size=128,
                      learning_rate_init=1e-3, random_state=0)
    hist = []
done = len(hist) if track else getattr(m, "_epochs_done", 0)
t = time.time()
for ep in range(done + 1, min(done + chunk, total) + 1):
    m.partial_fit(Xtr, ytr, classes=np.arange(10))
    m._epochs_done = ep
    if track:
        s = summarize(m.predict_proba(Xte), yte)
        s.update(epoch=ep, train_accuracy=float((m.predict(Xtr) == ytr).mean()))
        hist.append(s)
        print(f"{name} ep {ep:2d} acc {s['accuracy']:.4f} train {s['train_accuracy']:.4f} "
              f"conf {s['mean_conf']:.4f} ece {s['ece']:.4f}", flush=True)
pickle.dump(m, open(pk, "wb")); json.dump(hist, open(hk, "w"))
print(name, "epochs done:", m._epochs_done, f"({time.time()-t:.0f}s)")
