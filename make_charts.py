"""Plot every chart in the article from results/results.json (+ saved probabilities)."""
import json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom
from sklearn.datasets import load_digits
from confidence_lab import load_fashion, CLASSES, temper

R = json.load(open("results/results.json"))
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, RED, GREY = "#2563eb", "#ea580c", "#dc2626", "#6b7280"
SRC = "Source: author's experiment (Fashion-MNIST, MLP 512-256). Code: see repo."

def pct(p): return ">99.9%" if p > .999 else f"{100*p:.1f}%"

def foot(fig): fig.text(0.01, -0.02, SRC, fontsize=8.5, color=GREY)

# 1 drift over training
H = R["E2_training"]; ep = [h["epoch"] for h in H]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ep, [100*h["mean_conf"] for h in H], color=ORANGE, lw=2.5, label="How sure it says it is (avg confidence)")
ax.plot(ep, [100*h["accuracy"] for h in H], color=BLUE, lw=2.5, label="How often it is actually right (test accuracy)")
ax.fill_between(ep, [100*h["accuracy"] for h in H], [100*h["mean_conf"] for h in H], color=ORANGE, alpha=.12)
ax.set(xlabel="Training epoch", ylabel="%", ylim=(82, 98), title="Training longer made the model sure, not right")
ax.annotate(f"gap at epoch 40: {100*(H[-1]['mean_conf']-H[-1]['accuracy']):.1f} pts", (40, 91.8), ha="right", color=ORANGE)
ax.legend(frameon=False, loc="lower right"); foot(fig); fig.savefig("charts/01_confidence_drift.png"); plt.close()

# 2 confidence buckets
B = R["E1_clean"]["buckets"]
labels = ["<60%", "60–80%", "80–90%", "90–99%", "99–99.9%", ">99.9%"]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, [100*b["accuracy"] for b in B], color=BLUE)
for b, r in zip(bars, B):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.2, f"{100*r['accuracy']:.0f}%\n({100*r['share']:.0f}% of cases)", ha="center", fontsize=10)
ax.set(ylim=(0, 115), xlabel="Model's stated confidence", ylabel="Actual accuracy (%)", title="Higher confidence really does mean fewer mistakes")
foot(fig); fig.savefig("charts/02_confidence_buckets.png"); plt.close()

# 3 reliability before/after temperature
def rel(ax, rows, title, ece):
    mids = [r[0] for r in rows if r[2] is not None]; acc = [r[2] for r in rows if r[2] is not None]
    cf = [r[3] for r in rows if r[2] is not None]
    ax.bar(mids, acc, width=.09, color=BLUE, label="Actual accuracy")
    ax.bar(mids, np.array(cf)-np.array(acc), bottom=acc, width=.09, color=RED, alpha=.35, label="Overconfidence gap")
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect honesty")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", title=f"{title}\nECE = {ece:.3f}")
t = R["E6_temperature"]
fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
rel(axs[0], R["E1_clean"]["reliability"], "Raw model", t["before"]["ece"])
rel(axs[1], t["reliability_after"], f"After temperature scaling (T = {t['T']:.1f})", t["after"]["ece"])
axs[0].set_ylabel("Accuracy"); axs[0].legend(frameon=False, loc="upper left", fontsize=10)
foot(fig); fig.savefig("charts/03_reliability_temperature.png"); plt.close()

# 4 selective accuracy
S = R["E1_clean"]["selective"]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot([100*c for c, _ in S], [100*a for _, a in S], color=BLUE, lw=2.5, marker="o", ms=4)
for c, a in S:
    if round(c, 2) in (0.5, 0.8, 1.0): ax.annotate(f"{100*a:.1f}%", (100*c, 100*a), textcoords="offset points", xytext=(0, 10), ha="center")
ax.set(xlabel="Share of questions the model is allowed to answer (most confident first, %)", ylabel="Accuracy on answered (%)",
       ylim=(86, 101), title="Let it skip its least confident cases and accuracy jumps")
foot(fig); fig.savefig("charts/04_selective_accuracy.png"); plt.close()

# 5 rotation shift
rot = [s for s in R["E4_shift"] if s["kind"] == "rotation"]
fig, ax = plt.subplots(figsize=(9, 5))
lv = [s["level"] for s in rot]
ax.plot(lv, [100*s["mean_conf"] for s in rot], color=ORANGE, lw=2.5, marker="o", label="Avg confidence")
ax.plot(lv, [100*s["accuracy"] for s in rot], color=BLUE, lw=2.5, marker="o", label="Accuracy")
ax.plot(lv, [100*s["mean_conf_temp"] for s in rot], color=ORANGE, lw=1.5, ls="--", marker="o", ms=3, label="Avg confidence after temperature scaling")
ax.set(xlabel="Image rotated by (degrees)", ylabel="%", ylim=(0, 100), xticks=lv, title="Turn the photos sideways: accuracy collapses, confidence doesn't")
ax.legend(frameon=False, loc="center left"); foot(fig); fig.savefig("charts/05_rotation_shift.png"); plt.close()

# 6 confidence histograms: clothes vs digits vs noise
P = np.load("results/test_probs.npy").max(1)
Dg = np.load("results/handwritten_digits_probs.npy").max(1)
Nz = np.load("results/random_noise_probs.npy").max(1)
fig, ax = plt.subplots(figsize=(9, 5)); bins = np.linspace(0.1, 1, 19)
for x, lab, col in [(P, "Clothing (what it was trained on)", BLUE), (Dg, "Handwritten digits (never seen)", ORANGE), (Nz, "Pure random noise", RED)]:
    ax.hist(x, bins=bins, weights=np.ones(len(x))*100/len(x), histtype="step", lw=2.5, color=col, label=lab)
ax.set(xlabel="Model's confidence in its top answer", ylabel="% of inputs", title="It is just as sure about things it has never seen")
ax.legend(frameon=False, loc="upper left"); foot(fig); fig.savefig("charts/06_foreign_inputs.png"); plt.close()

# 7 gallery: digits and rotated clothes with predictions
main = pickle.load(open("results/models/main.pkl", "rb"))
d = load_digits(); digs = np.clip(np.stack([zoom(im, 28/8, order=1) for im in d.images])/16, 0, 1)
pd_ = np.load("results/handwritten_digits_probs.npy")
_, _, _, _, Xte, yte = load_fashion(seed=0)
picks = [i for i in np.argsort(-pd_.max(1))][::60][:5]
rotimg = rotate(Xte[:400].reshape(-1, 28, 28), 90, axes=(1, 2), reshape=False, order=1)
pr = main.predict_proba(rotimg.reshape(-1, 784).astype(np.float32))
wrongr = [i for i in np.argsort(-pr.max(1)) if pr[i].argmax() != yte[i]][:5]
fig, axs = plt.subplots(2, 5, figsize=(11, 5.4))
for a, i in zip(axs[0], picks):
    a.imshow(digs[i], cmap="gray_r"); a.axis("off")
    a.set_title(f"digit {d.target[i]} → \"{CLASSES[pd_[i].argmax()]}\"\n{pct(pd_[i].max())} sure", fontsize=10)
for a, i in zip(axs[1], wrongr):
    a.imshow(rotimg[i], cmap="gray_r"); a.axis("off")
    a.set_title(f"{CLASSES[yte[i]]} (90°) → \"{CLASSES[pr[i].argmax()]}\"\n{pct(pr[i].max())} sure", fontsize=10)
fig.suptitle("Real mistakes from the experiment, with the model's own confidence", fontsize=13)
foot(fig); fig.savefig("charts/07_confident_mistakes.png"); plt.close()

# 8 size sweep
Z = R["E3_size"]
fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(Z))
ax.bar(x-.2, [100*z["accuracy"] for z in Z], .4, color=BLUE, label="Accuracy (%)")
ax.bar(x+.2, [100*z["mean_conf"] for z in Z], .4, color=ORANGE, label="Avg confidence (%)")
for i, z in enumerate(Z): ax.text(i, 99, f"ECE\n{z['ece']:.3f}", ha="center", fontsize=9.5, color=RED)
ax.set(xticks=x, ylim=(80, 103), ylabel="%", title="Bigger model: a little more right, a lot more sure")
ax.set_xticklabels([f"{z['model']}\n{z['params']/1e3:.0f}K params" for z in Z], fontsize=9)
ax.legend(frameon=False, loc="lower right"); foot(fig); fig.savefig("charts/08_model_size.png"); plt.close()
print("charts written")
