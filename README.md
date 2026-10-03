# Does AI Know When It's Wrong?

Code for the Medium article *"AI Doesn't Know When It's Wrong. Or Does It? I Tested Whether Confidence Predicts Correctness."*

Small neural networks are trained from scratch on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (CPU only, seeded) to test whether a model's confidence predicts its mistakes.

| Experiment | Question | Headline result |
|---|---|---|
| E1 | Does confidence rank mistakes? | AUROC 0.878; answering the most confident 50% gives 99.2% accuracy |
| E2 | Does longer training inflate confidence? | ECE 0.010 (epoch 2) → 0.075 (epoch 40) while accuracy rose 85.4% → 88.0% |
| E3 | Do bigger models inflate confidence? | ECE 0.016 (MLP 16) → 0.077 (MLP 1024-512) |
| E4 | What happens under distribution shift? | Rotated 90°: 3.5% accuracy at 94.8% avg confidence |
| E5 | Inputs that aren't clothing? | Median confidence on handwritten digits: 99.3% |
| E6 | Does temperature scaling fix it? | ECE 0.075 → 0.019 on clean data; little help under shift |

## Run it
```bash
pip install -r requirements.txt
bash run_all.sh          # downloads data, trains 5 models, analyzes, plots
```
Training is resumable: `python train_stage.py NAME HIDDEN TOTAL_EPOCHS CHUNK [track]` continues from its checkpoint, so long runs can be split into chunks.

## Files
- `confidence_lab.py` – data loading, ECE, AUROC, selective accuracy, temperature scaling
- `train_stage.py` – resumable MLP training (scikit-learn), optional per-epoch tracking
- `analyze.py` – experiments E1–E6, writes `results/results.json`
- `make_charts.py` – every chart in the article (`charts/`)
- `results/results.json` – the exact numbers reported in the article

## Notes and limits
Models are small MLPs on 28×28 images, trained without regularization (`alpha=0`). Results illustrate calibration behaviour; they are not claims about any specific large language model. Temperature is fitted on a 5,000-image validation split only, never on test data.

## Credits
Fashion-MNIST © Zalando SE, MIT License. Handwritten digits from scikit-learn's `load_digits`. Methods follow Guo et al. (2017) and Hendrycks & Gimpel (2017).

License: MIT
