#!/usr/bin/env bash
# Reproduces every number and chart in the article. ~30-60 min on a laptop CPU.
set -e
bash download_data.sh
python train_stage.py main  512,256  40 40 track   # tracked every epoch (E2)
python train_stage.py w16   16       40 40
python train_stage.py w64   64       40 40
python train_stage.py w256  256      40 40
python train_stage.py w1024 1024,512 40 40
python analyze.py
python make_charts.py
