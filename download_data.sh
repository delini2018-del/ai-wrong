#!/usr/bin/env bash
# Fashion-MNIST (Zalando Research, MIT License)
mkdir -p data && cd data
B=https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -sL -o "$f" "$B/$f"; done
