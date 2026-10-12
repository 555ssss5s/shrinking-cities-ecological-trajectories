#!/usr/bin/env bash
set -euo pipefail
PANEL=${1:-'data/rseimean_删除指定城市(1).csv'}
COORDS=${2:-'data/收缩城市坐标数据.xlsx'}
stcp --panel "$PANEL" --coords "$COORDS" \
  --distance euclidean --k 5 \
  --chains 4 --n-iter 6000 --burn 2000 --thin 4 \
  --seed 20260610 --cp-first 2004 --cp-last 2017 \
  --out outputs/joint_4chains
