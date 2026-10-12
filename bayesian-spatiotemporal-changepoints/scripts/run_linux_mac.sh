#!/usr/bin/env bash
set -euo pipefail
PANEL=${1:-''}
COORDS=${2:-''}
stcp --panel "$PANEL" --coords "$COORDS" \
  --distance euclidean --k 5 \
  --chains 4 --n-iter 6000 --burn 2000 --thin 4 \
  --seed 20260610 --cp-first 2004 --cp-last 2017 \
  --out outputs/joint_4chains
