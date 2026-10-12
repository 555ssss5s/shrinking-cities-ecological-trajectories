# Bayesian spatiotemporal changepoint model


## Install and test

Python 3.10 or newer is required.

```bash
python -m venv .venv
python -m pip install -e ".[test]"
python -m pytest -q
```

## Run with local data

Prepare local input files outside version control, then run:

```bash
stcp --panel "data/panel.csv" --coords "data/coordinates.xlsx" \
  --distance euclidean --k 5 \
  --chains 4 --n-iter 6000 --burn 2000 --thin 4 \
  --seed 20260610 --cp-first 2004 --cp-last 2017 \
  --out outputs/joint_4chains
```

The panel must contain either long-format `city`, `year`, and `rsei` fields or
wide-format annual columns. Coordinates must contain `city` plus projected
`x`/`y` fields for Euclidean distance, or `lon`/`lat` fields when using
`--distance haversine`.


