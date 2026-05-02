# Experiments

This directory is left clean for local deterministic experiment runs from `run_experiments.py`.

RapidFire AI multi-config experiments should be run with:

```bash
python3 rapidfire_experiments.py --input data/golden_qa.json --experiments-path rapidfire_experiments
```

The RapidFire artifacts, including its own `rapidfire.log` and metric files, are written under `../rapidfire_experiments/` by default.
