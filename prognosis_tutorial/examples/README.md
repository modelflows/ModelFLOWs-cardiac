# Examples

This folder contains example configuration files to reproduce the main steps of the
prognosis tutorial.

## Contents

| File | Description |
|------|-------------|
| `param_train_regression.py` | `param` dictionary configured for regression mode (prognosis) |
| `param_train_classification.py` | `param` dictionary configured for classification mode |
| `param_test.py` | `param` dictionary for the test/evaluation script |

> Copy the relevant file into `Prognosis_scripts/` and rename it to match the target
> script, then adjust the dataset and output paths before running.

## Regression vs. Classification Mode

The only change needed to switch between modes is the `gt_outcome_directory` key:

```python
# Regression (prognosis)
param['gt_outcome_directory'] = '/path/to/ecos_outcome_gt'

# Classification (no scalar outcome needed)
param['gt_outcome_directory'] = None
```

All other parameters remain the same. See [TUTORIAL.md](../TUTORIAL.md) for a full
description of each parameter.

## Typical Usage

```bash
# Pre-train the MAE
python Prognosis_scripts/train_mae__demo-modelflowsapp.py

# Fine-tune for prognosis (set param['finetune'] first)
python Prognosis_scripts/train_mae__demo-modelflowsapp.py

# Evaluate on the test split
python Prognosis_scripts/test_mae__demo-modelflowsapp.py
```
