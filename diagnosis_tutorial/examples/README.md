# Examples

This folder contains example configuration files and helper scripts to reproduce the
main steps of the tutorial.

## Contents

| File | Description |
|------|-------------|
| `normalization_orig.py` | Example normalisation script configured for the echocardiography dataset |
| `train_vit_config.py` | Example parameter dictionary for ViT training |
| `test_vit_config.py` | Example parameter dictionary for ViT testing |

> Copy the relevant file to the corresponding script folder and adjust the dataset
> paths before running.

## Typical Usage


```bash
# 1. Compute training set statistics
python examples/normalization_orig.py

# 2. Train the ViT classifier
python Diagnosis_scripts/train_vit_modelflows-app.py

# 3. Evaluate on the test split
python Diagnosis_scripts/test_vit_modelflows-app.py
```

Refer to [TUTORIAL.md](../TUTORIAL.md) for a full description of each step.
