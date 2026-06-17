# AI-Based Cardiac Prognosis from Echocardiography — Documentation

This documentation accompanies the full step-by-step tutorial for automatic cardiac
prognosis from echocardiography image sequences using a Masked Autoencoder (MAE).

## Contents

- [TUTORIAL.md](../TUTORIAL.md) — Main tutorial: data preparation, modal decomposition
  reference, MAE pre-training, fine-tuning for prognosis, and evaluation.

## Modules

### Main scripts (`Prognosis_scripts/`)

| Module | Description |
|--------|-------------|
| `train_mae__demo-modelflowsapp.py` | MAE pre-training and supervised fine-tuning (PyTorch) |
| `test_mae__demo-modelflowsapp.py` | Model evaluation — regression and classification modes |
| `model_mae_image_loss.py` | MAE encoder–decoder with joint reconstruction + downstream loss |
| `engine_two_branch.py` | Training (`train_one_epoch`) and evaluation (`evaluate`) loops |

### Data utilities (`Prognosis_scripts/util/`)

| Module | Description |
|--------|-------------|
| `datasets.py` | Dataset builder supporting classification and regression modes; `timm`-based transforms |
| `results_functions.py` | Confusion matrix plotting and classification report export |
| `misc.py` | Distributed training helpers, model save/load, gradient scaler |
| `lr_decay.py` | Layer-wise learning rate decay (ELECTRA/BEiT style) |
| `lr_sched.py` | Cosine learning rate scheduler with warm-up |
| `pos_embed.py` | Positional embedding interpolation for fine-tuning at different resolutions |
| `crop.py` | Image cropping utility |
| `utils.py` | General utilities (argument export, resource monitoring) |

### Shared pre-processing (from the Diagnosis pipeline)

| Module | Description |
|--------|-------------|
| `Codes/Tutorial-ModalDecomp/SVD/mainSVD_orig.py` | SVD analysis of echo sequences |
| `Codes/Tutorial-ModalDecomp/HODMD/mainHODMD_IT.py` | HODMD-IT iterative decomposition |
| `Codes/Train-test/Normalization/normalization_orig.py` | Dataset mean/std computation |
| `LowCostHOSVD_GOODalgorithm/` | Low-Cost HOSVD MATLAB scripts |

## Key Parameters Reference

### Dual-mode switching

| `gt_outcome_directory` | Operating mode | Loss |
|------------------------|----------------|------|
| `None` | Classification | CrossEntropy / LabelSmoothing |
| `/path/to/outcomes` | Regression | MSELoss / WeightedMSELoss |

### Most commonly changed parameters

| Parameter | File | Description |
|-----------|------|-------------|
| `training_database_path` | train script | Root folder(s) of the training split |
| `validation_database_path` | train / test scripts | Root folder(s) of the val / test split |
| `gt_outcome_directory` | both scripts | Path to scalar outcome labels |
| `gt_outcome_scale_factor` | both scripts | Rescaling factor for outcome values |
| `output_dir` | train script | Where model checkpoints are saved |
| `model_path_test` | test script | Checkpoint to load for evaluation |
| `device` | both scripts | `'cuda'`, `'cuda:0'`, or `'cpu'` |
| `finetune` | train script | Pre-trained checkpoint for fine-tuning |
| `mean` / `std` | both scripts | Dataset statistics for normalisation |

## See also

- [TUTORIAL.md](../TUTORIAL.md)
- [examples/](../examples/)
- [notebooks/](../notebooks/)
