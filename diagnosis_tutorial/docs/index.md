# AI-Based Cardiac Diagnosis from Echocardiography — Documentation

This documentation accompanies the full step-by-step tutorial for automatic diagnosis
of cardiac pathologies from echocardiography image sequences.

## Contents

- [TUTORIAL.md](../TUTORIAL.md) — Main tutorial: data preparation, modal decomposition,
  MAE pre-training, ViT fine-tuning, and evaluation.

## Modules

### Data utilities (`Diagnosis_scripts/Utils/`)

| Module | Description |
|--------|-------------|
| `datasets.py` | Dataset loader for organised `.sequences` folders; data augmentation pipeline |
| `utils.py` | General utility functions (argument export, etc.) |

### Training (`Diagnosis_scripts/Training/`)

| Module | Description |
|--------|-------------|
| `model.py` | ViT classifier with Shifted Patch Tokenization and Locality Self-Attention |
| `batch_generator.py` | Class-balanced mini-batch generator |
| `callbacks.py` | Keras callbacks: warmup cosine-decay LR, TensorBoard, checkpointing, visualisation |

### Testing (`Diagnosis_scripts/Testing/`)

| Module | Description |
|--------|-------------|
| `results_functions.py` | Confusion matrix plotting and classification report export |

### Modal decomposition (`Codes/Tutorial-ModalDecomp/`)

| Module | Description |
|--------|-------------|
| `SVD/mainSVD_orig.py` | SVD analysis of echocardiography sequences |
| `HODMD/mainHODMD_IT.py` | HODMD-IT: iterative Higher-Order DMD on tensor data |
| `HODMD/DMDd.py` | DMD-d core algorithm and reconstruction utilities |
| `HODMD/hosvd.py` | HOSVD tensor decomposition |

### Low-Cost HOSVD (`LowCostHOSVD_GOODalgorithm/`)

| Module | Description |
|--------|-------------|
| `hosvd.m` | Standard HOSVD |
| `hosvd_lc_modo3.m` | Low-cost HOSVD — 3-D tensors |
| `hosvd_lc_multi2D.m` | Low-cost HOSVD — 4-D tensors |
| `hosvd_lc_multi3D.m` | Low-cost HOSVD — 5-D tensors |

### MAE pre-training (`Codes/Train-test/Train-test/`)

| Module | Description |
|--------|-------------|
| `train_mae__demo-modelflowsapp_diagnosis__orig.py` | MAE pre-training script (PyTorch) |
| `test_mae__demo-modelflowsapp_diagnosis__orig__orig-test.py` | MAE reconstruction test |
| `model_mae_image_loss.py` | MAE encoder–decoder architecture |
| `engine_two_branch.py` | Training and evaluation loops |

## See also

- [TUTORIAL.md](../TUTORIAL.md)
- [examples/](../examples/)
- [notebooks/](../notebooks/)
