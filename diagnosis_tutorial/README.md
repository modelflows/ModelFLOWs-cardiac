# AI-Based Cardiac Diagnosis from Echocardiography

End-to-end workflow for automatic diagnosis of cardiac pathologies from echocardiography
image sequences, combining modal decomposition (SVD, HODMD) with deep learning
(Masked Autoencoder pre-training + Vision Transformer classification).

## Repository Structure

```
TUTORIAL.md           Main step-by-step tutorial
docs/
    index.md          Module reference documentation
examples/             Example scripts and configuration files
notebooks/            Interactive Jupyter notebooks
images/               Figures referenced in the tutorial
```

## Quick Start

1. Organise your echocardiography dataset as `class/sequence/frames/*.npy`.
2. Run normalisation: `python Codes/Train-test/Normalization/normalization_orig.py`.
3. (Optional) Apply HODMD pre-processing: `python Codes/Tutorial-ModalDecomp/HODMD/mainHODMD_IT.py`.
4. Pre-train MAE: `python Codes/Train-test/Train-test/train_mae__demo-modelflowsapp_diagnosis__orig.py`.
5. Train ViT classifier: `python Diagnosis_scripts/train_vit_modelflows-app.py`.
6. Evaluate: `python Diagnosis_scripts/test_vit_modelflows-app.py`.

See [TUTORIAL.md](TUTORIAL.md) for the full detailed tutorial.

## Requirements

Install Python dependencies:

```bash
pip install -r Diagnosis_scripts/requirements.txt
```

Key libraries: TensorFlow 2.13, TensorFlow Addons 0.23, PyTorch, timm, einops,
scikit-learn, OpenCV, pydicom, h5py, wandb.

For the MATLAB Low-Cost HOSVD scripts, MATLAB R2021b or later is required.
