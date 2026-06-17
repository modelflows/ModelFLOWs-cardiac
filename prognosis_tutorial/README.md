# AI-Based Cardiac Prognosis from Echocardiography

End-to-end workflow for cardiac prognosis from echocardiography image sequences using
a Masked Autoencoder (MAE). Supports both **regression** (predicting a continuous
clinical outcome) and **classification** (condition labelling), controlled by a single
parameter.

## Repository Structure

```
TUTORIAL.md           Main step-by-step tutorial
docs/
    index.md          Module reference and parameter guide
examples/             Example configuration files
notebooks/            Interactive Jupyter notebooks
images/               Figures referenced in the tutorial
```

## Quick Start

1. Organise your echo dataset as `class/sequence/frames/*.npy`.
2. (Optional) Apply modal decomposition preprocessing — see Diagnosis tutorial.
3. Set `param['gt_outcome_directory']` to your outcome labels folder (regression), or
   leave as `None` (classification).
4. Pre-train the MAE:
   ```bash
   python Prognosis_scripts/train_mae__demo-modelflowsapp.py
   ```
5. Fine-tune for prognosis (set `param['finetune']` to the pre-trained checkpoint and
   re-run the same script).
6. Evaluate:
   ```bash
   python Prognosis_scripts/test_mae__demo-modelflowsapp.py
   ```

See [TUTORIAL.md](TUTORIAL.md) for the full detailed tutorial.

## Requirements

```bash
pip install -r Prognosis_scripts/requirements.txt
```

Key libraries: PyTorch 2.5, torchvision, timm 1.0, wandb, scikit-learn, pandas,
matplotlib, seaborn, OpenCV.

## Differences from the Diagnosis Pipeline

| Feature | Diagnosis | Prognosis |
|---------|-----------|-----------|
| Framework | TensorFlow / Keras | PyTorch |
| Model | ViT with SPT + LSA | MAE (`mae_vit_tiny`) |
| Task | Classification only | Classification OR Regression |
| Outcome labels | Folder names | Folder names / `gt_outcome_directory` |
| Pre-trained weights | `pre-trained-vit.h5` | `trained-mae.pth` |
