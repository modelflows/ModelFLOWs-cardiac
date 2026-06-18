# CardioView — Clinical Echocardiogram Diagnostics

Clinical desktop interface (Flet/Python) for the detection and classification of
cardiac pathologies from echocardiography DICOM videos, using MAE ViT deep learning
models. Supports both human and mouse echocardiography, with an additional cardiac
failure prognosis module for mouse data.

## Repository Structure

```
TUTORIAL.md           Main tutorial: interface walkthrough, tabs and workflows
docs/
    index.md          Code structure reference (modules, scripts, state dictionary)
examples/             Example scripts and configuration files
notebooks/            Interactive Jupyter notebooks
images/               Figures referenced in the tutorial
```

## Quick Start

1. Activate the Python ≥ 3.10 virtual environment with `flet`, `pydicom`,
   `opencv-python`, `torch`, `numpy`, `pandas`, `Pillow` and `torchinfo` installed.
2. Run `python main.py` to start the application.
3. Choose to connect to the cluster or continue offline.
4. In **Tab 1**, select the species (Human/Mouse), load a DICOM video and configure
   the preprocessing pipeline (video / SVD / HODMD / SVD+HODMD).
5. In **Tab 2**, select a model and dataset, then run inference to get the diagnosis.
6. For mouse data, go to **Tab 3** to run the prognosis model.

See [TUTORIAL.md](TUTORIAL.md) for the full detailed tutorial.

## Requirements

Key libraries: Flet 0.84, PyTorch, pydicom, OpenCV, NumPy, pandas, Pillow,
torchinfo. For HODMD pre-processing the `DMDd` and `hosvd` modules must be present
in the preprocessing folder.
