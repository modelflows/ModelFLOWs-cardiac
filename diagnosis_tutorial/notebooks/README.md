# Notebooks

Interactive Jupyter notebooks that reproduce the key steps of the tutorial.

## Available Notebooks

| Notebook | Location | Description |
|----------|----------|-------------|
| `train_diagnosis_modelflows_app.ipynb` | `Diagnosis_scripts/` | ViT training walkthrough with inline commentary |
| `test_diagnosis_modelflows_app.ipynb` | `Diagnosis_scripts/` | Model evaluation, confusion matrices, per-sequence analysis |
| `view_images_npy.ipynb` | `Diagnosis_scripts/` | Utility to visualise `.npy` echocardiography frames |

## Running the Notebooks

```bash
cd Diagnosis_scripts
jupyter notebook
```

Open the desired notebook in the browser. The testing notebook requires a trained
model checkpoint (either your own or the pre-trained weights provided in
`Diagnosis_scripts/pre-trained-vit.h5`).

> Adjust the dataset and model paths in the first cell of each notebook to match your
> local file system before executing.
