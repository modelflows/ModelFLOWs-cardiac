# Notebooks

Interactive Jupyter notebooks that reproduce the key steps of the prognosis tutorial.

## Available Notebooks

| Notebook | Location | Description |
|----------|----------|-------------|
| `train_prognosis_modelflows_app.ipynb` | `Prognosis_scripts/` | MAE pre-training and fine-tuning walkthrough with inline commentary |
| `test_prognosis_modelflows_app.ipynb` | `Prognosis_scripts/` | Model evaluation — regression RMSE, per-sequence analysis, confusion matrices |

## Running the Notebooks

```bash
cd Prognosis_scripts
jupyter notebook
```

Open the desired notebook in the browser.

- The **training notebook** will prompt you to configure the dataset paths and choose
  between regression and classification mode by setting `gt_outcome_directory`.
- The **testing notebook** requires a trained checkpoint. The pre-trained weights
  provided in `Prognosis_scripts/trained-mae.pth` can be used directly to skip the
  training step and explore the evaluation results.

> Adjust the dataset paths, `gt_outcome_directory`, and `model_path_test` in the
> first cell of each notebook to match your local file system before executing.

## Notebook vs. Script

Both notebooks are equivalent to their `.py` counterparts
(`train_mae__demo-modelflowsapp.py` and `test_mae__demo-modelflowsapp.py`). Use the
notebooks for interactive exploration and the scripts for batch runs on a cluster.
