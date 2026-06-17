# AI-Based Cardiac Prognosis from **Mice** Echocardiographies

> **Download the code repository before starting.**
> All scripts, notebooks, and utility modules required by this tutorial are available
> in the following repository:
>
> [Download / Clone the repository](https://github.com/modelflows/cardiac-prognosis-echo)
>
> Once downloaded, keep the folder structure intact — the scripts use relative imports
> that depend on it.

## Overview

This tutorial presents an end-to-end workflow for cardiac prognosis from
echocardiography (echo) image sequences. The pipeline applies a Masked Autoencoder
(MAE) — pre-trained in a self-supervised fashion and then fine-tuned — to predict a
clinical outcome from sequences of ultrasound frames.

The model supports two operating modes:

- **Classification mode** — the model predicts a discrete condition label (e.g.
  healthy vs. pathological) from the echo frames. Class membership is inferred from
  the folder names in the dataset, exactly as in the diagnosis pipeline.
- **Regression mode** — the model predicts a continuous outcome value (e.g. a
  survival time or disease severity score). Ground-truth scalar targets are loaded
  from a separate `gt_outcome_directory`. This is the primary mode for prognosis.

The same preprocessing pipeline (SVD and HODMD modal decomposition) used in the
diagnosis workflow applies here and is not repeated in detail; refer to the Diagnosis
tutorial for Steps 1 and 2.

## Workflow

```
DICOM / .npy sequences
        │
        ▼
Step 1  Data preparation
        (dataset organisation · normalisation)
        │
        ▼
Step 2  Modal decomposition pre-processing   ← see Diagnosis tutorial
        (SVD · HODMD-IT · Low-Cost HOSVD)
        │
        ▼
Step 3  MAE self-supervised pre-training
        (mask 75 % of patches · reconstruct · learn representations)
        │
        ▼
Step 4  Supervised fine-tuning for prognosis
        (classification or regression · AdamW · layer-wise LR decay)
        │
        ▼
Step 5  Evaluation
        (regression: RMSE per class · classification: confusion matrix)
```

### Pre-processing Pipeline Overview

The figure below summarises the modal decomposition pre-processing workflow shared
with the diagnosis pipeline. Starting from the original echocardiography sequence,
a first SVD step (SVD #1) extracts dominant spatial modes and produces low-rank SVD
reconstructions. These are then fed into two parallel branches: HODMD (capturing
temporal cardiac dynamics) and a second SVD pass (SVD #2) applied to the module,
real, and imaginary components of the HODMD modes.

![Pre-processing pipeline](images/preprocess.png)

---

## Step 1. Data Preparation

### 1.1 Dataset Organisation

The dataset must follow the same structure as the diagnosis pipeline:

```
dataset_split/              # e.g. ecos_orig_Training
    class_A/                # cardiac condition name
        sequence_001/
            frame_000.npy
            frame_001.npy
            ...
        sequence_002/
            ...
    class_B/
        ...
```

Each `.npy` file is a 2-D grayscale array representing one ultrasound frame. Split
the data into independent **Training**, **Validation**, and **Test** folders.

### 1.2 Ground-Truth Outcome Directory (Regression Mode Only)

When running in regression mode, a separate directory must be provided that contains
one scalar outcome value per sequence. The directory structure mirrors the dataset:

```
gt_outcome_directory/
    class_A/
        sequence_001.csv   # or .txt with a single numeric value
        sequence_002.csv
    class_B/
        ...
```

Set `param['gt_outcome_directory']` to this path. If left as `None`, the model
switches to classification mode automatically.

The `gt_outcome_scale_factor` parameter rescales the raw outcome values before
training and is reversed during evaluation (default: `100`).

### 1.3 Excluded Classes

Some condition classes may have too few samples to be useful for prognosis (e.g.
`'DB'` in the default configuration). List them in `param['excluded_classes']` and
they will be ignored during both training and testing:

```python
param['excluded_classes'] = ['DB']
```

---

## Step 2. Modal Decomposition Pre-processing

> **Important — mice data only.**
> The prognosis pipeline is designed exclusively for **mouse echocardiography**
> sequences. The `Codes/` folder contains two sets of scripts: scripts without
> `HUMAN` in their name target mouse data, and scripts with `HUMAN` target human
> cardiac data. For prognosis, **always use the non-HUMAN variants** listed below.
> Never use `mainSVD_HUMAN_*.py`, `normalization_HUMAN_orig.py`, or
> `data_load_HUMAN_*.py` in this workflow.

Refer to **Step 2 of the Diagnosis tutorial** for the full algorithmic description.
For prognosis (mice), use the following scripts:

| Step | Script to use | Script to avoid |
|------|---------------|-----------------|
| Normalisation | `Codes/Train-test/Normalization/normalization_orig.py` | `normalization_HUMAN_orig.py` |
| SVD analysis | `Codes/Tutorial-ModalDecomp/SVD/mainSVD_orig.py` | `mainSVD_HUMAN_*.py` |
| HODMD-IT | `Codes/Tutorial-ModalDecomp/HODMD/mainHODMD_IT.py` | `mainHODMD_IT_HUMAN_cesvima.py` |
| Data loading | `Codes/Tutorial-ModalDecomp/SVD/data_load_orig.py` | `data_load_HUMAN_*.py` |
| Low-Cost HOSVD | `LowCostHOSVD_GOODalgorithm/` (all variants) | — |

The reconstructed `.npy` frames produced by these scripts can be used as inputs to
the prognosis training pipeline in place of the original frames.

---

## Step 3. MAE Self-Supervised Pre-training

**Script:** `Prognosis_scripts/train_mae__demo-modelflowsapp.py`

A `mae_vit_tiny` Masked Autoencoder (from the `timm` library) is first pre-trained
without labels. During pre-training, 75 % of image patches are randomly masked and
the encoder–decoder learns to reconstruct the missing pixels. This forces the encoder
to build general visual representations of cardiac structures.

### 3.1 Model Architecture

The model is instantiated from `model_mae_image_loss.py`:

```python
model = models_mae.__dict__[args.model](
    patch_size      = args.patch_size,     # default: 16
    img_size        = args.input_size,     # default: 224
    num_classes     = len(dataset_train.classes),
    regression_mode = (args.gt_outcome_directory is not None),
    norm_pix_loss   = args.norm_pix_loss,
    in_chans        = args.num_channels,   # default: 3
)
```

> **Note:** The model requires `num_channels = 3`. Grayscale `.npy` frames are
> replicated across three channels during data loading.

The combined loss is a weighted sum of the MAE reconstruction loss and the downstream
task loss:

```
L = λ · L_reconstruction + (1 − λ) · L_downstream
```

where `λ = param['lambda_weight']` (default: `0.1`).

### 3.2 Data Augmentation

The augmentation pipeline (built inside `util/datasets.py` via `timm`) varies
depending on whether `is_train` is `True` or `False`:

**Training transforms:**
- Resize to `input_size × input_size`
- RandAugment (`rand-m9-mstd0.5-inc1`) — optional, controlled by `apply_randaugment`
- Random horizontal flip
- Random Erasing (`reprob = 0.25`, `remode = 'pixel'`)
- z-score normalisation using dataset `mean` and `std`

**Validation / Test transforms:**
- Resize only
- z-score normalisation

The dataset statistics must be pre-computed (see Step 1 of the Diagnosis tutorial,
`Codes/Train-test/Normalization/normalization_orig.py`) and set in:

```python
param['mean'] = 113.02734    # example value for the original echo dataset
param['std']  = 676.10596
```

### 3.3 Training Configuration

Edit the `param` dictionary at the top of `train_mae__demo-modelflowsapp.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `training_database_path` | List of training split root folders | — |
| `validation_database_path` | List of validation split root folders | — |
| `gt_outcome_directory` | Path to scalar outcome labels; `None` for classification | — |
| `gt_outcome_scale_factor` | Outcome rescaling factor | `100` |
| `excluded_classes` | Classes to ignore | `['DB']` |
| `output_dir` | Where to save checkpoints | — |
| `device` | `'cuda'`, `'cuda:0'`, `'cpu'` | `'cpu'` |
| `batch_size` | Mini-batch size | `64` |
| `epochs` | Number of training epochs | `100` |
| `model` | MAE variant | `'mae_vit_tiny'` |
| `input_size` | Spatial resolution fed to the model | `224` |
| `patch_size` | ViT patch size | `16` |
| `mask_ratio` | Fraction of patches masked during pre-training | `0.75` |
| `lambda_weight` | Weight of reconstruction loss in the joint objective | `0.1` |
| `blr` | Base learning rate (`lr = blr × batch_size / 256`) | `1e-3` |
| `weight_decay` | AdamW weight decay | `0.05` |
| `warmup_epochs` | LR warm-up epochs | `5` |
| `mean` | Dataset mean for normalisation | `113.02734` |
| `std` | Dataset std for normalisation | `676.10596` |

**Launch pre-training:**

```bash
python Prognosis_scripts/train_mae__demo-modelflowsapp.py
```

The best checkpoint (lowest validation MSE in regression mode, or highest accuracy in
classification mode) is saved automatically to `output_dir`. A pre-trained checkpoint
is already provided at `Prognosis_scripts/trained-mae.pth`.

### 3.4 Optimizer and Learning Rate Schedule

The optimizer is AdamW with layer-wise learning rate decay (following ELECTRA/BEiT):

```
lr_layer_i = base_lr × layer_decay^(num_layers − i)
```

with `layer_decay = 0.75`. The schedule uses a cosine decay after a linear warm-up
of `warmup_epochs` epochs, with a minimum LR floor of `min_lr = 1e-6`.

### 3.5 Loss Functions

| Mode | Loss |
|------|------|
| Regression (`gt_outcome_directory` set) | `MSELoss` (or `WeightedMSELoss` if `weights_classes` is non-empty) |
| Classification (no `gt_outcome_directory`) | `CrossEntropyLoss`, or `LabelSmoothingCrossEntropy` if `smoothing > 0` |
| Classification with Mixup/CutMix active | `SoftTargetCrossEntropy` |

Mixup and CutMix are disabled by default (`mixup = 0`, `cutmix = 0`). To enable them
for classification pre-training set `param['mixup'] = 0.8` and
`param['cutmix'] = 1.0`.

---

## Step 4. Fine-tuning for Prognosis

After pre-training, the encoder weights are loaded as a starting point for
supervised fine-tuning. Set the path to the pre-trained checkpoint:

```python
param['finetune'] = '/path/to/trained-mae.pth'
```

The fine-tuning script is the same (`train_mae__demo-modelflowsapp.py`): when
`param['finetune']` is non-empty and `param['eval']` is `False`, the checkpoint is
loaded, the classification/regression head is re-initialised, and the full model is
fine-tuned end-to-end.

Key considerations when switching from pre-training to fine-tuning:

- Set `param['finetune']` to the best pre-training checkpoint.
- Adjust `param['blr']` downward (e.g. `1e-4`) to avoid large gradient updates that
  destroy the learned representations.
- Reduce `param['epochs']` (e.g. to `50`).
- Make sure `param['gt_outcome_directory']` is correctly set for regression mode.

---

## Step 5. Testing and Evaluation

**Script:** `Prognosis_scripts/test_mae__demo-modelflowsapp.py`

### 5.1 Configuration

Set the following parameters:

```python
param['model_path_test']         = '/path/to/checkpoint-best.pth'
param['validation_database_path'] = ['/path/to/ecos_orig_Test']   # test split
param['gt_outcome_directory']     = '/path/to/ecos_outcome_gt'    # regression
param['gt_outcome_scale_factor']  = 100
param['excluded_classes']         = ['DB']
param['device']                   = 'cuda:0'   # or 'cpu'
param['test_results_path']        = '/path/to/save/results'
param['parallel_prediction']      = False  # True: all samples at once (experimental)
param['export_figures_seq_analysis'] = False  # True: save per-frame confidence plots
```

> In the test script, `validation_database_path` is repurposed as the **test** split
> path (the training path is not used but must be set).

### 5.2 Running the Test

```bash
python Prognosis_scripts/test_mae__demo-modelflowsapp.py
```

Or interactively via notebook:

```bash
jupyter notebook Prognosis_scripts/test_prognosis_modelflows_app.ipynb
```

### 5.3 Inference

Predictions are obtained via the model's `forward_test` method. By default
(`parallel_prediction = False`), samples are predicted one by one to avoid
out-of-memory issues:

```python
with torch.cuda.amp.autocast():
    for i in range(len(dataset_test)):
        output = model.forward_test(
            torch.unsqueeze(dataset_test[i][0], 0).to(device)
        )
        test_predictions[i, :] = output.detach().cpu().numpy()
```

In regression mode the scalar output is rescaled by `gt_outcome_scale_factor` before
computing error statistics. In classification mode the argmax of the output vector
gives the predicted class.

### 5.4 Regression Evaluation

When `gt_outcome_directory` is provided, the following statistics are computed
**per class and globally** and saved to `test_reg_global_stats.csv`:

| Column | Description |
|--------|-------------|
| `Mean_GT` | Mean ground-truth outcome value |
| `Std_GT` | Standard deviation of ground-truth values |
| `Mean_estimated` | Mean predicted outcome value |
| `Std_estimated` | Standard deviation of predicted values |
| `RMSE_estimated` | Root Mean Square Error of the predictions |
| `Max_estimated` | Maximum prediction error |
| `Min_estimated` | Minimum prediction error |

The per-frame and per-sequence prediction details are also exported:

```
test_results_<timestamp>/
    test_prediction_stats.csv      # per-frame: GT, class, estimated outcome, error
    test_seq_prediction_stats.csv  # per-sequence: GT, class, mean predicted outcome
    test_reg_global_stats.csv      # RMSE and error summary per class and global
    prediction_times.csv           # inference time per sample
```

### 5.5 Classification Evaluation

When `gt_outcome_directory = None`, the pipeline produces the same outputs as the
Diagnosis pipeline:

```
test_results_<timestamp>/
    test_prediction_stats.csv              # per-frame GT, prediction, confidence
    test_seq_prediction_stats.csv          # per-sequence aggregated predictions
    confusion_matrix.csv / .png            # frame-level
    confusion_matrix_mean.csv / .png       # sequence-level (mean aggregation)
    confusion_matrix_mean_seq.csv / .png
    confusion_matrix_max.csv  / .png       # sequence-level (max aggregation)
    confusion_matrix_max_seq.csv  / .png
    classification_report_*.csv
    prediction_times.csv
```

When `export_figures_seq_analysis = True`, per-frame confidence plots are also
saved for each test sequence under `seq_analysis/`.

### 5.6 Per-Sequence Aggregation

For **regression**, predictions from all frames of the same sequence are averaged to
obtain a single sequence-level estimate:

```
outcome_seq = mean(outcome_frame_1, ..., outcome_frame_N)
```

For **classification**, two strategies are used:

- **Mean** — argmax of the mean probability vector across all frames.
- **Max** — argmax of the element-wise maximum probability vector across all frames.

---

## Results

A successfully trained regression model returns RMSE values close to the natural
variability of the outcome within each class. The per-class statistics in
`test_reg_global_stats.csv` reveal whether the model generalises across conditions.

In classification mode the confusion matrices and F1 scores provide the standard
diagnostic picture; the per-sequence aggregation (especially mean) typically reduces
the effect of transient frame artefacts and improves overall accuracy.

---

## Related Links

- Notebook (training): `Prognosis_scripts/train_prognosis_modelflows_app.ipynb`
- Notebook (testing): `Prognosis_scripts/test_prognosis_modelflows_app.ipynb`
- Pre-trained checkpoint: `Prognosis_scripts/trained-mae.pth`
- Video: —
- Dataset: —
- Repository: —

---

## Contributors

- Andrés Bell
- Andres Sanchez
- Zhuoquen Zhao
