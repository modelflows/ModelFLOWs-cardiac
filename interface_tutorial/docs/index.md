# CardioView — Code Structure Reference

This documentation describes the complete structure of the CardioView repository,
detailing the function of every file and folder. It is meant as a quick reference;
implementation details are covered in [TUTORIAL.md](../TUTORIAL.md).

## Contents

- [TUTORIAL.md](../TUTORIAL.md) — Main tutorial: application overview, tabs, workflows
  and troubleshooting.

## Root Modules

The files in the application's root folder constitute the core of the interface and
its support systems (cluster communication, retraining, manuals):

| File | Function |
|---|---|
| `main.py` | Application entry point. Configures the Flet window (size, theme, custom title bar), builds the login screen and, after authentication, mounts the three main tabs. Holds the shared state dictionary that connects every component. |
| `theme.py` | Defines the corporate colour palette (`BG_DARK`, `ACCENT_BLUE`, `ACCENT_CYAN`, `SUCCESS`, `WARNING`, `DANGER`), gradients and constructor functions for reusable widgets (cards, gradient buttons, typography). |
| `i18n.py` | Internationalisation system. Contains the `_T` dictionary with `(Spanish, English)` pairs for every UI string, and the helper function `t(key, lang)` that each component calls to get the text in the active language. |
| `cluster.py` | Full SSH/SFTP communication layer with the multi-GPU cluster. Handles login (`test_connection`), selective upload of `.npy` files to the server (`upload_npy_structured`), remote inference launch (`run_inference_remote`), results download (`download_results`), pseudo-label generation and calls to the incremental retraining system. |
| `cluster_backup.py` | Performs incremental backup to the group's storage NAS (Synology MODELFLOWS) after every retraining cycle. Copies the preprocessing and inference folders of users that have already been trained, marks each copy as completed in the corresponding `label.json`, removes already-backed-up data from the cluster, and copies the retrained model checkpoint to the NAS. |
| `data_aggregator.py` | Second phase of the retraining pipeline. Scans the preprocessing directories of all users, collects folders labelled `use_for_training=True` and, once the number of samples reaches the minimum threshold, builds the structured retraining dataset (folders `orig/`, `svd_reconst/`, … up to ten types) with its `training/`, `validation/` and `test/` splits. |
| `label_manager.py` | Manages the `label.json` files associated with each preprocessing folder. Generates the initial pseudo-label from the inference results (`generate`), lets the clinician validate or correct the predicted class (`validate` / `validate_prognosis`) and exposes the stored label for remote reading (`read`). |
| `retrain_launcher.py` | Third phase of the retraining pipeline. Imports the training script for the corresponding task (human diagnosis, mouse diagnosis or mouse prognosis), builds the arguments with the `retraining_data/` paths and the active checkpoint, and launches training directly without going through the CLI. |
| `generate_manual.py` | Generates the LaTeX source of the Spanish user manual (`cardioview_manual.tex`), compilable with Overleaf or `pdflatex`. |
| `generate_manual_en.py` | English version of the script above; produces `cardioview_manual_en.tex`. |
| `visualization.py` | Auxiliary development script to visualise `.npy` frames and DICOM videos while debugging the preprocessing pipeline. Not part of the production flow of the interface. |

## The Shared State Dictionary (`state`)

The `state` dictionary is the central mechanism that keeps the session consistent
across all tabs. It is created in `main.py` and passed by reference to every
component; any relevant change triggers a full rebuild of the interface from its
updated content. The table below documents every key:

| Key | Content |
|---|---|
| `file_path` / `file_name` / `file_size` | Path, name and size of the currently loaded DICOM file. |
| `species` | `"human"` or `"mouse"`: determines diagnostic classes, available models, preprocessing scripts and colour palette. |
| `preprocessed` / `executed` | Booleans indicating whether the preprocessing configuration has been saved and whether the pipeline has run successfully. |
| `preproc_type` / `params` | Chosen pipeline type (`"video"`, `"svd"`, `"hodmd"`, `"svd_hodmd"`) and the full parameter dictionary per step. |
| `lang` | `"es"` or `"en"`: active language of the whole interface. |
| `frames` | NumPy arrays with the decoded DICOM frames, used by the interactive preview. |
| `cluster_user` / `cluster_password` / `cluster_connected` | SSH login credentials and remote session status (`True` when the connection is active). |
| `last_results_path` / `last_model_tokens` / `last_dataset_label` | Traceability of the latest computed result; allows refreshing the visualisation after language or species changes without re-running inference. |

## Folder `components/`

Contains the three modules that build the main tabs of the graphical interface:

| File | Function |
|---|---|
| `tab_carga.py` | «Load & Preprocessing» tab. Manages species selection, DICOM loading and decoding, the interactive preview, and the modal dialog to configure and run the preprocessing pipeline. After a successful run in cluster mode, it automatically triggers the `.npy` upload to the server. |
| `tab_resultados.py` | «Diagnosis» tab. Scans the available models, manages model and dataset selection, runs inference (local or remote depending on the session state), and builds the results visualisation: diagnosis cards, confidence bars, computational metrics, validation section and confusion matrix. |
| `tab_resultados_prognosis.py` | «Prognosis» tab (mouse data only). Reuses the model-selection and inference mechanics of the diagnosis tab, adapted to the regression task: estimates the time to cardiac failure in months and presents it with range and standard deviation. |

## Folder `preprocess/`

Contains the preprocessing scripts for **human echocardiography**. Each one
implements a step of the SVD/HODMD chain described in the
[*Diagnosis tutorial*](https://modelflows.github.io/modelflowsapp/software/tutorials/cardiac-tutorials/#diagnosis-tutorial):

| File | Function |
|---|---|
| `first_step_preprocess_raw.py` | Step 1: extracts frames from the DICOM, detects and crops the region of interest of the heart, applies inpainting, resizes to 224×224 and exports as `.npy` and `.png`. |
| `second_step_svd.py` | Step 2: Singular Value Decomposition on the frame tensor; generates SVD reconstructions and SVD modes. |
| `third_step_hodmd.py` | Step 3: HODMD on the SVD reconstructions from the previous step; generates HODMD reconstructions and DMD modes. |
| `fourth_step_svd.py` | Step 4 (unified file): second SVD on the modulus of the DMD modes. |
| `forth_step_svd_abs.py` | Step 4 — modulus component: SVD on the modulus of the DMD modes. |
| `forth_step_svd_imag.py` | Step 4 — imaginary component: SVD on the imaginary part of the modes. |
| `forth_step_svd_real.py` | Step 4 — real component: SVD on the real part of the DMD modes. |
| `DMDd.py` | Numba-accelerated implementation of standard DMD, used internally by `third_step_hodmd.py`. |
| `hosvd.py` | Implementation of the Higher-Order Singular Value Decomposition (HOSVD), used in the HODMD context. |
| `data_load_HUMAN_orig.py` | Loader of the original `.npy` arrays (step 1) for building the training dataset. |
| `data_load_HUMAN_abs.py` / `imag.py` / `real.py` | Loaders for the modulus, imaginary part and real part arrays of the DMD modes (step 4), respectively. |
| `utils.py` | Auxiliary functions common to the human pipeline (array operations, path management, conversions). |
| `utils_hodmd.py` | HODMD-specific utilities (construction of augmented Hankel matrices, mode normalisation). |

## Folder `mice_preprocess/`

Preprocessing scripts equivalent to those in `preprocess/` but adapted to the
particularities of **mouse echocardiography** (different parameters, overlaid-text
recognition and removal, different metadata handling):

| File | Function |
|---|---|
| `first_step_preprocess_raw.py` | Mouse Step 1: extraction, cleaning and export of frames, with additional detection of overlaid text (lightweight OCR + inpainting). |
| `second_step_svd.py` | Step 2 SVD adapted to the dimensions and parameters of murine sequences. |
| `third_step_hodmd.py` | Step 3 HODMD with default parameters tuned to mouse cardiac dynamics (higher heart rate, fewer snapshots). |
| `fourth_step_svd_abs.py` / `imag.py` / `real.py` | Step 4 second-SVD scripts for each component of the DMD modes (mouse). |
| `DMDd.py` | Same DMD implementation as in `preprocess/`, included here for folder independence. |
| `hosvd.py` | Same for HOSVD. |
| `data_load_orig.py` / `abs.py` / `imag.py` / `real.py` | `.npy` loaders for the four representation types (original, modulus, imaginary, real) in the mouse context. |
| `utils.py` / `utils_hodmd.py` | Generic and HODMD utilities analogous to those in `preprocess/`. |

## Folder `execute_models/`

Contains the inference scripts and the model definition, for both local execution
and remote launch from the cluster:

| File | Function |
|---|---|
| `run_inference.py` | Main diagnosis inference script. Loads the checkpoint, finds every `.npy` under the given data path, predicts the class of each frame and writes the result files (`.csv`). It is invoked both as a local subprocess and remotely on the cluster (`--device cuda`). |
| `run_inference_prognosis.py` | Regression variant of the script above for the mouse prognosis task. Estimates the time to cardiac failure in months from the sequence frames. |
| `model_mae_image_loss.py` | PyTorch definition of the MAE-ViT model with an image-reconstruction-based loss function, used in inference. |
| `debug_npy.py` | Lightweight diagnostic script: inspects the content and shape of the `.npy` files in a preprocessing folder without loading the model. |

**Subfolder `execute_models/util/`:**

| File | Function |
|---|---|
| `datasets.py` | PyTorch `Dataset` classes to load `.npy` files during inference and training; supports the ten representation types. |
| `datasets_backup_cluster.py` | Variant of `datasets.py` aimed at access from the storage cluster during the retraining phase. |
| `results_functions.py` | Functions to compute and write the results CSV files (per-frame and per-sequence predictions, computational metrics). |
| `misc.py` | Miscellaneous utilities: checkpoint loading, TensorBoard metric logging, moving average for loss stabilisation. |
| `pos_embed.py` | Generation of sinusoidal positional encodings for the ViT (2D and 1D). |
| `lr_decay.py` | Construction of parameter groups with layer-wise learning-rate decay, following the BEiT scheme. |
| `lr_sched.py` | Learning-rate scheduler with cosine warmup phase. |
| `lars.py` | LARS (Layer-wise Adaptive Rate Scaling) optimizer without rate scaling for 1D parameters. |
| `crop.py` | TensorFlow/TPU-compatible implementation of `RandomResizedCrop`, used as data augmentation. |
| `utils.py` | General utility functions (byte-to-readable-format conversion, seed management, logging). |

## Folder `retrain_model/`

Contains the training scripts (incremental retraining) and the full definition of
the MAE-ViT models. Shares the `util/` subfolder with `execute_models/`, with
equivalent utilities.

| File | Function |
|---|---|
| `train_mae_diagnosisnewb256_...py` | Retraining script for the diagnosis model with batch size 256, over the full combination of representations (orig, SVD, HODMD, SVD of modes). |
| `train_mae__...py` | Retraining variant of the human diagnosis model with the same combination of representations but a different batch configuration. |
| `train_mae_prognosis_...py` | Retraining script for the mouse prognosis model; additionally requires a directory with the ground-truth values (`gt_outcome/`). |
| `models_mae.py` | Full PyTorch definition of the MAE: asymmetric ViT encoder with random masking, lightweight decoder and per-patch loss function. |
| `models_vit.py` | Definition of the pure Vision Transformer for the classification/regression head, including the Tiny, Small, Base and Large variants. |
| `model_mae_image_loss.py` | MAE variant with a full-image reconstruction loss function (not only masked patches). |
| `engine_two_branch.py` | Dual-branch training loop: combines the MAE reconstruction loss with the classification/regression loss in a single pass. |

**Subfolder `retrain_model/util/`** contains the same modules as
`execute_models/util/` (`datasets.py`, `datasets_backup_cluster.py`,
`results_functions.py`, `misc.py`, `pos_embed.py`, `lr_decay.py`, `lr_sched.py`,
`lars.py`, `crop.py`, `utils.py`), with versions adapted to the full training
context (augmentation support, TensorBoard metric logging, periodic checkpoint
management).

## Model Folders

The three model folders packaged with the application each contain one or more
subfolders named after the pattern `model_mae_<task>__<tokens>__<date>`. Each
subfolder includes the following files:

| File | Content |
|---|---|
| `checkpoint.pth` / `checkpoint-N.pth` | Model weights in PyTorch format. The numbered variant (`checkpoint-0.pth`, `-1.pth`, …) corresponds to models saved at multiple epochs; `checkpoint.pth` is the final or only checkpoint of the cycle. |
| `input_parameters.txt` | Training parameters used: ViT architecture, batch size, epochs, learning rate, representation types, data split, seed, etc. Allows reproducing or auditing the training of each model. |
| `log.txt` | Per-epoch metric log (losses, accuracy, reconstruction loss), generated during training. |
| `events.out.tfevents...` | TensorBoard event file for visualising the training curve. |

The three model folders and their specific contents are:

| Folder | Hosted model |
|---|---|
| `models/` | **Human diagnosis** model (classes NORM, HIPER, ISQ). Contains a subfolder with a single checkpoint (`checkpoint-0.pth`) plus log and parameter files. Trained on the full combination of SVD and HODMD representations. |
| `mice_models/` | **Mouse diagnosis** model (classes CTL, DB, OB, SAH). Contains a subfolder with a single checkpoint (`checkpoint.pth`), trained equally on every available representation. |
| `prognosis_models/` | **Mouse prognosis** model (regression of time to cardiac failure). Contains multiple intermediate checkpoints (`checkpoint-0.pth` to `checkpoint-3.pth`), allowing evaluation of different training points or resuming the cycle from any epoch. |

## Folder `user_manual/`

Contains the user manuals in PDF format generated from the `generate_manual.py` and
`generate_manual_en.py` scripts:

- `manual_usuario_interfaz.pdf` — manual in Spanish.
- `interface_user_manual.pdf` — manual in English.

The interface opens them with the operating system's default PDF viewer when the
corresponding option is selected in the «Settings → User manual» menu, serving as
reference documentation integrated into the application itself.

## See also

- [TUTORIAL.md](../TUTORIAL.md)
- [examples/](../examples/)
- [notebooks/](../notebooks/)
