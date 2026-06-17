# Introduction

## What this tutorial covers

This tutorial documents two prediction scripts — plus an optional statistics
helper — for **automated analysis of echocardiograms** (cardiac ultrasound).
Both prediction scripts are thin, easy-to-edit wrappers around the deep-learning
models released with the
[EchoNet-Dynamic](https://github.com/echonet/dynamic) project from Stanford.

| Script | Task | Model | Main output |
|--------|------|-------|-------------|
| `PredictSegmentation.py` | Left-ventricle (LV) segmentation | DeepLabV3-ResNet50 | Masks, area curves, annotated videos, CSV |
| `PredictEF.py` | Ejection-fraction (EF) prediction | R(2+1)D-18 video network | One EF value per video, CSV |
| `CalculateStats.py` | Per-channel `MEAN`/`STD` for normalisation | — | Printed stats to paste into the two scripts |

```{admonition} Who is this for?
:class: tip
Anyone who has a folder of echocardiogram videos (or `.npy` frames) and wants to
(a) trace the left-ventricle contour over time, or (b) estimate the ejection
fraction — **without** retraining any model. You only edit a few path variables
and run the script.
```

---

## Background: what the models do

### Left-ventricle segmentation

The left ventricle is the heart's main pumping chamber. **Segmentation** means
labelling, pixel by pixel, which part of each ultrasound frame belongs to the LV
cavity. From the segmented mask we can measure the LV **area** in every frame.
As the heart beats, this area rises (filling, *diastole*) and falls (contraction,
*systole*). This curve can be used to calculate the heart rate by Fast Fourier transform (FFT).

### Ejection fraction (EF)

The **ejection fraction** is the percentage of blood pumped out of the left
ventricle with each beat:

```{math}
\mathrm{EF} = \frac{V_{\text{diastole}} - V_{\text{systole}}}{V_{\text{diastole}}} \times 100\%
```

It is one of the most important clinical measures of heart function. A normal EF
is roughly **50–70 %**; lower values can indicate heart failure. `PredictEF.py`
predicts EF end-to-end from a video, so it frees people from measuring volumes by hand.

```{note}
The `Status` column in the EF output uses simple thresholds: *Low* (< 50 %),
*Normal* (50–70 %), or *High* (> 70 %). This is a coarse screening heuristic,
**not** a clinical diagnosis.
```

---

## The end-to-end workflow

```
                 (optional) CalculateStats.py ──► MEAN / STD
                                                       │ paste into both scripts
                                                       ▼
Echo videos / .npy frames
          │
          ├─────────────► PredictSegmentation.py ──► masks + LV area + systole + CSV
          │
          └─────────────► PredictEF.py ────────────► predicted EF (%) + CSV
```

The two scripts are independent — you can run either one on its own. They adopt different weights, named `deeplabv3_resnet50_random.pt` and `r2plus1d_18_32_2_pretrained.pt`, respectively.

---

## How the scripts are configured

Neither script takes command-line arguments — you edit a small **path block at
the top of the file**. Paths are written as `...\EchoNet-dynamic\…`; replace the
leading `...` with the folder that holds `EchoNet-dynamic` on your machine. For
example, in `PredictSegmentation.py`:

```python
INPUT_DIR  = r"...\EchoNet-dynamic\demo\video"
WEIGHTS    = r"...\EchoNet-dynamic\stats\deeplabv3_resnet50_random.pt"
OUTPUT_DIR = r"...\EchoNet-dynamic\demo_output"
```

and in `PredictEF.py`:

```python
INPUT_DIR  = r"...\EchoNet-dynamic\demo\video"
WEIGHTS    = r"...\EchoNet-dynamic\stats\r2plus1d_18_32_2_pretrained.pt"
OUTPUT_CSV = r"...\EchoNet-dynamic\demo_output\EFprediction.csv"
```

---

## What you need before starting

- [ ] A Python environment with PyTorch (see [Installation](02_installation))
- [ ] The two pretrained weight files (`r2plus1d_18_32_2_pretrained.pt`, `deeplabv3_resnet50_random.pt`)
- [ ] A folder of echocardiogram videos (`.avi` / `.mp4`) and/or `.npy` frames
- [ ] (Optional but recommended) an NVIDIA GPU with CUDA
