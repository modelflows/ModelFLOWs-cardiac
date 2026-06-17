# EchoNet Cardiac Analysis Tutorial

Left-ventricle (LV) segmentation and ejection-fraction (EF) prediction from
echocardiograms, built on the [EchoNet-Dynamic](https://github.com/echonet/dynamic)
deep-learning models. The tutorial walks through three ready-to-run scripts.

📖 **Read the tutorial online:** <https://zhuoqun.github.io/Segmentation-and-EF-prediction/>

## What's inside

```
EchoNet-dynamic/
├── scripts/        PredictSegmentation.py · PredictEF.py · CalculateStats.py
└── demo/           a sample echo video and .npy frames
sphinx/sphinx/       the Sphinx tutorial source (source/ + Makefile)
```

## Model weights & dataset labels (not committed)

The pretrained `.pt` weights (>100 MB each) and the dataset label CSVs are **not**
stored in this repo:

- **Weights** → [EchoNet-Dynamic releases](https://github.com/echonet/dynamic/releases)
  — save them as `EchoNet-dynamic/stats/r2plus1d_18_32_2_pretrained.pt` and
  `EchoNet-dynamic/stats/deeplabv3_resnet50_random.pt` (the script `WEIGHTS` paths
  already use these names).
- **`FileList.csv` / `VolumeTracings.csv`** → from the
  [EchoNet-Dynamic dataset](https://echonet.github.io/dynamic/) (used for validation).

## Build the tutorial locally

```bash
pip install -r requirements.txt
sphinx-build -b html sphinx/sphinx/source sphinx/sphinx/build/html
```

Then open `sphinx/sphinx/build/html/index.html`.

The site is also built and published automatically by GitHub Actions
(`.github/workflows/docs.yml`) on every push to `main`.
