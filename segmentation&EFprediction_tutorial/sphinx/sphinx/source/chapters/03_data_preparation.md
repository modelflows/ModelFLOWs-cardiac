# Data Preparation

Both scripts read **a folder of files** and process everything they find. This
chapter explains the accepted formats and how each script scans the folder.

## Accepted input formats

| Format | Extension | Segmentation | EF prediction |
|--------|-----------|:------------:|:-------------:|
| Video | `.avi`, `.mp4` | ✅ | ✅ |
| NumPy array (single frame) | `.npy` | ✅ | ❌ |

```{admonition} Key difference
:class: important
`PredictSegmentation.py` auto-detects **both** videos and `.npy` files in the
same folder and handles each with a dedicated module. `PredictEF.py` only looks
for `.avi`/`.mp4` videos — EF is a *temporal* measurement and needs a full clip.
```

---

## How each script scans the folder

### Segmentation

```python
all_files   = glob.glob(os.path.join(input_dir, "*"))
video_files = [f for f in all_files if f.lower().endswith(('.avi', '.mp4'))]
npy_files   = [f for f in all_files if f.lower().endswith('.npy')]
```

So a single `INPUT_DIR` may mix videos and `.npy` frames; each group is processed
separately. (There are two modules in the code. Module A for `.npy`, Module B for videos.)

### EF prediction

```python
self.video_paths = glob.glob(os.path.join(video_dir, "*.avi")) + \
                   glob.glob(os.path.join(video_dir, "*.mp4"))
```

Only top-level `.avi`/`.mp4` files are picked up. **Sub-folders are not searched
recursively** — put your videos directly inside the input folder.

---

## Image / video assumptions

The scripts were built around EchoNet-Dynamic data, so they expect:

- **Apical-4-chamber (A4C)** echocardiogram views for meaningful EF.
- Frames are internally resized to **112 × 112** pixels, so the source
  resolution does not need to match — but very low quality inputs degrade
  accuracy.
- Videos are read with OpenCV; for EF, each frame is converted to grayscale and
  back to RGB (the model expects 3 identical channels).

### `.npy` frame handling (segmentation only)

Module A is deliberately tolerant of `.npy` layouts. It normalises any array to a
`(H, W, 3)` uint8 image:

- 2-D `(H, W)` grayscale → stacked into 3 channels
- channel-first `(C, H, W)` → transposed to `(H, W, C)`
- single-channel `(H, W, 1)` → repeated to 3 channels
- pixel values are min–max scaled to `0–255`

Anything with an unsupported number of dimensions is skipped with a warning.

---

## Normalisation constants

Every frame is scaled to `[0, 1]` and then normalised with a per-channel **mean**
and **standard deviation**, defined in a single `MEAN`/`STD` block at the top of
each script:

```python
MEAN = [0.128, 0.129, 0.130]
STD  = [0.196, 0.196, 0.197]
```

These same statistics are used consistently across **all** paths:

| Script / module | Mean (RGB) | Std (RGB) | Scale |
|-----------------|-----------|-----------|-------|
| Segmentation — `.npy` | `MEAN` | `STD` | 0–1 |
| Segmentation — video | `MEAN` | `STD` | 0–1 |
| EF — video | `MEAN` | `STD` | 0–1 |

```{admonition} Recompute for your own data
:class: note
The defaults come from the EchoNet-Dynamic training set. If your data looks
different, recompute `MEAN`/`STD` with `CalculateStats.py` and paste the result
into both scripts — see [Normalization Statistics](04_normalization_stats).
```

---

## Project folder layout

If you keep this layout, the paths in the scripts will match your files as-is:

```text
EchoNet-dynamic/
├── scripts/                     # the three .py scripts
├── stats/                       # model weights
│   ├── deeplabv3_resnet50_random.pt
│   └── r2plus1d_18_32_2_pretrained.pt
└── demo/                        # input data (replace with your own)
    ├── video/   *.avi / *.mp4
    └── npy/     *.npy            # only used by the segmentation script
```

To run on your own study, you can easily drop your videos into `demo/video/` (and any `.npy`
frames into `demo/npy/`), or just point `INPUT_DIR` at a folder of your choice.

With the data in place, continue to
[LV Segmentation](05_segmentation) or [EF Prediction](06_ejection_fraction).
