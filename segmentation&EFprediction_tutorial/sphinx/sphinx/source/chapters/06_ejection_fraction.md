# Ejection-Fraction Prediction

This chapter covers `PredictEF.py` — it predicts the ejection fraction (EF) of
the left ventricle directly from echocardiogram videos.

## Configure the paths

Edit the configuration block at the **top of `PredictEF.py`**:

```python
INPUT_DIR  = r"...\EchoNet-dynamic\demo\video"
WEIGHTS    = r"...\EchoNet-dynamic\stats\r2plus1d_18_32_2_pretrained.pt"
OUTPUT_CSV = r"...\EchoNet-dynamic\demo_output\EFprediction.csv"

MEAN = [0.128, 0.129, 0.130]
STD  = [0.196, 0.196, 0.197]
```

| Variable | Meaning |
|----------|---------|
| `INPUT_DIR` | Folder scanned for `.avi`/`.mp4` videos |
| `WEIGHTS` | R(2+1)D-18 checkpoint (`stats\r2plus1d_18_32_2_pretrained.pt`) |
| `OUTPUT_CSV` | Output CSV file path (its folder is created automatically) |
| `MEAN`, `STD` | Per-channel normalisation stats (0–1 scale) |

To run on your **own** videos, point `INPUT_DIR` at your folder.

Keep the default `MEAN`/`STD` for EchoNet-style data, or recompute them for your
own dataset — see [Normalization Statistics](04_normalization_stats).

The script's `main()` also accepts a `device_name` argument (default `'cuda'`),
which automatically falls back to CPU when no GPU is present.

---

## Run the script

```bash
conda activate echonet
python scripts/PredictEF.py
```

Typical output:

```text
Using device: cuda for inference...
Successfully located 12 videos. Starting prediction...
[patient001.avi] -> contains 145 clips, overall predicted EF: 58.42% (Normal)
[patient002.avi] -> contains 98 clips, overall predicted EF: 34.10% (Low)
[patient003.avi] -> contains 120 clips, overall predicted EF: 73.80% (High)
...
Saving prediction results to: ...\EchoNet-dynamic\demo_output\EFprediction.csv
✅ Save complete!
```

---

## What happens under the hood

### Model

```python
model = torchvision.models.video.r2plus1d_18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 1)   # 1 output = EF regression
```

This is a **spatio-temporal** network: it takes short video clips (not single
frames) and regresses a single EF number.

### Clip sampling (test-time augmentation)

Each video is turned into many overlapping clips. With defaults `frames=32` and
`period=2`, a clip spans `32 × 2 = 64` raw frames, sampled every 2nd frame:

```python
target_length = self.frames * self.period          # 64
for start in range(T - target_length + 1):          # one clip per start offset
    clip = video_tensor[start : start + target_length : self.period]
```

So a long video produces hundreds of clips. If a video has fewer than 64 frames,
it is padded and treated as a single clip.

Each clip is scaled to `[0, 1]` (divided by 255) and then normalised with the
configured `MEAN`/`STD`:

```python
clips = np.array(clips).astype(np.float32) / 255.0
clips = (clips - self.mean) / self.std
```

### Prediction and averaging

For each video, clips are fed to the GPU in chunks (`block_size = 10`) to avoid
running out of memory, and the per-clip EF predictions are **averaged** into one
final value:

```python
pred_ef = np.mean(pred_list)
if pred_ef < 50:
    status = "Low"
elif pred_ef <= 70:
    status = "Normal"
else:
    status = "High"
```

```{admonition} Tune the chunk size for your GPU
:class: tip
`block_size = 10` is conservative. If you have ≥12 GB VRAM you can raise it to 20
or 40 (in the `main()` function) to speed things up. Lower it if you hit
out-of-memory errors.
```

---

## Output

A single CSV (`OUTPUT_CSV`) with **one row per video**:

| Column | Meaning |
|--------|---------|
| `FileName` | Source video name |
| `Predicted_EF(%)` | Averaged EF prediction, rounded to 2 decimals |
| `Status` | `Low` (< 50), `Normal` (50–70), or `High` (> 70) |

The file is written with `utf-8-sig` encoding so it opens cleanly in Excel.

```{admonition} Clinical disclaimer
:class: warning
The `Status` label is a fixed-threshold screening heuristic (Low / Normal / High),
not a diagnosis. Always have predictions reviewed by a qualified clinician before
drawing conclusions.
```

Continue to [Results & Troubleshooting](07_results_troubleshooting).
