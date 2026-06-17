# Results & Troubleshooting

The quickest check is to eyeball the overlays and area curves. For a
**quantitative** one, lean on the EchoNet-Dynamic dataset: it ships a ground-truth
EF for all **10,030** videos, plus the traced LV boundary on the systole and
diastole frames of one cardiac cycle.

## Validating against the EchoNet-Dynamic ground truth

That ground truth lives in two files, both are on the [Downloads](08_downloads) page.

| File | Ground truth it provides |
|------|--------------------------|
| `FileList.csv` | The true **EF** (and `ESV`/`EDV`) for every video |
| `VolumeTracings.csv` | The traced **LV boundary** on the end-diastole (ED) and end-systole (ES) frames of one cardiac cycle |


### EF — compare against `FileList.csv`

Run `PredictEF.py` on the videos (ideally the **test** split, so the model is
scored on clips it never trained on), then join your `EFprediction.csv` to
`FileList.csv` on `FileName` and measure the error:

```python
import pandas as pd
pred = pd.read_csv(r"...\EchoNet-dynamic\demo_output\EFprediction.csv")
gt   = pd.read_csv(r"...\EchoNet-dynamic\FileList.csv")
pred["FileName"] = pred["FileName"].str.replace(".avi", "", regex=False)
m = pred.merge(gt[["FileName", "EF"]], on="FileName")
print(f"EF MAE = {(m['Predicted_EF(%)'] - m['EF']).abs().mean():.2f}%")
```

For reference, the original EchoNet-Dynamic model reaches a mean absolute error of
about **4 %** on the test split; landing near that means your pipeline is sound.

### Segmentation — compare against `VolumeTracings.csv`

`VolumeTracings.csv` labels exactly **two frames** per video — the ED frame
(largest LV) and the ES frame (smallest LV). The first row of each frame is the
LV long axis; the rest are chords `(X1, Y1)`–`(X2, Y2)` spanning the cavity that
together trace the ventricle outline. That gives you two checks:

- **Systole timing** — the ES frame should coincide with a frame your script flags
  `ComputerSystole = 1` in `video_lv_area.csv`, while the ED frame should fall on a
  peak of the `Size` curve.
- **Boundary overlap** — turn the tracing into a filled mask and score it against
  your predicted mask on the same frame with the Dice coefficient:

```python
import numpy as np
from skimage.draw import polygon

def tracing_to_mask(rows, size=112):
    # rows: VolumeTracings rows for ONE (FileName, Frame); skip row 0 (long axis)
    chords = rows.iloc[1:]
    x = np.concatenate([chords.X1.values, chords.X2.values[::-1]])
    y = np.concatenate([chords.Y1.values, chords.Y2.values[::-1]])
    rr, cc = polygon(y, x, (size, size))
    m = np.zeros((size, size), np.uint8); m[rr, cc] = 1
    return m

def dice(a, b):
    return 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum())
```

On the ED and ES frames, a Dice above ~0.9 means your segmentation closely follows
the human tracing.

```{admonition} Coordinates are already at 112 × 112
:class: note
The released EchoNet-Dynamic videos and the `VolumeTracings.csv` coordinates are
both at `112 × 112` — the same scale the scripts use — so the masks line up
directly. If you switch to full-resolution clips, rescale the `X/Y` values (or
your mask) to a common size first.
```

---

## Common errors

```{admonition} Error: state_dict size mismatch when loading weights
:class: error
You loaded the wrong checkpoint. `PredictSegmentation.py` needs the
**DeepLabV3-ResNet50** weights; `PredictEF.py` needs the **R(2+1)D-18** weights.
See the table in [Installation](02_installation).
```

```{admonition} "No .avi or .mp4 videos found"
:class: error
The input folder path is wrong, empty, or the videos sit in a sub-folder. Both
scripts only scan the **top level** of the folder. Put videos directly inside it.
```

```{admonition} ModuleNotFoundError: No module named 'echonet'
:class: error
Only `PredictSegmentation.py` needs `echonet` (to save annotated videos). Install
it from the EchoNet-Dynamic repo (see [Installation](02_installation)), or run
only `PredictEF.py` if you do not need segmentation videos.
```

```{admonition} CUDA out of memory
:class: error
- EF: lower `block_size` in `PredictEF.py` (e.g. 10 → 4).
- Segmentation: lower the video batch size (`batch_size = 20`) in Module B.
- Or force CPU: set `device_name='cpu'` / let the script detect no GPU.
```

```{admonition} Emoji / encoding errors in the Windows console
:class: error
`PredictSegmentation.py` already calls `sys.stdout.reconfigure(encoding="utf-8")`.
If you still see `UnicodeEncodeError`, run from Windows Terminal / PowerShell
rather than the legacy `cmd.exe`, or set `set PYTHONUTF8=1` before running.
```

```{admonition} Predictions look wrong but no error is raised
:class: warning
Most often a **domain/normalisation mismatch**: the inputs differ from
EchoNet-Dynamic data (different scanner, view, contrast). The default `MEAN`/`STD`
assume EchoNet-like grayscale A4C clips — recompute them for your own dataset, see
[Normalization Statistics](04_normalization_stats).
```

---

## Performance tips

- A CUDA GPU is roughly 10–50× faster than CPU for these models.
- For EF, larger `block_size` = fewer kernel launches = faster (until you hit a
  memory limit).
- For segmentation, the video module is the slow part; `.npy` frames are quick.

---

## Reproducibility checklist

- [ ] Record which checkpoint (`r2plus1d_18_32_2_pretrained.pt` / `deeplabv3_resnet50_random.pt`) produced each result.
- [ ] Keep the input folder unchanged, or copy it alongside the output.
- [ ] Note the script version / date (the segmentation header is dated 5/5/2026).
- [ ] Archive the output CSVs together with the annotated media.
