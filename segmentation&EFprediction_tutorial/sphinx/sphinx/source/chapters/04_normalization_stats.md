# Computing Normalization Statistics

Both prediction scripts normalise every frame with a per-channel **mean** and
**standard deviation**:

```python
# Top of PredictEF.py and PredictSegmentation.py
MEAN = [0.128, 0.129, 0.130]
STD  = [0.196, 0.196, 0.197]
```

These defaults come from the EchoNet-Dynamic training set and work well on
EchoNet-like data. If your videos come from a **different scanner, view, or
contrast**, recomputing these statistics on your own data can reduce the domain
shift and improve segmentation/EF accuracy. `CalculateStats.py` does exactly
that.

```{admonition} Optional step
:class: note
You only need this chapter if you want to retune the `MEAN`/`STD` for your own
dataset. To just run the models on EchoNet-style data, keep the defaults and skip
ahead to [Segmentation](05_segmentation).
```

---

## Configure the paths

`INPUT_DIR` points at the demo videos; for **meaningful** statistics, point it at
your full dataset instead:

```python
INPUT_DIR = r"...\EchoNet-dynamic\demo\video"   # use your full Videos\ folder for real stats
CSV_PATH  = None                                # or r"...\FileList.csv" (TRAIN split only)
```

| Variable | Meaning |
|----------|---------|
| `INPUT_DIR` | Folder containing the `.avi`/`.mp4` videos to measure |
| `CSV_PATH` | Optional `FileList.csv`; keep `None` to use **every** video |

```{admonition} The demo is just a smoke test — use the full dataset for real stats
:class: warning
Statistics from a single demo video are not representative. For real `MEAN`/`STD`
values, set `INPUT_DIR` to the `Videos/` folder of the open
[EchoNet-Dynamic dataset](https://echonet.github.io/dynamic/) and set `CSV_PATH`
to its `FileList.csv` — that file ships with the dataset and already contains the
`Split` column used to pick the training set.
```

---

## Two modes: training-only vs full scan

The script decides automatically how to pick videos:

- **Training Set Only** — if `CSV_PATH` points to a CSV that has both a
  `FileName` and a `Split` column, only the rows where `Split == TRAIN` are used.
  Filenames without an extension get `.avi` appended, and missing files are
  skipped.
- **Full Dataset** — used when no CSV is given, the file is missing, the required
  columns are absent, or no `TRAIN` rows are found. Every `.avi`/`.mp4` in the
  folder is measured.

```{admonition} Why training-only matters
:class: tip
Statistics should be computed on the **training split only**, never the test set,
to avoid leaking information. Provide a `FileList.csv` with a `Split` column to
keep this clean.
```

---

## Run the script

```bash
conda activate echonet
python scripts/CalculateStats.py
```

It streams every frame, converts BGR→RGB, resizes to `112 × 112`, scales to
`[0, 1]`, and accumulates per-channel sums to compute:

```{math}
\mu = \frac{1}{N}\sum x, \qquad \sigma = \sqrt{\frac{1}{N}\sum x^2 - \mu^2}
```

Typical output (pointed at the full EchoNet-Dynamic dataset):

```text
📄 CSV file detected: ...\EchoNet-Dynamic\FileList.csv
🎯 Found 7460 training videos in CSV. Verifying local files...
📁 Final count: 7460 videos ready for processing. Starting calculation...
============================================================
🎓 Normalization parameters for [Training Set Only] calculated!
============================================================
MEAN = [0.128, 0.129, 0.130]
STD  = [0.196, 0.196, 0.197]
============================================================
```

---

## Use the result

Copy the printed `MEAN` and `STD` lines into the configuration block at the top
of **both** `PredictEF.py` and `PredictSegmentation.py`, replacing the defaults.
The next run of either script will use your dataset's statistics.

```{admonition} Requires pandas
:class: note
This script imports `pandas` to read the CSV. Install it with
`pip install pandas` if you have not already (see [Installation](02_installation)).
```
