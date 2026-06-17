# FAQ — Frequently Asked Questions

## Do I need a GPU?

No, but it helps a lot. Both scripts auto-detect CUDA and fall back to CPU. On
CPU the video models (especially EF and the segmentation video module) are much
slower.

## Do I have to use the command line?

Yes — you run `python scripts/PredictSegmentation.py` /
`python scripts/PredictEF.py`. There are **no command-line arguments**;
configuration is done by editing the path variables at the top of each file.

## Can I process a whole folder at once?

Yes. Each script processes **every** matching file in the input folder. Note that
sub-folders are not searched recursively.

## What is the difference between the two scripts?

`PredictSegmentation.py` outlines the LV and measures its area frame by frame
(and works on `.npy` images too). `PredictEF.py` predicts a single ejection
fraction per video. They use different models and different weight files
(`r2plus1d_18_32_2_pretrained.pt` vs `deeplabv3_resnet50_random.pt`).

## Where do I get the weight files?

Download both from the
[EchoNet-Dynamic releases page](https://github.com/echonet/dynamic/releases)
(also linked from the [Downloads](../chapters/08_downloads) page).

## Why are predictions on my data poor?

The models were trained on EchoNet-Dynamic apical-4-chamber clips. Data from a
different scanner, view, or contrast can cause a domain shift. The default
`MEAN`/`STD` assume EchoNet-like grayscale inputs — recompute them on your own
data with `CalculateStats.py` and paste them into both scripts.

## What does `CalculateStats.py` do?

It scans a folder of videos (optionally just the `TRAIN` rows of a
`FileList.csv`), computes the per-channel mean and standard deviation, and prints
`MEAN`/`STD` lines ready to paste into `PredictEF.py` and `PredictSegmentation.py`.
Use it when adapting the models to a new dataset.

## Can I change clip length / sampling for EF?

Yes — `FolderVideoDataset(frames=32, period=2)` controls the clip window. Changing
these alters how many clips are sampled per video; keep them consistent with how
the weights were trained for best accuracy.

## How is "systole" decided in the segmentation script?

By finding the troughs (negative peaks) of the LV-area curve with
`scipy.signal.find_peaks` — peaks at least 20 frames apart, with a prominence of
50 % of the spread between the `n**0.05` and `n**0.95` order statistics of the
frame sizes (the original EchoNet heuristic, not the 5th/95th percentiles). It
only runs on videos with more than 10 frames.

## Is the `Status` column a medical diagnosis?

No. It is a fixed-threshold heuristic — Low (< 50 %), Normal (50–70 %), High
(> 70 %) — for screening only. Clinical interpretation must be done by a qualified
professional.
