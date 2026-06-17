# Installation & Setup

This chapter gets your environment ready to run both scripts.

## Create a Python environment

A dedicated [Conda](https://docs.conda.io/) environment which keeps the dependencies
isolated is recommended. Python 3.9 is a safe choice for the EchoNet-Dynamic stack. For example, we name the environment as `echonet_dynamic`.

```bash
conda create -n echonet_dynamic python=3.9
conda activate echonet_dynamic
```

```{admonition} Windows note
:class: note
On this machine Conda lives at `C:\Users\model\miniconda3`. If the `conda`
command is not recognised, open the **"Anaconda Prompt"** from the Start menu, or
first run `C:\Users\model\miniconda3\Scripts\activate.bat`.
```

---

## Install PyTorch

Install PyTorch **matching your CUDA version** for GPU acceleration. Check the
[official selector](https://pytorch.org/get-started/locally/) for the exact
command. A typical CUDA 11.8 install:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

CPU-only (works, but video processing will be slow):

```bash
pip install torch torchvision
```

Verify the GPU is visible:

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

---

## Install the remaining dependencies

Both scripts rely on the following packages:

| Package | Used for | Needed by |
|---------|----------|-----------|
| `torch`, `torchvision` | Models and inference | both |
| `opencv-python` (`cv2`) | Reading videos / images, drawing | both |
| `numpy` | Array maths | both |
| `scipy` | Systole peak detection (`scipy.signal`) | segmentation |
| `scikit-image` (`skimage`) | Drawing the trace marker on videos | segmentation |
| `matplotlib` | Saving the area-vs-time plots | segmentation |
| `tqdm` | Progress bars | segmentation, stats |
| `pandas` | Reading `FileList.csv` | `CalculateStats.py` |
| `echonet` | Saving annotated videos (`echonet.utils.savevideo`) | segmentation |

Install them in one go:

```bash
pip install opencv-python numpy scipy scikit-image matplotlib tqdm pandas
```

### Installing the `echonet` package

`PredictSegmentation.py` imports `echonet` to write annotated videos. Install it
from the official repository:

```bash
git clone https://github.com/echonet/dynamic.git
cd dynamic
pip install -e .
```

```{admonition} Only need EF prediction?
:class: tip
`PredictEF.py` does **not** import `echonet`. If you only run EF prediction you
can skip the `echonet`, `scipy`, `skimage`, `matplotlib` and `tqdm` installs.
```

---

## Obtain the pretrained weights

Both scripts load a pretrained `.pt` checkpoint. Download both from the
[EchoNet-Dynamic releases page](https://github.com/echonet/dynamic/releases)
(also linked from the [Downloads](08_downloads) page). Save them into the
project's **`stats/`** folder under the exact names below — that is the location
each script's `WEIGHTS` path points at (`...\EchoNet-dynamic\stats\…`).

| Script | Expected weight | Architecture |
|--------|-----------------|--------------|
| `PredictSegmentation.py` | `deeplabv3_resnet50_random.pt` | DeepLabV3-ResNet50 |
| `PredictEF.py` | `r2plus1d_18_32_2_pretrained.pt` | R(2+1)D-18 |

```{admonition} Use the matching checkpoint
:class: warning
The segmentation script builds a `deeplabv3_resnet50` head with **1 output
channel**, and the EF script builds an `r2plus1d_18` with a **1-unit final
linear layer**. Loading the wrong checkpoint into either will raise a
`state_dict` size-mismatch error.
```

---

## The project layout

The project is a single self-contained folder. With the scripts, weights and
demo data in place it looks like this:

```text
EchoNet-dynamic/
├── scripts/
│   ├── PredictSegmentation.py
│   ├── PredictEF.py
│   └── CalculateStats.py
├── stats/
│   ├── deeplabv3_resnet50_random.pt
│   └── r2plus1d_18_32_2_pretrained.pt
└── demo/
    ├── video/   0X243FDE8AE0A05B6F.avi
    └── npy/     frame_0000.npy … (138)
```

Keep `scripts/`, `stats/` and `demo/` with these names so the paths shown in each
script match. Don't have the files yet? Grab the scripts, weights and demo data
from the [Downloads](08_downloads) page.

You are now ready to prepare your input data
([next chapter](03_data_preparation)).
