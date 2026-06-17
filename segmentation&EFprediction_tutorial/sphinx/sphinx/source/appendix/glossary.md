# Glossary

```{glossary}
Echocardiogram
  An ultrasound scan of the heart. The input to both scripts.

A4C (Apical 4-Chamber)
  A standard echocardiographic view showing all four heart chambers. The view
  the EF model was trained on.

Left Ventricle (LV)
  The heart's main pumping chamber. Its cavity is what the segmentation model
  outlines.

Segmentation
  Pixel-wise labelling of an image — here, marking which pixels belong to the LV
  cavity.

Mask
  The binary image produced by segmentation: `1` inside the LV, `0` elsewhere.

LV Area
  The number of mask pixels in a frame (reported at the 112×112 scale). Used as a
  proxy for cavity size over time.

Systole
  The contraction phase of the heartbeat, when the LV is smallest. Detected as
  the troughs of the area curve.

Diastole
  The filling phase, when the LV is largest.

Ejection Fraction (EF)
  The percentage of blood ejected from the LV per beat. Normal is roughly 50–70 %.

DeepLabV3-ResNet50
  The convolutional segmentation network used by `PredictSegmentation.py`.

R(2+1)D-18
  The spatio-temporal video network used by `PredictEF.py` to regress EF.

Clip
  A short sequence of sampled frames fed to the EF model. Many overlapping clips
  per video are averaged into one EF value.

Checkpoint (.pt)
  A saved file of trained model weights, loaded at runtime (here `r2plus1d_18_32_2_pretrained.pt`
  and `deeplabv3_resnet50_random.pt`).

EchoNet-Dynamic
  The Stanford project providing the models and pretrained weights both scripts
  build on: <https://echonet.github.io/dynamic/>.
```
