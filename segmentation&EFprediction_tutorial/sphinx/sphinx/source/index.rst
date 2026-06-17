.. EchoNet Cardiac Analysis Tutorial documentation master file

#################################
EchoNet Cardiac Analysis Tutorial
#################################

.. epigraph::

   *Left-ventricle segmentation and ejection-fraction prediction from
   echocardiograms — from setup to results.*

Welcome to this tutorial on automated **echocardiography analysis**. It walks you
through two ready-to-run scripts built on top of the
`EchoNet-Dynamic <https://github.com/echonet/dynamic>`_ deep-learning models:

- **PredictSegmentation.py** — segments the left ventricle (LV) in ultrasound
  videos and ``.npy`` images, measures LV area and detects systolic frames.
- **PredictEF.py** — predicts the **ejection fraction (EF)** of the heart directly
  from an apical-4-chamber ultrasound video.
- **CalculateStats.py** — (optional) computes the per-channel ``MEAN``/``STD``
  used to normalise inputs, so you can retune the models for your own dataset.

.. note::
   Both scripts run on Windows or Linux, with or without an NVIDIA GPU.
   A GPU is strongly recommended for video processing but not required.

----

.. toctree::
   :maxdepth: 2
   :caption: 📖 Tutorial Contents
   :numbered:

   chapters/01_introduction
   chapters/02_installation
   chapters/03_data_preparation
   chapters/04_normalization_stats
   chapters/05_segmentation
   chapters/06_ejection_fraction
   chapters/07_results_troubleshooting
   chapters/08_downloads

----

.. toctree::
   :maxdepth: 1
   :caption: 📚 Appendices

   appendix/glossary
   appendix/faq

----

.. rubric:: Tutorial Overview

.. list-table::
   :widths: 5 35 60
   :header-rows: 1

   * - #
     - Chapter
     - What you will learn
   * - 1
     - Introduction
     - The two tools, EchoNet-Dynamic background, the workflow
   * - 2
     - Installation
     - Conda environment, dependencies, model weights
   * - 3
     - Data Preparation
     - Accepted input formats (videos and ``.npy``), folder layout
   * - 4
     - Normalization Statistics
     - Computing ``MEAN``/``STD`` for your own data with ``CalculateStats.py``
   * - 5
     - LV Segmentation
     - Running ``PredictSegmentation.py``, reading its outputs
   * - 6
     - EF Prediction
     - Running ``PredictEF.py``, interpreting EF values
   * - 7
     - Results & Troubleshooting
     - Output files, common errors and how to fix them
   * - 8
     - Downloads
     - The three scripts and two pretrained weights, ready to download

----

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
