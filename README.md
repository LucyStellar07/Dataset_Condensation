# CondTSC: Dataset Condensation for Time Series Classification

**A PyTorch reproduction of "Dataset Condensation for Time Series Classification via Dual Domain Matching" (Liu & Hao, 2026).**

This repository contains the code, data loaders, and visualization scripts to reproduce a Dual-Domain Dataset Condensation framework. It synthesizes a miniature, highly informative dataset (less than 4 percent of the original size) that can train deep learning models to nearly the same accuracy as the massive original dataset.

## Overview

Standard dataset condensation techniques were designed for spatial images and perform poorly on temporal sensor data because they ignore the spectrum information. This project implements **CondTSC**, which solves this by utilizing **Dual Domain Gradient Matching**.
By matching network gradients in both the **Time Domain** and the **Frequency Domain**, the synthetic data retains relevant features of the original dataset.

This project validates the condensation algorithm on two distinct UCI benchmark datasets to evaluate performance across different temporal topologies:

### Human Activity Recognition (HAR)

- **Type:** Dense, multi-channel physical telemetry  
  *(9 channels, length 128)*

- **Classes:** 6  
  *(Walking, Sitting, Standing, Laying, Walking Upstairs, Walking Downstairs)*

- **Purpose:**  
  Evaluates the algorithm's ability to preserve periodic and high-frequency human motion dynamics in condensed synthetic samples.

---

### Synthetic Control Chart Time Series

- **Type:** Univariate time-series with global geometric patterns  
  *(1 channel, length 60)*

- **Classes:** 6  
  *(Normal, Cyclic, Increasing Trend, Decreasing Trend, Upward Shift, Downward Shift)*

- **Purpose:**  
  Evaluates the algorithm's capability to preserve smooth, coordinate-dependent temporal morphology and long-range structural patterns.
## Repository Structure

```text
├── main.py                 # The core bi-level condensation loop and model evaluations
├── condtsc_modules.py      # Contains Dualmodel, KCenter Init, and Augmentations
├── generating_plots.py     # Standalone script to generate figures
├── data_loader.py          # Loading dataset
├── README.md               # Project documentation
├── Report.pdf               # Project report
```
An implementation by Medha Bhavani Girish, EE23B112, as an assignment for Foundations of Machine Learning (DA5400)
