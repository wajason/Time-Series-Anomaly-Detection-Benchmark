# Time Series Anomaly Detection Benchmark: Series2Graph vs. Deep Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wajason/Time-Series-Anomaly-Detection-Benchmark/blob/main/Implement_the_Time_series_Anomaly_detection.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

## 📖 Overview

This repository provides a comparative implementation of graph-based and deep learning-based methods for **unsupervised time series anomaly detection**. The project focuses on evaluating the **Series2Graph** algorithm against reconstruction-based deep learning baselines (**AutoEncoder** and **LSTM-AD**) using the **TSB-UAD (Time Series Benchmark for Unsupervised Anomaly Detection)** dataset.

The implementation and analysis are inspired by the insights presented in:
> **Advances in Time-Series Anomaly Detection**
> *John Paparrizos, Paul Boniol, Qinghua Liu, Themis Palpanas*
> KDD '25 (ACM SIGKDD Conference on Knowledge Discovery and Data Mining)

## 🚀 Implemented Methods

We compare three distinct approaches to anomaly detection:

1.  **Series2Graph (S2G)**:
    * **Type**: Graph-based / Unsupervised
    * **Mechanism**: Converts time series subsequences into a graph representation (nodes and edges). Anomalies are identified as subsequences with low transition probabilities (rare paths) in the graph.
    * **Key Advantage**: Parameter-efficient and robust to noise.

2.  **AE-MLP (AutoEncoder)**:
    * **Type**: Reconstruction-based / Deep Learning
    * **Mechanism**: Compresses input windows into a latent space using an MLP encoder and attempts to reconstruct them. The **Reconstruction Error** (MSE) is used as the anomaly score.

3.  **LSTM-AD (LSTM AutoEncoder)**:
    * **Type**: Forecasting/Reconstruction-based / Deep Learning
    * **Mechanism**: Utilizes LSTM layers to capture temporal dependencies. Similar to AE, it uses reconstruction error as the anomaly indicator but is designed to handle sequential correlations better.

## 📊 Experimental Results

Experiments were conducted on the **TSB-UAD** public dataset. The performance was evaluated using **AUC-ROC** and **AUC-PR**.

| Rank | Model | Type | AUC-ROC | AUC-PR |
| :--- | :--- | :--- | :--- | :--- |
| 🥇 | **Series2Graph** | Graph-based | **0.6605** | **0.0831** |
| 🥈 | AE-MLP | Reconstruction | 0.5288 | 0.0609 |
| 🥉 | LSTM-AD | Forecasting | 0.3870 | 0.0407 |

### Analysis
Our results align with the observations in Paparrizos et al. (2025):
* **Series2Graph outperforms deep learning baselines** in this unsupervised setting, demonstrating superior robustness without extensive hyperparameter tuning.
* **AE-MLP** performance is marginal (near random guess), suggesting that simple reconstruction error is insufficient for this dataset's anomaly types.
* **LSTM-AD** underperforms significantly (AUC < 0.5), potentially due to overfitting on the training data where it learned to reconstruct anomalies as well as normal patterns.

## 📂 Dataset

The project automatically downloads and extracts the **TSB-UAD Public** dataset.
* **Source**: [TheDatum.org / TSB-UAD](https://www.thedatum.org/datasets/TSB-UAD-Public.zip)
* **Structure**: Univariate time series with corresponding binary anomaly labels (0: normal, 1: anomaly).
