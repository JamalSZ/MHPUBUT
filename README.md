# Multi-Horizon Predictability Bounds for Time Series Forecasting

[![Paper](https://img.shields.io/badge/Paper-AAAI%202026-blue)](https://anonymous.4open.science/r/MHPUBUT-81C3)
[![Code](https://img.shields.io/badge/Code-Python-green)](https://anonymous.4open.science/r/MHPUBUT-81C3)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

This repository contains the official implementation of the paper **"A Mode-Agnostic Multi-Horizon Upper Bound for Univariate Time Series Prediction"** (AAAI 2026). Our work introduces a principled framework for quantifying the fundamental limits of predictability in multi-horizon time series forecasting using information-theoretic and chaos-theoretic approaches.



## 🌟 Overview

This work addresses the fundamental challenge of determining the theoretical limits of predictability in multi-horizon time series forecasting. Our framework provides:

1. **Information-theoretic bounds** based on horizon-specific entropy rate estimation
2. **Chaos-theoretic bounds** using Lyapunov exponents to model predictability decay
3. **Noise correction mechanisms** for tighter bound estimation
4. **Model-agnostic benchmarks** applicable to any forecasting approach

The theoretical bounds are validated across diverse real-world datasets and state-of-the-art forecasting models, demonstrating their utility as absolute benchmarks for forecasting performance.

## 🔑 Key Features

- **Multi-horizon predictability bounds**: Quantifies fundamental limits across forecast horizons
- **Dual theoretical approaches**: Combines information theory and deterministic chaos theory
- **Noise-robust estimation**: Includes variance-based correction mechanisms
- **Comprehensive validation**: Tested on ETTh1, EnergyConsumption, and Sunspots datasets
- **Reproducible pipeline**: Complete implementation with Docker support

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster model training)
- Docker (optional, for containerized execution)

### Dataset Access
Datasets are automatically downloaded via the Darts library. For manual access:
from darts.datasets import ETTh1Dataset, EnergyDataset, SunspotsDataset

### Experimental Setup
  - Train-test split: 80:20 with rolling-origin evaluation
  - Forecasting models: DeepAR, NHiTS, PatchTST, Informer, FEDformer
  - Metrics: ε-tolerance predictability (Equation 2 in paper)
  - Hardware: Intel Xeon Silver 4214 (48 cores), 188GB RAM

### Standard Installation
```bash
# Clone the repository
git clone https://anonymous.4open.science/r/MHPUBUT-81C3
cd MHPUBUT-81C3

### Docker
# Build Docker image
docker build -t mhpubut .

# Run container
docker run -it --rm mhpubut ```


