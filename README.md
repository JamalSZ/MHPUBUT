# MHPUBUT
Multi-horizon Predictability Upper Bound in Univariate Time Series
# Multi-Horizon Predictability Upper Bound in Univariate Time Series (MHPUBUT)

This repository contains the official implementation and experimental pipeline for the paper:

**Multi-Horizon Predictability Upper Bound in Univariate Time Series**  
Anonymous submission to AAAI 2025

---

## 📖 Overview

This project investigates the **fundamental limits of predictability** in univariate time series.  
We derive theoretical upper bounds using **entropy-rate estimation** and **Lyapunov exponents**, and validate them with reproducible experiments.

Key contributions of the paper:

- ⚖️ A unified framework connecting **chaotic dynamical systems** and **stochastic processes**.  
- 📉 Derivation of multi-horizon error growth rates using entropy and Lyapunov theory.  
- 🔬 Empirical validation across synthetic and real-world datasets.  
- 🛠️ A reproducible pipeline to compute:
  - Entropy-rate estimates (`h`) via **Lempel-Ziv compression**  
  - Predictability ratio (`π₀ = 1 - h / log |X|`)  
  - Lyapunov spectrum and effective exponents (`λ̂eff`)  
  - Forecast error propagation over increasing horizons  

For complete details, see the paper: [anonymous-submission-latex-2025.pdf](./anonymous-submission-latex-2025.pdf).

---

## 🚀 Getting Started

### Prerequisites
- [Docker](https://www.docker.com/get-started) installed on your system.

### Build the Docker Image
Clone the repository and build the image:

```bash
git clone <repo-url>
cd MHPUBUT
docker build -t mhpubut .

