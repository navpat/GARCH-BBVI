# GARCH-BBVI
BBVI implementation for FIGARCH and GJR-GARCH
# GARCH parameter estimation via Black Box Variational Inference

This repository implements a Bayesian framework for estimating GARCH-family models (**FIGARCH** and **GJR-GARCH**) using **Black Box Variational Inference (BBVI)**. It provides a comparative analysis between traditional Quasi-Maximum Likelihood (QML) point estimation and Bayesian posterior approximation on MSFT equity returns.

## 🚀 Key Features
* **Bayesian Uncertainty:** Full posterior distributions for model parameters, allowing for rigorous risk and uncertainty analysis.
* **Stable Parameterization:** Implements unconstrained parameter mapping (Proposition 2, Magris & Iosifidis) via sigmoid and exponential transforms to ensure model stationarity.
* **Dual Architectures:**
    * **FIGARCH(1, d, 1):** Captures hyperbolic decay (long memory) with an $ARCH(\infty)$ truncation ($K=200$).
    * **GJR-GARCH(1,1):** Specifically models leverage effects (asymmetric volatility response).
* **Modern Inference:** Uses the **Reparameterization Trick** and automatic differentiation via `autograd` to optimize the Evidence Lower Bound (ELBO).

## 📁 Project Structure
```text
.
├── gjrgarch_run.py         # Main driver for GJR-GARCH analysis
├── figarch_run.py          # Main driver for FIGARCH analysis
├── requirements.txt        # Project dependencies
├── models/
│   ├── __init__.py         # Package initialization
│   ├── bbvi.py             # Base Class for Variational Inference
│   ├── figarch_model.py    # FIGARCH logic & transformations
│   ├── gjrgarch_model.py   # GJR-GARCH logic & transformations
│   ├── gjrgarch_qml.py     # QML point estimation baseline
│   └── gjrgarch_plots.py   # Visualization suite
└── docs/
    └── BBVI.pdf          # Theoretical documentation (LaTeX)

