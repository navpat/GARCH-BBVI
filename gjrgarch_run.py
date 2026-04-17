"""
gjrgarch_run.py
Driver for GJR-GARCH(1,1) BBVI vs QML comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from models.gjrgarch_model import GJRGARCHModel
from models.gjrgarch_qml import fit_qml, qml_metrics, qml_test_sigma2
from models.gjrgarch_plots import plot_elbo, plot_volatility, plot_param_distributions

# ── data ─────────────────────────────────────────────────────────────────────
# Using MSFT as per your previous experiments
ticker = 'MSFT'
msft   = yf.download(ticker, start='2018-01-01', end='2023-06-30')
prices = msft['Close'][ticker].values.flatten()
y      = np.diff(np.log(prices)) * 100

split           = int(len(y) * 0.75)
y_train, y_test = y[:split], y[split:]
print(f"Ticker: {ticker} | Train: {len(y_train)} | Test: {len(y_test)}")

# ── sigma2 helper (Markovian Recursion) ───────────────────────────────────────
def compute_sigma2_sample(z_sample, y):
    """
    Compute GJR-GARCH sigma2 for a single unconstrained sample z (4,).
    Follows Proposition 2 transforms.
    """
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    
    omega = np.exp(z_sample[0])
    alpha = sigmoid(z_sample[1])
    gamma = sigmoid(z_sample[2]) * (2.0 * (1.0 - alpha) + alpha) - alpha
    beta  = sigmoid(z_sample[3]) * (1.0 - alpha - 0.5 * gamma)
    
    T = len(y)
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(y)
    
    y2 = y**2
    for t in range(1, T):
        leverage = 1.0 if y[t-1] < 0 else 0.0
        sigma2[t] = omega + alpha * y2[t-1] + gamma * (y2[t-1] * leverage) + beta * sigma2[t-1]
        
    return np.maximum(sigma2, 1e-8)

# ── performance metrics ───────────────────────────────────────────────────────
def compute_metrics(sigma2_mean, y):
    y2   = y ** 2
    nll  = 0.5 * np.mean(np.log(2 * np.pi) + np.log(sigma2_mean) + y2 / sigma2_mean)
    rmse = np.sqrt(np.mean((sigma2_mean - y2) ** 2))
    mad  = np.mean(np.abs(sigma2_mean - y2))
    qlik = np.mean(np.log(sigma2_mean) + y2 / sigma2_mean)
    return nll, rmse, mad, qlik

# ── BBVI — GJR-GARCH ─────────────────────────────────────────────────────────
print("\nRunning BBVI (GJR-GARCH)...")
model = GJRGARCHModel(y=y_train)

# Hyperparameters per Magris & Iosifidis (2023)
init_params     = np.zeros(8)
init_params[:4] = 0.0                  # mu_0 = 0
init_params[4:] = np.log(np.sqrt(0.1)) # Sigma_0 = 0.1*I (std dev = sqrt(0.1))

vi_params = model.run_VI(
    init_params,
    num_samples = 40,
    step_size   = 0.002,
    num_iters   = 1500,
    how         = 'reparam',
    mass        = 0.5,
    optimizer   = 'sgd_momentum',
)

vi_mu, vi_sigma = model.unpack_params(vi_params)
omega_vi, alpha_vi, gamma_vi, beta_vi = model.parameter_handler(vi_mu)

# ── Plot ELBO ────────────────────────────────────────────────────────────────
plot_elbo(model)

# ── performance metrics (7000 posterior samples) ─────────────────────────────
print("\nComputing performance metrics from 7000 posterior samples...")
N_EVAL   = 7000
vi_draws = vi_mu + vi_sigma * np.random.randn(N_EVAL, 4)

# Multi-sample volatility computation
sigma2_train_samples = np.array([compute_sigma2_sample(vi_draws[i], y_train) for i in range(N_EVAL)])
sigma2_test_samples  = np.array([compute_sigma2_sample(vi_draws[i], y_test) for i in range(N_EVAL)])

# Posterior mean of sigma2
sigma2_train_mean = sigma2_train_samples.mean(axis=0)
sigma2_test_mean  = sigma2_test_samples.mean(axis=0)

nll_train, rmse_train, mad_train, qlik_train = compute_metrics(sigma2_train_mean, y_train)
nll_test,  rmse_test,  mad_test,  qlik_test  = compute_metrics(sigma2_test_mean,  y_test)

bbvi_train = {'NLL': nll_train, 'RMSE': rmse_train, 'MAD': mad_train, 'Qlik': qlik_train}
bbvi_test  = {'NLL': nll_test,  'RMSE': rmse_test,  'MAD': mad_test,  'Qlik': qlik_test}

# ── QML Baseline ─────────────────────────────────────────────────────────────
print("\nRunning QML...")
qml = fit_qml(y_train, n_restarts=5)
qml_train_metrics = qml_metrics(qml['sigma2'], y_train)

sigma2_test_qml = qml_test_sigma2(qml, y_test, y_train)
qml_test_metrics = qml_metrics(sigma2_test_qml, y_test)

# ── Results Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Metric':10} {'BBVI Train':>12} {'BBVI Test':>12} {'QML Train':>12} {'QML Test':>12}")
print("-" * 70)
for m in ['NLL', 'RMSE', 'MAD', 'Qlik']:
    print(f"{m:10} {bbvi_train[m]:>12.4f} {bbvi_test[m]:>12.4f} "
          f"{qml_train_metrics[m]:>12.4f} {qml_test_metrics[m]:>12.4f}")
print("=" * 70)

print(f"\nBBVI posterior mean: ω={omega_vi:.5f} α={alpha_vi:.4f} γ={gamma_vi:.4f} β={beta_vi:.4f}")
print(f"QML point estimate: ω={qml['omega']:.5f} α={qml['alpha']:.4f} γ={qml['gamma']:.4f} β={qml['beta']:.4f}")

# ── Plotting ─────────────────────────────────────────────────────────────────
plot_volatility(sigma2_train_samples, sigma2_test_samples, y_train, y_test)
plot_param_distributions(vi_mu, vi_sigma, qml)