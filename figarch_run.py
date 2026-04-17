"""
figarch_run.py
1. BBVI initialized per paper: mu_0=0, Sigma_0=0.1*I, lr=0.005, momentum=0.4, 50 MC draws
2. Plot ELBO convergence
3. Compute performance metrics: NLL, RMSE, MAD, Qlik (train and test)
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from models.figarch_model import FIGARCHModel
from models.figarch_qml import fit_qml, qml_metrics, qml_test_sigma2
from models.figarch_plots import plot_elbo, plot_volatility, plot_param_distributions

# ── data ─────────────────────────────────────────────────────────────────────
msft   = yf.download('MSFT', start='2018-01-01', end='2023-06-30')
prices = msft['Close']['MSFT'].values.flatten()
y      = np.diff(np.log(prices)) * 100

split           = int(len(y) * 0.75)
y_train, y_test = y[:split], y[split:]
print(f"Train: {len(y_train)} | Test: {len(y_test)}")
print(f"var(y_train)={np.var(y_train):.4f} | var(y_test)={np.var(y_test):.4f}")


# ── sigma2 helper ─────────────────────────────────────────────────────────────
def compute_sigma2(model, z_sample, y, L_lag):
    """Compute sigma2 for a single unconstrained sample z (4,)."""
    z      = z_sample[None, :]
    ds     = 1 / (1 + np.exp(-z[:, 1]))
    phis   = 1 / (1 + np.exp(-z[:, 2])) * (1 - ds) / 2
    betas  = 1 / (1 + np.exp(-z[:, 3])) * (phis + ds)
    omegas = np.exp(z[:, 0])

    delta = [ds]
    for j in range(1, model.K):
        delta.append((j - 1 - ds) / j * delta[j - 1])
    lam = [ds - betas + phis]
    for j in range(1, model.K):
        lam.append(betas * lam[j - 1] + delta[j] - phis * delta[j - 1])

    LAM        = np.stack(lam, axis=1)
    LAM        = np.maximum(LAM, 0.0)
    arch_terms = (L_lag @ LAM.T).T
    sigma2     = omegas[:, None] + arch_terms
    sigma2     = np.where(np.arange(sigma2.shape[1]) == 0, np.var(y), sigma2)
    sigma2     = np.maximum(sigma2, 1e-8)
    return sigma2[0]   # (T,)


# ── performance metrics ───────────────────────────────────────────────────────
def compute_metrics(sigma2_mean, y):
    """
    sigma2_mean : (T,) posterior mean of conditional variance
    y           : (T,) returns
    Returns NLL, RMSE, MAD, Qlik  (all positive, lower = better)
    """
    y2   = y ** 2
    nll  = 0.5 * np.mean(np.log(2 * np.pi) + np.log(sigma2_mean) + y2 / sigma2_mean)
    rmse = np.sqrt(np.mean((sigma2_mean - y2) ** 2))
    mad  = np.mean(np.abs(sigma2_mean - y2))
    qlik = np.mean(np.log(sigma2_mean) + y2 / sigma2_mean)
    return nll, rmse, mad, qlik


# ── BBVI — paper hyperparameters ─────────────────────────────────────────────
print("\nRunning BBVI...")
model        = FIGARCHModel(y=y_train)
model.y_test = y_test
model.L_test = FIGARCHModel.build_L(y_test, model.K)

init_params     = np.zeros(8)
init_params[:4] = 0.0                   # mu_0 = 0
init_params[4:] = np.log(np.sqrt(0.1)) # Sigma_0 = 0.1*I

vi_params = model.run_VI(
    init_params,
    num_samples = 50,
    step_size   = 0.005,
    num_iters   = 2500,
    how         = 'reparam',
    mass        = 0.4,
    optimizer   = 'sgd_momentum',
)

vi_mu, vi_sigma = model.unpack_params(vi_params)
omega_vi, d_vi, phi_vi, beta_vi = model.parameter_handler(vi_mu)
print(f"\nBBVI posterior mean (constrained):")
print(f"  ω={omega_vi:.6f}  d={d_vi:.4f}  φ={phi_vi:.4f}  β={beta_vi:.4f}")

# ── plot: ELBO convergence ────────────────────────────────────────────────────
plot_elbo(model)

# ── performance metrics from 7000 posterior samples (as per paper) ────────────
print("\nComputing performance metrics from 7000 posterior samples...")
N_EVAL   = 7000
vi_draws = vi_mu + vi_sigma * np.random.randn(N_EVAL, 4)

L_train = FIGARCHModel.build_L(y_train, model.K)
y_full  = np.concatenate([y_train, y_test])
L_full  = FIGARCHModel.build_L(y_full, model.K)
L_test  = L_full[len(y_train):]

sigma2_train_samples = np.array([
    compute_sigma2(model, vi_draws[i], y_train, L_train)
    for i in range(N_EVAL)
])   # (7000, T_train)

sigma2_test_samples = np.array([
    compute_sigma2(model, vi_draws[i], y_test, L_test)
    for i in range(N_EVAL)
])

# posterior mean of sigma2 — eq. 6 in paper (transformed posterior mean)
sigma2_train_mean = sigma2_train_samples.mean(axis=0)
sigma2_test_mean  = sigma2_test_samples.mean(axis=0)

nll_train,  rmse_train,  mad_train,  qlik_train  = compute_metrics(sigma2_train_mean, y_train)
nll_test,   rmse_test,   mad_test,   qlik_test   = compute_metrics(sigma2_test_mean,  y_test)

bbvi_train = {'NLL': nll_train, 'RMSE': rmse_train, 'MAD': mad_train, 'Qlik': qlik_train}
bbvi_test  = {'NLL': nll_test,  'RMSE': rmse_test,  'MAD': mad_test,  'Qlik': qlik_test}

# ── summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print(f"{'Metric':10} {'Train':>12} {'Test':>12}")
print("-" * 58)
print(f"{'NLL':10} {nll_train:>12.4f} {nll_test:>12.4f}")
print(f"{'RMSE':10} {rmse_train:>12.4f} {rmse_test:>12.4f}")
print(f"{'MAD':10} {mad_train:>12.4f} {mad_test:>12.4f}")
print(f"{'Qlik':10} {qlik_train:>12.4f} {qlik_test:>12.4f}")
print("=" * 58)

print(f"\nBBVI parameter estimates (transformed posterior mean):")
print(f"  ω={omega_vi:.5f}  d={d_vi:.4f}  φ={phi_vi:.4f}  β={beta_vi:.4f}")

# ── plot: conditional volatility ──────────────────────────────────────────────
plot_volatility(sigma2_train_samples, sigma2_test_samples, y_train, y_test)

#QML baseline
print("\nRunning QML...")
qml = fit_qml(y_train, K=200, n_restarts=5)

# train metrics
qml_train = qml_metrics(qml['sigma2'], y_train)

# test metrics — properly conditioned on train lags
sigma2_test_qml = qml_test_sigma2(qml, y_test, y_train, K=200)
qml_test = qml_metrics(sigma2_test_qml, y_test)

# compare
print(f"\n{'Metric':10} {'BBVI Train':>12} {'BBVI Test':>12} {'QML Train':>12} {'QML Test':>12}")
print("-" * 70)
for m in ['NLL', 'RMSE', 'MAD', 'Qlik']:
    print(f"{m:10} {bbvi_train[m]:>12.4f} {bbvi_test[m]:>12.4f} "
          f"{qml_train[m]:>12.4f} {qml_test[m]:>12.4f}")
    

#Plot distributions 
plot_param_distributions(vi_mu, vi_sigma, qml)

