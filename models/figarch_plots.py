"""
figarch_plots.py
Plotting utilities for FIGARCH BBVI vs QML comparison.

Usage in figarch_run.py:
    from figarch_plots import plot_elbo, plot_volatility, plot_param_distributions

    plot_elbo(model)
    plot_volatility(sigma2_train_samples, sigma2_test_samples, y_train, y_test)
    plot_param_distributions(vi_mu, vi_sigma, qml)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ── internal helpers ──────────────────────────────────────────────────────────

def _parameter_handler(z):
    """Unconstrained z (4,) → (omega, d, phi, beta)."""
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    omega = np.exp(z[0])
    d     = sigmoid(z[1])
    phi   = sigmoid(z[2]) * (1 - d) / 2
    beta  = sigmoid(z[3]) * (phi + d)
    return omega, d, phi, beta


def _build_L(y, K):
    """Build lag matrix of squared returns (T, K)."""
    T           = len(y)
    idx         = np.arange(T)[:, None] - 1 - np.arange(K)[None, :]
    valid       = idx >= 0
    idx_clipped = np.where(valid, idx, 0)
    return np.where(valid, y[idx_clipped] ** 2, 0.0)


# ── public plot functions ─────────────────────────────────────────────────────

def plot_elbo(model, title='FIGARCH BBVI — ELBO Convergence', smoothing=20,
              savepath='results/figures/elbo_convergence.png', dpi=150):
    """Plot BBVI ELBO convergence (train and test)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    model.plot_elbo(label='BBVI', color='steelblue', ax=ax, smoothing=smoothing)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.show()
    print(f"Saved: {savepath}")


def plot_volatility(sigma2_train_samples, sigma2_test_samples, y_train, y_test,
                    title='FIGARCH BBVI — Conditional Volatility',
                    savepath='results/figures/conditional_volatility.png', dpi=150):
    """Plot posterior mean conditional volatility with 95% CI for train and test."""
    sigma2_train_mean = sigma2_train_samples.mean(axis=0)
    sigma2_test_mean  = sigma2_test_samples.mean(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    axes[0].plot(np.sqrt(sigma2_train_mean), color='steelblue', lw=0.8, label='BBVI cond. vol.')
    axes[0].fill_between(
        range(len(y_train)),
        np.sqrt(np.percentile(sigma2_train_samples, 2.5,  axis=0)),
        np.sqrt(np.percentile(sigma2_train_samples, 97.5, axis=0)),
        alpha=0.3, color='steelblue', label='95% CI'
    )
    axes[0].set_title('Train — Conditional Volatility')
    axes[0].set_ylabel('Volatility (%)')
    axes[0].legend(fontsize=8)

    axes[1].plot(np.sqrt(sigma2_test_mean), color='steelblue', lw=0.8, label='BBVI cond. vol.')
    axes[1].fill_between(
        range(len(y_test)),
        np.sqrt(np.percentile(sigma2_test_samples, 2.5,  axis=0)),
        np.sqrt(np.percentile(sigma2_test_samples, 97.5, axis=0)),
        alpha=0.3, color='steelblue', label='95% CI'
    )
    axes[1].set_title('Test — Conditional Volatility')
    axes[1].set_ylabel('Volatility (%)')
    axes[1].legend(fontsize=8)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.show()
    print(f"Saved: {savepath}")


def plot_param_distributions(vi_mu, vi_sigma, qml, n_draws=7000,
                             savepath='results/figures/param_distributions.png', dpi=150):
    """
    Plot BBVI posterior histograms with QML point estimates as vertical red lines.
    """
    # ── BBVI draws (mean-field: diagonal) ────────────────────────────────────
    bbvi_draws = vi_mu + vi_sigma * np.random.randn(n_draws, 4)

    # ── QML point estimates (ignore broken covariance/Hessian) ───────────────
    qml_z = qml['z']
    qml_c = _parameter_handler(qml_z)

    param_names_u = ['z₀  (log ω)', 'z₁  (logit d)', 'z₂  (logit φ)', 'z₃  (logit β)']
    param_names_c = ['ω', 'd', 'φ', 'β']

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for i in range(4):
        # ── unconstrained (top row) ───────────────────────────────────────────
        ax = axes[0, i]
        # Plot BBVI Histogram
        ax.hist(bbvi_draws[:, i], bins=60, density=True, alpha=0.5,
                color='steelblue', label='BBVI Posterior')
        
        # Plot BBVI theoretical PDF curve
        xs = np.linspace(bbvi_draws[:, i].min(), bbvi_draws[:, i].max(), 200)
        ax.plot(xs, norm.pdf(xs, vi_mu[i], vi_sigma[i]), color='steelblue', lw=2)
        
        # Plot QML Point Estimate as a red vertical line
        ax.axvline(qml_z[i], color='tomato', linestyle='--', lw=2, label='QML Point')
        
        ax.set_title(param_names_u[i])
        ax.legend(fontsize=8)

        # ── constrained (bottom row) ──────────────────────────────────────────
        ax     = axes[1, i]
        # Transform BBVI draws to constrained space
        bbvi_c = np.array([_parameter_handler(bbvi_draws[j])[i] for j in range(n_draws)])
        
        # Plot BBVI Histogram
        ax.hist(bbvi_c, bins=60, density=True, alpha=0.5, color='steelblue', label='BBVI Posterior')
        
        # Plot QML Point Estimate (transformed) as a red vertical line
        ax.axvline(qml_c[i], color='tomato', linestyle='--', lw=2, label='QML Point')
        
        ax.set_title(param_names_c[i])
        ax.legend(fontsize=8)

    axes[0, 0].set_ylabel('Unconstrained', fontsize=10)
    axes[1, 0].set_ylabel('Constrained',   fontsize=10)

    plt.suptitle('FIGARCH — BBVI Posterior vs QML Point Estimate', fontsize=13)
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.show()
    print(f"Saved: {savepath}")