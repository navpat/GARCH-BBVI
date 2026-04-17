"""
gjrgarch_plots.py
Plotting utilities for GJR-GARCH(1,1) BBVI vs QML comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ── internal helpers ──────────────────────────────────────────────────────────

def _parameter_handler(z):
    """
    Unconstrained z (4,) → (omega, alpha, gamma, beta)
    Follows Proposition 2 from Magris & Iosifidis.
    """
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    
    omega = np.exp(z[0])
    alpha = sigmoid(z[1])
    # gamma transform from paper
    gamma = sigmoid(z[2]) * (2.0 * (1.0 - alpha) + alpha) - alpha
    # beta transform for stationarity
    beta  = sigmoid(z[3]) * (1.0 - alpha - 0.5 * gamma)
    
    return omega, alpha, gamma, beta


# ── public plot functions ─────────────────────────────────────────────────────

def plot_elbo(model, title='GJR-GARCH BBVI — ELBO Convergence', smoothing=20,
              savepath='results/figures/gjr_elbo_convergence.png', dpi=150):
    """Plot BBVI ELBO convergence."""
    fig, ax = plt.subplots(figsize=(9, 5))
    model.plot_elbo(label='BBVI', color='seagreen', ax=ax, smoothing=smoothing)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.show()
    print(f"Saved: {savepath}")


def plot_volatility(sigma2_train_samples, sigma2_test_samples, y_train, y_test,
                    title='GJR-GARCH BBVI — Conditional Volatility',
                    savepath='results/figures/gjr_conditional_volatility.png', dpi=150):
    """Plot posterior mean conditional volatility with 95% CI."""
    sigma2_train_mean = sigma2_train_samples.mean(axis=0)
    sigma2_test_mean  = sigma2_test_samples.mean(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    # Train Plot
    axes[0].plot(np.sqrt(sigma2_train_mean), color='seagreen', lw=0.8, label='BBVI mean vol.')
    axes[0].fill_between(
        range(len(y_train)),
        np.sqrt(np.percentile(sigma2_train_samples, 2.5,  axis=0)),
        np.sqrt(np.percentile(sigma2_train_samples, 97.5, axis=0)),
        alpha=0.3, color='seagreen', label='95% CI'
    )
    axes[0].set_title('Train — Conditional Volatility')
    axes[0].set_ylabel('Volatility (%)')
    axes[0].legend(fontsize=8)

    # Test Plot
    axes[1].plot(np.sqrt(sigma2_test_mean), color='seagreen', lw=0.8, label='BBVI mean vol.')
    axes[1].fill_between(
        range(len(y_test)),
        np.sqrt(np.percentile(sigma2_test_samples, 2.5,  axis=0)),
        np.sqrt(np.percentile(sigma2_test_samples, 97.5, axis=0)),
        alpha=0.3, color='seagreen', label='95% CI'
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
                             savepath='results/figures/gjr_param_distributions.png', dpi=150):
    """
    Plot GJR-GARCH BBVI posterior distributions vs QML point estimates.
    """
    bbvi_draws = vi_mu + vi_sigma * np.random.randn(n_draws, 4)
    
    qml_z = qml['z']
    qml_c = _parameter_handler(qml_z)

    param_names_u = [r'$z_0$ (log $\omega$)', r'$z_1$ (logit $\alpha$)', 
                     r'$z_2$ (logit $\gamma$)', r'$z_3$ (logit $\beta$)']
    param_names_c = [r'$\omega$', r'$\alpha$', r'$\gamma$', r'$\beta$']

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for i in range(4):
        # ── Unconstrained Row ────────────────────────────────────────────────
        ax = axes[0, i]
        ax.hist(bbvi_draws[:, i], bins=60, density=True, alpha=0.5,
                color='seagreen', label='BBVI Posterior')
        
        xs = np.linspace(bbvi_draws[:, i].min(), bbvi_draws[:, i].max(), 200)
        ax.plot(xs, norm.pdf(xs, vi_mu[i], vi_sigma[i]), color='seagreen', lw=2)
        ax.axvline(qml_z[i], color='tomato', linestyle='--', lw=2, label='QML Point')
        
        ax.set_title(param_names_u[i])
        ax.legend(fontsize=8)

        # ── Constrained Row ──────────────────────────────────────────────────
        ax = axes[1, i]
        bbvi_c = np.array([_parameter_handler(bbvi_draws[j])[i] for j in range(n_draws)])
        
        ax.hist(bbvi_c, bins=60, density=True, alpha=0.5, color='seagreen', label='BBVI Posterior')
        ax.axvline(qml_c[i], color='tomato', linestyle='--', lw=2, label='QML Point')
        
        ax.set_title(param_names_c[i])
        ax.legend(fontsize=8)

    axes[0, 0].set_ylabel('Unconstrained', fontsize=10)
    axes[1, 0].set_ylabel('Constrained',   fontsize=10)

    plt.suptitle('GJR-GARCH — BBVI Posterior vs QML Point Estimate', fontsize=13)
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi)
    plt.show()
    print(f"Saved: {savepath}")