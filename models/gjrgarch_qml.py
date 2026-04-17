"""
gjrgarch_qml.py
Quasi Maximum Likelihood estimator for GJR-GARCH(1,1).
Point estimation using Proposition 2 transforms for stability.
"""

import warnings
import numpy as np
from scipy.optimize import minimize

# ── transforms (Proposition 2) ────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _parameter_handler(z):
    """
    Unconstrained z (4,) → (omega, alpha, gamma, beta)
    Follows Proposition 2 from Magris & Iosifidis.
    """
    omega = np.exp(z[0])
    alpha = _sigmoid(z[1])
    # Ensures alpha + gamma >= 0
    gamma = _sigmoid(z[2]) * (2.0 * (1.0 - alpha) + alpha) - alpha
    # Ensures alpha + 0.5*gamma + beta < 1
    beta  = _sigmoid(z[3]) * (1.0 - alpha - 0.5 * gamma)
    
    return omega, alpha, gamma, beta

# ── GJR-GARCH recursion ───────────────────────────────────────────────────────

def _compute_sigma2(z, y):
    """Compute conditional variance sequence for GJR-GARCH(1,1)."""
    omega, alpha, gamma, beta = _parameter_handler(z)
    T = len(y)
    sigma2 = np.zeros(T)
    
    # Initialization: use unconditional variance of the sample
    sigma2[0] = np.var(y)
    
    y2 = y**2
    for t in range(1, T):
        # Indicator for negative shock (leverage effect)
        leverage = 1.0 if y[t-1] < 0 else 0.0
        
        sigma2[t] = (omega + 
                     alpha * y2[t-1] + 
                     gamma * (y2[t-1] * leverage) + 
                     beta * sigma2[t-1])
        
    return np.maximum(sigma2, 1e-8)

# ── likelihood ────────────────────────────────────────────────────────────────

def _neg_log_likelihood(z, y):
    """Total NLL for GJR-GARCH."""
    sigma2 = _compute_sigma2(z, y)
    y2 = y**2
    # Gaussian Likelihood (standard for QML)
    ll = 0.5 * (np.log(2 * np.pi) + np.log(sigma2) + y2 / sigma2)
    val = np.sum(ll)
    return val if np.isfinite(val) else 1e10

# ── starting points ───────────────────────────────────────────────────────────

def _build_inits(y, n_restarts, rng):
    var_y = np.var(y)
    
    # Deterministic start: simple GARCH(1,1) logic
    z0 = np.zeros(4)
    z0[0] = np.log(var_y * 0.05) 
    z0[1] = -2.0  # alpha small
    z0[2] = 0.0   # gamma neutral
    z0[3] = 1.0   # beta high persistence
    
    inits = [z0]
    for _ in range(n_restarts - 1):
        z = rng.normal(0, 1.0, size=4)
        z[0] = np.log(var_y * rng.uniform(0.01, 0.2))
        inits.append(z)
    return inits

# ── main estimator ────────────────────────────────────────────────────────────

def fit_qml(y, n_restarts=10, seed=42, verbose=True):
    """
    Fit GJR-GARCH(1,1) by Quasi Maximum Likelihood.
    """
    T   = len(y)
    rng = np.random.default_rng(seed)
    inits = _build_inits(y, n_restarts, rng)

    best_result = None
    best_nll    = np.inf
    n_converged = 0

    for i, z0 in enumerate(inits):
        try:
            res = minimize(
                _neg_log_likelihood,
                z0,
                args    = (y,),
                method  = 'L-BFGS-B',
                options = {'maxiter': 5000, 'ftol': 1e-12}
            )
            if res.success:
                n_converged += 1

            if res.fun < best_nll:
                best_nll    = res.fun
                best_result = res

            if verbose:
                omega, alpha, gamma, beta = _parameter_handler(res.x)
                status = '✓' if res.success else '✗'
                print(f"  [{status}] Restart {i+1:2d} | nll={res.fun:.2f} | "
                      f"ω={omega:.4f} α={alpha:.4f} γ={gamma:.4f} β={beta:.4f}")

        except Exception as e:
            if verbose: print(f"  [✗] Restart {i+1} failed: {e}")

    if best_result is None:
        raise RuntimeError("All QML restarts failed.")

    z_opt = best_result.x
    omega, alpha, gamma, beta = _parameter_handler(z_opt)
    sigma2 = _compute_sigma2(z_opt, y)

    return {
        'z': z_opt,
        'omega': omega,
        'alpha': alpha,
        'gamma': gamma,
        'beta': beta,
        'nll': best_nll,
        'sigma2': sigma2,
        'converged': n_converged > 0
    }

# ── metrics ───────────────────────────────────────────────────────────────────

def qml_metrics(sigma2, y):
    y2   = y ** 2
    nll  = 0.5 * np.mean(np.log(2 * np.pi) + np.log(sigma2) + y2 / sigma2)
    rmse = np.sqrt(np.mean((sigma2 - y2) ** 2))
    mad  = np.mean(np.abs(sigma2 - y2))
    qlik = np.mean(np.log(sigma2) + y2 / sigma2)
    return {'NLL': nll, 'RMSE': rmse, 'MAD': mad, 'Qlik': qlik}

def qml_test_sigma2(qml_result, y_test, y_train):
    """Compute test sigma2 by continuing the recursion from train end."""
    y_full = np.concatenate([y_train, y_test])
    sigma2_full = _compute_sigma2(qml_result['z'], y_full)
    return sigma2_full[len(y_train):]