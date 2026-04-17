"""
figarch_qml.py
Quasi Maximum Likelihood estimator for FIGARCH(1,d,1).

Standard errors use the Bollerslev-Wooldridge (1992) sandwich estimator:
    Cov(z) = H⁻¹ B H⁻¹
where H is the numerical Hessian of the total NLL and B = Σ sₜsₜᵀ is the
outer product of per-observation score vectors. This is robust to
non-normality of returns — the standard in the GARCH literature.
"""

import warnings
import numpy as np
from scipy.optimize import minimize


# ── transforms ────────────────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _parameter_handler(z):
    """Unconstrained z (4,) → (omega, d, phi, beta)."""
    omega = np.exp(z[0])
    d     = _sigmoid(z[1])
    phi   = _sigmoid(z[2]) * (1 - d) / 2
    beta  = _sigmoid(z[3]) * (phi + d)
    return omega, d, phi, beta


# ── lag matrix ────────────────────────────────────────────────────────────────

def _build_L(y, K):
    """Build lag matrix of squared returns (T, K)."""
    T           = len(y)
    idx         = np.arange(T)[:, None] - 1 - np.arange(K)[None, :]
    valid       = idx >= 0
    idx_clipped = np.where(valid, idx, 0)
    return np.where(valid, y[idx_clipped] ** 2, 0.0)


# ── FIGARCH recursion ─────────────────────────────────────────────────────────

def _compute_lam(d, phi, beta, K):
    """
    Compute ARCH(inf) weights λ for FIGARCH(1,d,1).
    Returns (K,) array; negative weights truncated to 0 (standard in FIGARCH).
    """
    delta = [d]
    for j in range(1, K):
        delta.append((j - 1 - d) / j * delta[j - 1])

    lam = [d - beta + phi]
    for j in range(1, K):
        lam.append(beta * lam[j - 1] + delta[j] - phi * delta[j - 1])

    return np.maximum(np.array(lam), 0.0)


def _compute_sigma2(z, L_lag, y, K):
    """Compute conditional variance sequence for unconstrained z (4,)."""
    omega, d, phi, beta = _parameter_handler(z)
    LAM       = _compute_lam(d, phi, beta, K)
    sigma2    = omega + L_lag @ LAM
    sigma2[0] = np.var(y)               # backcast first observation
    return np.maximum(sigma2, 1e-8)


# ── likelihood ────────────────────────────────────────────────────────────────

def _obs_nll(z, L_lag, y, K):
    """
    Per-observation negative log-likelihood contributions (T,).
    Used for score computation in the sandwich estimator.
    """
    sigma2 = _compute_sigma2(z, L_lag, y, K)
    y2     = y ** 2
    return 0.5 * (np.log(2 * np.pi) + np.log(sigma2) + y2 / sigma2)


def _neg_log_likelihood(z, L_lag, y, K):
    """Total NLL (sum). Returns 1e10 if non-finite."""
    val = _obs_nll(z, L_lag, y, K).sum()
    return val if np.isfinite(val) else 1e10


# ── sandwich standard errors ──────────────────────────────────────────────────

def _sandwich_se(z, L_lag, y, K, eps=1e-5):
    """
    Bollerslev-Wooldridge (1992) sandwich covariance: H⁻¹ B H⁻¹.

    H : numerical Hessian of total NLL          (4, 4)
    B : outer product of per-obs scores Σ sₜsₜᵀ (4, 4)

    Per-obs scores computed by central differences on _obs_nll,
    so each sₜ = ∂ℓₜ/∂z evaluated at the optimum.

    Returns
    -------
    se  : (4,) standard errors in unconstrained space, or None
    cov : (4, 4) full sandwich covariance matrix, or None
    """
    T = len(y)
    n = len(z)

    # ── Hessian of total NLL (central differences) ───────────────────────────
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej  = np.zeros(n), np.zeros(n)
            ei[i]   = eps
            ej[j]   = eps
            H[i, j] = (
                _neg_log_likelihood(z + ei + ej, L_lag, y, K)
                - _neg_log_likelihood(z + ei - ej, L_lag, y, K)
                - _neg_log_likelihood(z - ei + ej, L_lag, y, K)
                + _neg_log_likelihood(z - ei - ej, L_lag, y, K)
            ) / (4 * eps ** 2)

    # ── per-observation scores (T, 4) via central differences ────────────────
    scores = np.zeros((T, n))
    for k in range(n):
        ek           = np.zeros(n)
        ek[k]        = eps
        scores[:, k] = (
            _obs_nll(z + ek, L_lag, y, K)
            - _obs_nll(z - ek, L_lag, y, K)
        ) / (2 * eps)

    # ── B = Σ sₜ sₜᵀ (outer product of gradients) ───────────────────────────
    B = scores.T @ scores           # (4, 4)

    # ── sandwich ─────────────────────────────────────────────────────────────
    try:
        Hi   = np.linalg.inv(H)
        cov  = Hi @ B @ Hi
        diag = np.diag(cov)
        if np.any(diag < 0):
            warnings.warn(
                "Sandwich covariance has negative diagonal — SE unreliable. "
                "Hessian may not be PD at optimum.",
                RuntimeWarning
            )
            return None, None
        return np.sqrt(diag), cov
    except np.linalg.LinAlgError:
        warnings.warn("Hessian singular — sandwich SE unavailable.", RuntimeWarning)
        return None, None


# ── starting points ───────────────────────────────────────────────────────────

def _build_inits(y, n_restarts, rng):
    """
    Generate starting points in unconstrained space.
    First point is deterministic; remaining cover a wide range of omega values.
    """
    var_y = np.var(y)

    z0    = np.zeros(4)
    z0[0] = np.log(var_y * 0.05)   # omega: small fraction of variance
    z0[1] = 0.0                     # d     = sigmoid(0) = 0.5
    z0[2] = -3.0                    # phi   ≈ 0
    z0[3] = -1.0                    # beta  ≈ 0.27 * (phi + d)

    inits = [z0]
    for _ in range(n_restarts - 1):
        z    = rng.normal(0, 0.5, size=4)
        z[0] = np.log(var_y * rng.uniform(0.01, 0.5))
        inits.append(z)

    return inits

def _prior_penalty(z, prior_std=1.0):
    """
    Gaussian log-prior on unconstrained params (negative, since we minimise NLL).
    Equivalent to N(0, prior_std²) on each z.
    """
    return 0.5 * np.sum(z**2) / prior_std**2

def _penalized_nll(z, L_lag, y, K, prior_std=1.0):
    return _neg_log_likelihood(z, L_lag, y, K) + _prior_penalty(z, prior_std)
# ── main estimator ────────────────────────────────────────────────────────────

def fit_qml(y, K=200, n_restarts=10, seed=42, verbose=True):
    """
    Fit FIGARCH(1,d,1) by Quasi Maximum Likelihood.

    Standard errors use the Bollerslev-Wooldridge (1992) sandwich estimator,
    robust to non-normality of returns.

    Parameters
    ----------
    y          : (T,) array of returns
    K          : truncation lag for ARCH(inf) representation
    n_restarts : number of random restarts (first is deterministic)
    seed       : RNG seed for restarts
    verbose    : print progress

    Returns
    -------
    dict with keys:
        z               : optimal unconstrained parameters (4,)
        omega, d, phi, beta : constrained parameters
        se              : (4,) sandwich SE in unconstrained space, or None
        cov             : (4, 4) sandwich covariance matrix, or None
        t_stats         : (4,) t-statistics z / se, or None
        nll             : negative log-likelihood at optimum
        nll_per_obs     : nll / T
        sigma2          : (T,) fitted conditional variance
        converged       : bool
        n_converged     : int
    """
    T     = len(y)
    L_lag = _build_L(y, K)
    rng   = np.random.default_rng(seed)
    inits = _build_inits(y, n_restarts, rng)

    best_result = None
    best_nll    = np.inf
    n_converged = 0

    for i, z0 in enumerate(inits):
        try:
            res = minimize(
                _penalized_nll,
                z0,
                args    = (L_lag, y, K),
                method  = 'L-BFGS-B',
                options = {'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-9},
            )
            if res.success:
                n_converged += 1

            if res.fun < best_nll:
                best_nll    = res.fun
                best_result = res

            if verbose:
                omega, d, phi, beta = _parameter_handler(res.x)
                status = '✓' if res.success else '✗'
                print(f"  [{status}] Restart {i+1:2d}/{n_restarts} | "
                      f"nll={res.fun:.2f} | "
                      f"ω={omega:.4f}  d={d:.4f}  φ={phi:.4f}  β={beta:.4f}")

        except Exception as e:
            if verbose:
                print(f"  [✗] Restart {i+1} failed: {e}")

    if best_result is None:
        raise RuntimeError("All QML restarts failed.")

    if n_converged == 0:
        warnings.warn(
            "No QML restart converged — results may be unreliable.",
            RuntimeWarning
        )

    z_opt               = best_result.x
    omega, d, phi, beta = _parameter_handler(z_opt)
    sigma2              = _compute_sigma2(z_opt, L_lag, y, K)

    # ── sandwich SE ──────────────────────────────────────────────────────────
    if verbose:
        print("\nComputing sandwich standard errors...")
    se, cov = _sandwich_se(z_opt, L_lag, y, K)
    t_stats = z_opt / se if se is not None else None

    if verbose:
        print(f"\nQML estimates ({n_converged}/{n_restarts} restarts converged):")
        print(f"  ω={omega:.5f}  d={d:.4f}  φ={phi:.4f}  β={beta:.4f}")
        print(f"  NLL={best_nll:.4f}  |  NLL/T={best_nll/T:.4f}")
        if se is not None:
            labels = ['z₀ (log ω)', 'z₁ (logit d)', 'z₂ (logit φ)', 'z₃ (logit β)']
            print(f"\n  {'Param':>14} {'Estimate':>10} {'SE':>10} {'t-stat':>10}")
            print(f"  {'-'*46}")
            for lbl, zi, sei, ti in zip(labels, z_opt, se, t_stats):
                print(f"  {lbl:>14} {zi:>10.4f} {sei:>10.4f} {ti:>10.4f}")

    return {
        'z'          : z_opt,
        'omega'      : omega,
        'd'          : d,
        'phi'        : phi,
        'beta'       : beta,
        'se'         : se,
        'cov'        : cov,
        't_stats'    : t_stats,
        'nll'        : best_nll,
        'nll_per_obs': best_nll / T,
        'sigma2'     : sigma2,
        'converged'  : n_converged > 0,
        'n_converged': n_converged,
    }


# ── metrics ───────────────────────────────────────────────────────────────────

def qml_metrics(sigma2, y):
    """
    Forecast evaluation metrics from fitted sigma2.
    Returns dict: NLL, RMSE, MAD, Qlik  (all lower = better).
    """
    y2   = y ** 2
    nll  = 0.5 * np.mean(np.log(2 * np.pi) + np.log(sigma2) + y2 / sigma2)
    rmse = np.sqrt(np.mean((sigma2 - y2) ** 2))
    mad  = np.mean(np.abs(sigma2 - y2))
    qlik = np.mean(np.log(sigma2) + y2 / sigma2)
    return {'NLL': nll, 'RMSE': rmse, 'MAD': mad, 'Qlik': qlik}


# ── out-of-sample sigma2 ──────────────────────────────────────────────────────

def qml_test_sigma2(qml_result, y_test, y_train, K=200):
    """
    Compute test-set sigma2 using parameters estimated on train.
    Concatenates train+test so test rows are conditioned on up to K
    lags of train squared returns (avoids cold-start bias).
    """
    y_full = np.concatenate([y_train, y_test])
    L_full = _build_L(y_full, K)
    L_test = L_full[len(y_train):]

    omega, d, phi, beta = _parameter_handler(qml_result['z'])
    LAM    = _compute_lam(d, phi, beta, K)
    sigma2 = omega + L_test @ LAM
    return np.maximum(sigma2, 1e-8)