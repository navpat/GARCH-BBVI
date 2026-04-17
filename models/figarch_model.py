import autograd.numpy as np
import autograd.numpy.random as npr
import numpy as onp

from .bbvi import BaseBBVIModel


class FIGARCHModel(BaseBBVIModel):

    # ── data helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def build_L(y, K):
        T           = len(y)
        idx         = onp.arange(T)[:, None] - 1 - onp.arange(K)[None, :]
        valid       = idx >= 0
        idx_clipped = onp.where(valid, idx, 0)
        return onp.where(valid, y[idx_clipped] ** 2, 0.0)   # (T, K)

    def __init__(self, y, K=200):
        super().__init__()
        self.y        = y
        self.T        = len(y)
        self.K        = K
        self.n_params = 4
        self.L_lag    = self.build_L(y, K)

    # ── parameter transforms ──────────────────────────────────────────────────
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def parameter_handler(self, z):
        """Unconstrained z (4,) → (omega, d, phi, beta)."""
        omega = np.exp(z[0])
        d     = self._sigmoid(z[1])
        phi   = self._sigmoid(z[2]) * (1 - d) / 2
        beta  = self._sigmoid(z[3]) * (phi + d)
        return omega, d, phi, beta

    # ── log jacobian ──────────────────────────────────────────────────────────
    def _log_jacobian(self, z):
        """log |det J| for the transform unconstrained → constrained."""
        omega, d, phi, beta = self.parameter_handler(z)
        s2  = self._sigmoid(z[2])
        s3  = self._sigmoid(z[3])
        lj  = z[0]                                           # omega: exp
        lj += np.log(d + 1e-8) + np.log(1 - d + 1e-8)      # d: sigmoid
        lj += np.log(s2 + 1e-8) + np.log(1 - s2 + 1e-8)    # phi: sigmoid part
        lj += np.log((1 - d) / 2.0 + 1e-8)                  # phi: scale
        lj += np.log(s3 + 1e-8) + np.log(1 - s3 + 1e-8)    # beta: sigmoid part
        lj += np.log(phi + d + 1e-8)                         # beta: scale
        return lj

    # ── log prior — unit normal on unconstrained params (per paper) ───────────
    def _log_prior(self, z):
        """Standard normal prior N(0, I) on unconstrained z."""
        return -0.5 * np.sum(z ** 2)

    # ── log p(z | y) — shared target for BBVI and MCMC ───────────────────────
    def log_prob(self, z):
        """z : (n_samples, 4) unconstrained → (n_samples,)"""
        n_samples = z.shape[0]

        omegas = np.exp(z[:, 0])
        ds     = self._sigmoid(z[:, 1])
        phis   = self._sigmoid(z[:, 2]) * (1 - ds) / 2
        betas  = self._sigmoid(z[:, 3]) * (phis + ds)

        # fractional difference weights (n_samples, K)
        delta = [ds]
        for j in range(1, self.K):
            delta.append((j - 1 - ds) / j * delta[j - 1])

        lam = [ds - betas + phis]
        for j in range(1, self.K):
            lam.append(betas * lam[j - 1] + delta[j] - phis * delta[j - 1])

        LAM = np.stack(lam, axis=1)     
        
        LAM = np.maximum(LAM, 0.0)   # truncate negative weights — standard in FIGARCH                     # (n_samples, K)

        # conditional variance (n_samples, T)
        arch_terms = (self.L_lag @ LAM.T).T
        sigma2     = omegas[:, None] + arch_terms
        T_cur      = sigma2.shape[1]
        sigma2     = np.where(onp.arange(T_cur) == 0, np.var(self.y), sigma2)
        sigma2     = np.maximum(sigma2, 1e-8)

        # print(f"omegas: {omegas}")
        # print(f"arch_terms min: {arch_terms.min()}, max: {arch_terms.max()}")
        # print(f"sigma2 min: {sigma2.min()}, max: {sigma2.max()}")
        # print(f"LAM min: {float(LAM.min())}, max: {float(LAM.max())}")
        # print(f"L_lag min: {self.L_lag.min()}, max: {self.L_lag.max()}")

        # log likelihood (n_samples,)
        y2 = self.y ** 2
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + y2[None, :] / sigma2, axis=1)

        # prior + jacobian per sample
        # prior — N(0,I) on all unconstrained params
        lp_lj = -0.5 * np.sum(z ** 2, axis=1)

        # jacobian — vectorized
        s2  = self._sigmoid(z[:, 2])
        s3  = self._sigmoid(z[:, 3])
        lj  = z[:, 0]
        lj += np.log(ds + 1e-8) + np.log(1 - ds + 1e-8)
        lj += np.log(s2 + 1e-8) + np.log(1 - s2 + 1e-8)
        lj += np.log((1 - ds) / 2.0 + 1e-8)
        lj += np.log(s3 + 1e-8) + np.log(1 - s3 + 1e-8)
        lj += np.log(phis + ds + 1e-8)

        lp_lj = lp_lj + lj
        return ll + lp_lj

    # ── mean-field Gaussian variational family ────────────────────────────────
    #   params layout: [ mu (4) | log_sigma (4) ]  total = 8

    def unpack_params(self, params):
        mu    = params[:self.n_params]
        sigma = np.exp(params[self.n_params:])
        return mu, sigma

    def sample_var_approx(self, params, n_samples=1000):
        mu, sigma = self.unpack_params(params)
        return mu + sigma * npr.randn(n_samples, self.n_params)

    def log_var_approx(self, z, params):
        mu, sigma = self.unpack_params(params)
        return np.sum(
            -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((z - mu) / sigma) ** 2,
            axis=1
        )

    # ── callback ──────────────────────────────────────────────────────────────
    def callback(self, params, t, g):
        if t % 200 == 0:
            mu, sigma = self.unpack_params(params)
            omega, d, phi, beta = self.parameter_handler(mu)
            print(f"Iter {t:4d} | ω={omega:.6f}  d={d:.4f}  φ={phi:.4f}  β={beta:.4f}"
                  f"  |  σ={sigma.round(4)}")