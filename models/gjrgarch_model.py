import autograd.numpy as np
import autograd.numpy.random as npr
import numpy as onp

from .bbvi import BaseBBVIModel

class GJRGARCHModel(BaseBBVIModel):

    def __init__(self, y):
        super().__init__()
        self.y        = y
        self.T        = len(y)
        self.n_params = 4  # omega, alpha, gamma, beta

    # ── parameter transforms (Proposition 2) ──────────────────────────────────
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def parameter_handler(self, z):
        """
        Unconstrained z (4,) → (omega, alpha, gamma, beta)
        Follows Proposition 2 from Magris & Iosifidis.
        """
        # 1. ω = exp(θω)
        omega = np.exp(z[0])
        
        # 2. α = f(θα)
        alpha = self._sigmoid(z[1])
        
        # 3. γ = f(θγ)(2(1 - α) + α) - α
        # This ensures alpha + gamma >= 0 and constraints from paper
        gamma = self._sigmoid(z[2]) * (2.0 * (1.0 - alpha) + alpha) - alpha
        
        # 4. β = f(θβ)(1 - α - 0.5 * γ)
        # This ensures stationarity: alpha + 0.5*gamma + beta < 1
        beta = self._sigmoid(z[3]) * (1.0 - alpha - 0.5 * gamma)
        
        return omega, alpha, gamma, beta

    # ── log jacobian ──────────────────────────────────────────────────────────
    def _log_jacobian(self, z):
        """log |det J| for the Proposition 2 transforms."""
        # Extract components for readability
        s1 = self._sigmoid(z[1])
        s2 = self._sigmoid(z[2])
        s3 = self._sigmoid(z[3])
        
        alpha = s1
        
        # Jacobian terms
        lj  = z[0]                                     # omega: exp
        lj += np.log(s1 * (1 - s1) + 1e-8)             # alpha: sigmoid
        lj += np.log(s2 * (1 - s2) + 1e-8) + np.log(2.0 - alpha + 1e-8) # gamma
        lj += np.log(s3 * (1 - s3) + 1e-8) + np.log(1.0 - alpha - 0.5 * (s2*(2-alpha)-alpha) + 1e-8) # beta
        
        return lj

    # ── log p(z | y) ─────────────────────────────────────────────────────────
    def log_prob(self, z):
        """Vectorized GJR-GARCH Log-Probability"""
        print(".", end="", flush=True)

        n_samples = z.shape[0]
        
        # Transform all 50 samples at once: shape (n_samples, 4)
        # We transpose z so parameter_handler receives (4, n_samples)
        omegas, alphas, gammas, betas = self.parameter_handler(z.T)

        # Pre-compute y^2 and the leverage indicator
        y2 = self.y**2
        leverage = (self.y < 0).astype(float)
        
        # We still need a loop over time (T), but we process 
        # all 50 samples in parallel at each timestep t.
        sigma2_list = []
        
        # Initialize for all samples
        current_sigma2 = np.full(n_samples, np.var(self.y))
        sigma2_list.append(current_sigma2)
        
        for t in range(1, self.T):
            # Vectorized update across the 'n_samples' dimension
            current_sigma2 = (omegas + 
                              alphas * y2[t-1] + 
                              gammas * (y2[t-1] * leverage[t-1]) + 
                              betas * current_sigma2)
            sigma2_list.append(current_sigma2)

        # Stack into (T, n_samples) then transpose to (n_samples, T)
        sigma2 = np.stack(sigma2_list, axis=1) 
        sigma2 = np.maximum(sigma2, 1e-8)

        # Likelihood (vectorized across samples)
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + y2[None, :] / sigma2, axis=1)

        # Prior and Jacobian
        lp = -0.5 * np.sum(z ** 2, axis=1)/100
        lj = self._log_jacobian(z.T)

        return ll + lp + lj

    # ── Variational Family Helpers ────────────────────────────────────────────
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
        if t % 10 == 0:
            mu, sigma = self.unpack_params(params)
            omega, alpha, gamma, beta = self.parameter_handler(mu)
            persistence = alpha + 0.5 * gamma + beta
            print(f"Iter {t:4d} | ω={omega:.6f} α={alpha:.4f} γ={gamma:.4f} β={beta:.4f} "
                  f"| Persist={persistence:.4f}")