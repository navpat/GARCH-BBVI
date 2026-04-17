import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

import datetime as dt

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad
from autograd.misc.optimizers import adam
from autograd.core import getval


# ── SGD with momentum ─────────────────────────────────────────────────────────
def sgd_momentum(grad_fn, x, callback=None, num_iters=2000, step_size=0.005, mass=0.4):
    velocity = np.zeros_like(x)
    for i in range(num_iters):
        g        = grad_fn(x, i)
        velocity = mass * velocity - (1.0 - mass) * g
        x        = x + step_size * velocity
        if callback:
            callback(x, i, g)
    return x


# ======Model====
class BaseBBVIModel(metaclass=ABCMeta):
    """
    An abstract base class providing the structure for a general Bayesian
    inference problem to be solved using black box variational inference.
    """
    def __init__(self):
        self._init_var_params = None
        self._var_params      = None
        self.N_SAMPLES        = None
        self.elbo_history     = []
        self.y_test           = None

    # -------User-specified methods (abstract) -------
    @abstractmethod
    def unpack_params(self, params):
        pass

    @abstractmethod
    def log_var_approx(self, z, params):
        pass

    @abstractmethod
    def sample_var_approx(self, params, n_samples=1000):
        pass

    @abstractmethod
    def log_prob(self, z):
        pass

    def callback(self, params, t, g):
        pass

    # -------ELBO estimators-------
    def _objfunc(self, params, t):
        samps = self.sample_var_approx(getval(params), n_samples=self.N_SAMPLES)
        return -np.mean(
            self.log_var_approx(samps, params)
            * (self.log_prob(samps) - self.log_var_approx(samps, getval(params)))
        )

    def _objfuncCV(self, params, t):
        samps = self.sample_var_approx(getval(params), n_samples=self.N_SAMPLES)
        a_hat = np.mean(self.log_prob(samps) - self.log_var_approx(samps, getval(params)))
        return -np.mean(
            self.log_var_approx(samps, params)
            * (self.log_prob(samps) - self.log_var_approx(samps, getval(params)) - a_hat)
        )

    def _estimate_ELBO(self, params, t):
        samps = self.sample_var_approx(params, n_samples=self.N_SAMPLES)
        return -np.mean(self.log_prob(samps) - self.log_var_approx(samps, params), axis=0)

    def _estimate_ELBO_noscore(self, params, t):
        samps = self.sample_var_approx(params, n_samples=self.N_SAMPLES)
        return -np.mean(self.log_prob(samps) - self.log_var_approx(samps, getval(params)), axis=0)

    # -------ELBO evaluation (no grad needed) -------
    def _eval_elbo(self, params, n_samples=50):
        samps = self.sample_var_approx(getval(params), n_samples=n_samples)
        return float(np.mean(self.log_prob(samps) - self.log_var_approx(samps, getval(params))))

    # -------Optimization-------
    def run_VI(self, init_params, num_samples=50, step_size=0.005, num_iters=2000,
               how='reparam', mass=0.4, optimizer='sgd_momentum'):
        hows = ['stochsearch', 'reparam', 'noscore']
        if how not in hows:
            raise KeyError('Allowable VI methods are', hows)

        self.N_SAMPLES    = num_samples
        self.elbo_history = []

        if how == 'stochsearch':
            _tmp_gradient = grad(self._objfunc)
        elif how == 'reparam':
            _tmp_gradient = grad(self._estimate_ELBO)
        elif how == 'noscore':
            _tmp_gradient = grad(self._estimate_ELBO_noscore)

        self._init_var_params = init_params

        def _tracking_callback(params, t, g):
            train_elbo = self._eval_elbo(params, n_samples=50)

            if self.y_test is not None:
                y_saved, L_saved = self.y, self.L_lag
                self.y, self.L_lag = self.y_test, self.L_test
                test_elbo = self._eval_elbo(params, n_samples=50)
                self.y, self.L_lag = y_saved, L_saved
            else:
                test_elbo = float('nan')

            self.elbo_history.append((t, train_elbo, test_elbo))
            self.callback(params, t, g)

        s = dt.datetime.now()

        if optimizer == 'sgd_momentum':
            self._var_params = sgd_momentum(
                _tmp_gradient,
                self._init_var_params,
                step_size = step_size,
                mass      = mass,
                num_iters = num_iters,
                callback  = _tracking_callback,
            )
        elif optimizer == 'adam':
            self._var_params = adam(
                _tmp_gradient,
                self._init_var_params,
                step_size = step_size,
                num_iters = num_iters,
                callback  = _tracking_callback,
            )
        else:
            raise KeyError('Allowable optimizers are: sgd_momentum, adam')

        print('done in:', dt.datetime.now() - s)
        return self._var_params

    # -------Plotting-------
    def plot_elbo(self, label='BBVI', color='steelblue', ax=None, smoothing=10):
        if not self.elbo_history:
            raise RuntimeError("No ELBO history found — run run_VI first.")

        history    = np.array(self.elbo_history)
        iters      = history[:, 0]
        train_elbo = history[:, 1]
        test_elbo  = history[:, 2]

        def smooth(x, w):
            if w <= 1:
                return x
            return np.convolve(x, np.ones(w) / w, mode='valid')

        iters_s = iters[:len(smooth(train_elbo, smoothing))]

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(iters_s, smooth(train_elbo, smoothing),
                label=f'{label} Train', color=color, linewidth=2)
        ax.plot(iters_s, smooth(test_elbo, smoothing),
                label=f'{label} Test', color=color, linewidth=2, linestyle='--')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        ax.set_title('ELBO Convergence')
        ax.legend()

        return ax


if __name__ == '__main__':
    pass