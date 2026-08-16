"""Exact GP regression with an ARD Matern-3/2 kernel and measured noise.

Three modelling choices, each of which is a statement about the data rather than
a default:

**Matern-3/2 over RBF.** An RBF kernel assumes the underlying function is
infinitely differentiable. A pressure distribution with a sharp leading-edge
suction peak is not, and imposing that smoothness produces ringing around the
peak. Matern-3/2 assumes once-differentiable sample paths, which is the honest
prior for this class of signal.

**Automatic Relevance Determination.** One lengthscale per feature, learned. The
model is not told which parameters matter; it recovers their relative importance
from the marginal likelihood, and the fitted lengthscales are then readable as a
sensitivity ranking.

**Fixed, per-point noise.** The likelihood consumes the measured standard errors
instead of fitting one global noise level. Measurement quality is not uniform,
and a single scalar makes the posterior overconfident exactly where the data is
worst -- which is usually the region of interest.
"""

from __future__ import annotations

from typing import Optional, Tuple

import gpytorch
import numpy as np
import torch


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood, ard_dims: int):
        super().__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_dims)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class Surrogate:
    """Fit-and-predict wrapper around the exact GP."""

    def __init__(self, device: str = "cpu", seed: Optional[int] = 0):
        self.device = torch.device(device)
        self.seed = seed
        self.model: Optional[ExactGP] = None
        self.likelihood = None
        self.history: list[dict] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise: np.ndarray,
        iters: int = 700,
        lr: float = 0.1,
        log_every: int = 100,
        verbose: bool = True,
    ) -> "Surrogate":
        if self.seed is not None:
            torch.manual_seed(self.seed)

        Xt = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.float64, device=self.device)
        nt = torch.as_tensor(
            np.clip(noise, 1e-10, None), dtype=torch.float64, device=self.device
        )

        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=nt, learn_additional_noise=True
        ).to(dtype=torch.float64, device=self.device)
        self.model = ExactGP(Xt, yt, self.likelihood, ard_dims=X.shape[1]).to(
            dtype=torch.float64, device=self.device
        )

        self.model.train()
        self.likelihood.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.history.clear()
        for step in range(1, iters + 1):
            opt.zero_grad()
            loss = -mll(self.model(Xt), yt)
            loss.backward()
            opt.step()

            if step % log_every == 0 or step == 1:
                ls = self.lengthscales()
                entry = {"iter": step, "loss": float(loss.item()), "lengthscales": ls}
                self.history.append(entry)
                if verbose:
                    pretty = ", ".join(f"{v:.3f}" for v in ls)
                    print(f"  iter {step:4d}/{iters}  loss {loss.item():+.4f}  ARD ls [{pretty}]")

        return self

    def lengthscales(self) -> list[float]:
        ls = self.model.covar_module.base_kernel.lengthscale
        return ls.detach().cpu().numpy().ravel().tolist()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation of the latent function."""
        if self.model is None:
            raise RuntimeError("call fit() before predict()")
        self.model.eval()
        self.likelihood.eval()
        Xt = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = self.model(Xt)
            return (
                post.mean.cpu().numpy(),
                post.variance.clamp_min(0).sqrt().cpu().numpy(),
            )

    # ---- persistence ---------------------------------------------------

    def state(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "likelihood": self.likelihood.state_dict(),
            "history": self.history,
        }


def metrics(
    truth: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    noise_var: Optional[np.ndarray] = None,
) -> dict:
    """Accuracy and calibration.

    RMSE alone says nothing about whether a probabilistic model's uncertainty is
    usable. Coverage -- how often the truth lands inside the stated interval,
    nominally 0.95 at two sigma -- and sharpness are what decide whether the
    error bars can be trusted in a design study.

    Two coverages are reported, and the distinction matters. ``sigma`` from the
    GP is the posterior spread of the *latent function*. A measurement is that
    function plus its own measurement error, so scoring noisy observations
    against the latent band under-covers by construction, however well
    calibrated the model is. Pass ``noise_var`` -- the withheld points' own
    measurement variances -- to get the *predictive* interval,
    ``sqrt(sigma^2 + noise_var)``, which is the one that should hit 0.95.
    Reporting only the first number is a standard way to make a well-calibrated
    surrogate look overconfident, and reporting only the second hides a model
    whose latent uncertainty has genuinely collapsed.
    """
    err = mean - truth
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))

    out = {
        "n": int(truth.size),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "coverage_latent": float(np.mean(np.abs(err) <= 2.0 * sigma)),
        "sharpness_latent": float(np.mean(2.0 * sigma)),
    }

    if noise_var is not None:
        total = np.sqrt(sigma**2 + np.clip(noise_var, 0.0, None))
        out["coverage_predictive"] = float(np.mean(np.abs(err) <= 2.0 * total))
        out["sharpness_predictive"] = float(np.mean(2.0 * total))

    return out
