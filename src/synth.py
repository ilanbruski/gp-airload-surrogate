"""Synthetic airload dataset generator.

Writes an HDF5 file in the layout the loader expects, so the whole pipeline can
be exercised without any measured data. The field is analytic: a thin-airfoil
flavoured chordwise pressure distribution whose peak strength and location are
modulated by the design parameters, plus a heteroskedastic noise model so the
per-point standard errors are meaningful rather than decorative.

Nothing here is a physical model of anything. It exists to produce a signal with
the awkward properties the real problem has -- a sharp leading-edge peak, smooth
parameter dependence, and measurement error that varies across the surface.
"""

from __future__ import annotations

import itertools
from typing import Dict, Sequence

import h5py
import numpy as np

from config import DoEConfig


def chordwise_stations(n_upper: int, n_lower: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine-clustered tap positions, dense at the leading edge."""
    beta_u = np.linspace(0.0, np.pi, n_upper)
    beta_l = np.linspace(0.0, np.pi, n_lower)
    return 0.5 * (1.0 - np.cos(beta_u)), 0.5 * (1.0 - np.cos(beta_l))


def _field(xc: np.ndarray, params: Dict[str, float], surface: str) -> np.ndarray:
    """Analytic pressure-coefficient-like distribution.

    Upper surface carries a suction peak near the leading edge whose depth grows
    with incidence; the lower surface is a milder recovery. Both drift smoothly
    with the remaining parameters so a surrogate has something learnable.
    """
    alpha = params.get("alpha", 0.0)
    station = params.get("station", 0.0)
    speed = params.get("speed", 30.0)

    # peak strength: grows with incidence, decays outboard, mild speed effect
    strength = (0.6 + 0.11 * alpha) * (1.0 - 0.25 * station) * (speed / 30.0) ** 0.2
    # peak location: migrates forward as incidence increases
    x_peak = 0.09 / (1.0 + 0.06 * max(alpha, -4.0))

    safe = np.clip(xc, 1e-3, None)
    if surface == "upper":
        peak = -strength * np.exp(-((safe - x_peak) ** 2) / (2 * 0.045**2))
        recovery = 0.42 * (safe**0.62) - 0.30
        return peak + recovery
    lift = 0.30 * strength * np.exp(-((safe - 0.16) ** 2) / (2 * 0.11**2))
    return lift + 0.22 * (safe**0.5) - 0.16


def _stderr(xc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Measurement standard error: worst at the leading edge, floor elsewhere.

    This is the whole reason the model uses a fixed-noise likelihood -- error is
    a function of position, not a single number for the whole surface.
    """
    base = 0.004 + 0.030 * np.exp(-xc / 0.07)
    return base * rng.uniform(0.75, 1.25, size=xc.shape)


def generate(
    path: str,
    cfg: DoEConfig,
    n_upper: int = 19,
    n_lower: int = 13,
    seed: int = 0,
) -> str:
    """Write a synthetic dataset to `path` covering every combination in `cfg`."""
    rng = np.random.default_rng(seed)
    xc_u, xc_l = chordwise_stations(n_upper, n_lower)

    with h5py.File(path, "w") as h5:
        root = h5.create_group(cfg.root)
        root.attrs["synthetic"] = True
        root.attrs["features"] = ",".join(cfg.feature_names)

        for combo in cfg.combinations():
            grp = root.require_group("/".join(cfg.key_parts(combo)))
            for surface, xc, n in (("up", xc_u, n_upper), ("low", xc_l, n_lower)):
                clean = _field(xc, combo, "upper" if surface == "up" else "lower")
                se = _stderr(xc, rng)
                noisy = clean + rng.normal(0.0, se)
                grp.create_dataset(f"coefavg{surface}", data=noisy.reshape(n, 1))
                grp.create_dataset(f"coefstderr{surface}", data=se.reshape(n, 1))
                grp.create_dataset(f"xc{surface}", data=xc.reshape(n, 1))

    return path


def summarise(path: str) -> str:
    """One-line description of a generated file, for logging."""
    with h5py.File(path, "r") as h5:
        root = list(h5.keys())[0]
        leaves = []

        def walk(g):
            for k in g:
                obj = g[k]
                if isinstance(obj, h5py.Group):
                    walk(obj)
                elif k.startswith("coefavg"):
                    leaves.append(obj.shape[0])

        walk(h5[root])
    return f"{path}: {len(leaves)} surface records, {sum(leaves)} measurement points"
