"""HDF5 traversal, feature assembly and normalization.

Expected layout, one group per configuration, keyed by the encoded parameter
values from the config:

    <root>/<key0>/<key1>/.../  coefavgup      (n_upper, 1)
                               coefstderrup   (n_upper, 1)
                               xcup           (n_upper, 1)
                               coefavglow     (n_lower, 1)
                               coefstderrlow  (n_lower, 1)
                               xclow          (n_lower, 1)

MATLAB v7.3 files are HDF5 and are read the same way. Arrays are squeezed and
orientation-checked on the way in, because MATLAB and NumPy disagree about which
axis is which often enough that trusting the shape is a bug waiting to happen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from config import DoEConfig
from exclusion import Plan, is_excluded


@dataclass
class Record:
    """One configuration, one surface."""

    combo: Dict[str, float]
    xc: np.ndarray       # (n,) chordwise position
    value: np.ndarray    # (n,) measured coefficient
    stderr: np.ndarray   # (n,) measurement standard error


def _column(node: h5py.Dataset) -> np.ndarray:
    arr = np.asarray(node).squeeze()
    return np.atleast_1d(arr).astype(np.float64)


def read(path: str, cfg: DoEConfig, surface: str) -> List[Record]:
    """Read every configuration present in the file for one surface."""
    if surface not in ("up", "low"):
        raise ValueError(f"surface must be 'up' or 'low', got {surface!r}")

    records: List[Record] = []
    with h5py.File(path, "r") as h5:
        if cfg.root not in h5:
            raise KeyError(f"root group {cfg.root!r} not found in {path}")
        root = h5[cfg.root]

        for combo in cfg.combinations():
            node = root
            for part in cfg.key_parts(combo):
                if part not in node:
                    node = None
                    break
                node = node[part]
            if node is None:
                continue

            try:
                value = _column(node[f"coefavg{surface}"])
                stderr = _column(node[f"coefstderr{surface}"])
                xc = _column(node[f"xc{surface}"])
            except KeyError:
                continue

            n = min(len(value), len(stderr), len(xc))
            records.append(
                Record(combo=dict(combo), xc=xc[:n], value=value[:n], stderr=stderr[:n])
            )

    if not records:
        raise RuntimeError(
            f"no usable records for surface {surface!r} in {path} -- "
            "check that the config root and feature keys match the file"
        )
    return records


def split(
    cfg: DoEConfig, records: Sequence[Record], plan: Plan
) -> Tuple[List[Record], List[Record]]:
    """Partition into training and withheld sets according to the plan."""
    train, held = [], []
    for rec in records:
        (held if is_excluded(cfg, rec.combo, plan) else train).append(rec)
    if not train:
        raise RuntimeError("exclusion removed every configuration; nothing left to train on")
    return train, held


def assemble(
    cfg: DoEConfig, records: Sequence[Record]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack records into (X, y, noise_variance).

    Feature order is ``[x/c, *config features]``. The chordwise coordinate leads
    because it is the only one left dimensional.
    """
    xs, ys, ns = [], [], []
    for rec in records:
        block = np.empty((len(rec.xc), 1 + len(cfg.features)))
        block[:, 0] = rec.xc
        for j, f in enumerate(cfg.features, start=1):
            block[:, j] = float(rec.combo[f.name])
        xs.append(block)
        ys.append(rec.value)
        # standard error -> variance; the /4 mirrors reporting a 2-sigma interval
        ns.append((rec.stderr**2) / 4.0)

    return np.vstack(xs), np.concatenate(ys), np.concatenate(ns)


def query_matrix(
    cfg: DoEConfig, combo: Dict[str, float], xc: np.ndarray
) -> np.ndarray:
    """Build the prediction design matrix for one configuration over a chord sweep."""
    block = np.empty((len(xc), 1 + len(cfg.features)))
    block[:, 0] = xc
    for j, f in enumerate(cfg.features, start=1):
        block[:, j] = float(combo[f.name])
    return block


class Normalizer:
    """Z-score every feature except the chordwise coordinate.

    ``x/c`` already lives on [0, 1], it is the axis every prediction is plotted
    against, and normalizing it only destroys the physical reading of the learned
    lengthscale. Everything else is z-scored so that a parameter measured in the
    tens does not dominate the distance metric purely through its units.
    """

    def __init__(self, X: np.ndarray, skip_first: bool = True):
        self.skip = 1 if skip_first else 0
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.mean[: self.skip] = 0.0
        self.std[: self.skip] = 1.0
        self.std[self.std < 1e-12] = 1.0  # a feature held constant must not blow up

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class TargetScaler:
    """Centre and scale the target so the GP's constant mean starts sensibly."""

    def __init__(self, y: np.ndarray):
        self.mean = float(y.mean())
        self.std = float(y.std()) or 1.0

    def forward(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean

    def inverse_scale(self, sigma: np.ndarray) -> np.ndarray:
        return sigma * self.std

    def forward_variance(self, var: np.ndarray) -> np.ndarray:
        return var / (self.std**2)
