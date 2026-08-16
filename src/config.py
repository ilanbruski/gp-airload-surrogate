"""Design-of-experiments configuration.

The parameter grid is data, not code. A campaign lives in a YAML file, gets
validated once here, and everything downstream -- dataset traversal, HDF5 key
construction, exclusion logic, bounds checking on user queries -- reads from
this object. Swapping in a different experiment means writing a different YAML,
not editing the source.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Sequence

import yaml


def encode(value: float) -> str:
    """Filesystem- and HDF5-safe encoding of a numeric parameter value.

    ``-5``   -> ``n5``
    ``0.25`` -> ``0p25``
    ``30``   -> ``30``
    """
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value)
    return text.replace("-", "n").replace(".", "p")


@dataclass(frozen=True)
class Feature:
    name: str
    key: str
    values: Sequence[float]

    @property
    def lo(self) -> float:
        return float(min(self.values))

    @property
    def hi(self) -> float:
        return float(max(self.values))


@dataclass
class Restriction:
    """Not every parameter combination need exist.

    Real test matrices are ragged: a given configuration may only have been run
    at a subset of incidences. ``when`` selects the configuration, ``allow``
    lists the values that exist for it.
    """

    when: Dict[str, float]
    allow: Dict[str, List[float]]

    def applies(self, combo: Dict[str, float]) -> bool:
        return all(combo.get(k) == v for k, v in self.when.items())

    def permits(self, combo: Dict[str, float]) -> bool:
        return all(combo.get(k) in vals for k, vals in self.allow.items())


@dataclass
class DoEConfig:
    root: str
    features: List[Feature]
    restrictions: List[Restriction] = field(default_factory=list)
    campaigns: Dict[int, List[Dict[str, float]]] = field(default_factory=dict)

    # ---- construction -------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "DoEConfig":
        with open(path) as fh:
            raw = yaml.safe_load(fh)

        features = [
            Feature(name=f["name"], key=f["key"], values=list(f["values"]))
            for f in raw["features"]
        ]
        restrictions = [
            Restriction(when=r["when"], allow=r["allow"])
            for r in raw.get("restrictions", [])
        ]
        campaigns = {
            int(k): [dict(c) for c in v] for k, v in (raw.get("campaigns") or {}).items()
        }

        cfg = cls(
            root=raw["root"],
            features=features,
            restrictions=restrictions,
            campaigns=campaigns,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate feature names in config: {names}")
        keys = [f.key for f in self.features]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate feature keys in config: {keys}")

        known = set(self.combination_keys())
        for cid, combos in self.campaigns.items():
            for combo in combos:
                missing = set(names) - set(combo)
                if missing:
                    raise ValueError(
                        f"campaign {cid}: combination {combo} is missing {sorted(missing)}"
                    )
                if self.key(combo) not in known:
                    raise ValueError(
                        f"campaign {cid}: combination {combo} is not in the grid "
                        "(check restrictions)"
                    )

    # ---- grid traversal -----------------------------------------------

    @property
    def feature_names(self) -> List[str]:
        return [f.name for f in self.features]

    def combinations(self) -> Iterator[Dict[str, float]]:
        """Every configuration the design of experiments actually contains."""
        for values in itertools.product(*(f.values for f in self.features)):
            combo = dict(zip(self.feature_names, values))
            if all(
                r.permits(combo) for r in self.restrictions if r.applies(combo)
            ):
                yield combo

    def combination_keys(self) -> List[str]:
        return [self.key(c) for c in self.combinations()]

    # ---- key construction ---------------------------------------------

    def key_parts(self, combo: Dict[str, float]) -> List[str]:
        return [f"{f.key}{encode(combo[f.name])}" for f in self.features]

    def key(self, combo: Dict[str, float]) -> str:
        return ".".join(self.key_parts(combo))

    # ---- query validation ---------------------------------------------

    def in_bounds(self, combo: Dict[str, float]) -> bool:
        return all(f.lo <= float(combo[f.name]) <= f.hi for f in self.features)

    def out_of_bounds(self, combo: Dict[str, float]) -> List[str]:
        return [
            f"{f.name}={combo[f.name]} outside [{f.lo}, {f.hi}]"
            for f in self.features
            if not (f.lo <= float(combo[f.name]) <= f.hi)
        ]

    def on_grid(self, combo: Dict[str, float]) -> bool:
        """True when every feature value coincides with a grid point."""
        return all(float(combo[f.name]) in [float(v) for v in f.values] for f in self.features)
