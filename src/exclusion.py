"""Train/test exclusion — the part that decides whether a number means anything.

A surrogate trained on a grid point and then scored on that same grid point is
reporting interpolation of its own training data. The whole value of the model
is its behaviour at configurations it has not seen, so the framework refuses to
train on anything it is about to be graded on.

Three cases, and they are genuinely different:

``campaign``
    A named validation set. Every configuration in it is removed from training
    before the first gradient step, and predictions are scored against the
    measurements held back.

``exact``
    A user asks for a configuration that happens to exist in the dataset. That
    one configuration is removed, the model retrained, and the prediction
    compared against the measurement. Skipping the removal here is the most
    common way a surrogate ends up reporting an accuracy it does not have.

``interpolated``
    A user asks for a configuration that falls between grid points. There is no
    duplicate to leak, so nothing is excluded and the model trains on everything
    available. Excluding "nearby" points here would be actively harmful: it
    throws away the very neighbours the prediction depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from config import DoEConfig


@dataclass
class Plan:
    """What to withhold, what to predict, and how the result should be read."""

    mode: str                      # "campaign" | "exact" | "interpolated"
    exclude: List[Dict[str, float]]  # withheld from training
    targets: List[Dict[str, float]]  # predicted afterwards
    label: str                     # checkpoint / output key

    @property
    def has_truth(self) -> bool:
        """Whether measurements exist to score the prediction against."""
        return self.mode in ("campaign", "exact")

    def describe(self) -> str:
        if self.mode == "campaign":
            return (
                f"campaign mode: {len(self.exclude)} configuration(s) withheld from "
                "training and scored against measurement"
            )
        if self.mode == "exact":
            return (
                "exact custom: the requested configuration exists in the dataset, "
                "so it is withheld from training before being predicted"
            )
        return (
            "interpolated custom: the request falls between grid points, so no "
            "exclusion is needed and the model trains on the full dataset"
        )


def for_campaign(cfg: DoEConfig, campaign: int) -> Plan:
    if campaign not in cfg.campaigns:
        available = sorted(cfg.campaigns) or ["none defined"]
        raise KeyError(f"campaign {campaign} not in config (available: {available})")
    combos = cfg.campaigns[campaign]
    return Plan(
        mode="campaign",
        exclude=list(combos),
        targets=list(combos),
        label=f"campaign{campaign}",
    )


def for_custom(cfg: DoEConfig, combo: Dict[str, float]) -> Plan:
    missing = set(cfg.feature_names) - set(combo)
    if missing:
        raise ValueError(f"query is missing feature(s): {sorted(missing)}")

    problems = cfg.out_of_bounds(combo)
    if problems:
        raise ValueError("query outside the trained envelope: " + "; ".join(problems))

    if cfg.on_grid(combo):
        return Plan(
            mode="exact",
            exclude=[combo],
            targets=[combo],
            label=f"exact.{cfg.key(combo)}",
        )

    return Plan(
        mode="interpolated",
        exclude=[],
        targets=[combo],
        label="interpolated.all",
    )


def resolve(
    cfg: DoEConfig,
    campaign: Optional[int] = None,
    custom: Optional[Dict[str, float]] = None,
) -> Plan:
    if (campaign is None) == (custom is None):
        raise ValueError("specify exactly one of campaign or custom")
    if campaign is not None:
        return for_campaign(cfg, campaign)
    return for_custom(cfg, custom)


def is_excluded(cfg: DoEConfig, combo: Dict[str, float], plan: Plan) -> bool:
    """Membership by encoded key, so float formatting cannot cause a silent miss."""
    if not plan.exclude:
        return False
    return cfg.key(combo) in {cfg.key(e) for e in plan.exclude}
