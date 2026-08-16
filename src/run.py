#!/usr/bin/env python3
"""Command-line entry point: train a surrogate, predict, score, plot.

    # generate a synthetic dataset and run the campaign-1 validation
    python src/run.py --generate --campaign 1

    # query a configuration that exists in the grid (withheld, then predicted)
    python src/run.py --custom "alpha=5,station=0.25,speed=30"

    # query a configuration between grid points (nothing withheld)
    python src/run.py --custom "alpha=2.5,station=0.3,speed=27"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset as ds
import exclusion
import plots
import synth
from config import DoEConfig
from gp import Surrogate, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def parse_custom(text: str) -> Dict[str, float]:
    """``"alpha=5,station=0.25"`` -> ``{"alpha": 5.0, "station": 0.25}``"""
    out: Dict[str, float] = {}
    for chunk in text.split(","):
        if "=" not in chunk:
            raise ValueError(f"malformed term {chunk!r}; expected name=value")
        name, value = chunk.split("=", 1)
        out[name.strip()] = float(value)
    return out


def surface_label(surface: str) -> str:
    return "upper" if surface == "up" else "lower"


def run_surface(cfg, data_path, surface, plan, args, outdir):
    print(f"\n=== {surface_label(surface)} surface — {plan.label} ===")
    print(f"    {plan.describe()}")

    records = ds.read(data_path, cfg, surface)
    train, held = ds.split(cfg, records, plan)
    print(f"    {len(train)} configuration(s) in training, {len(held)} withheld")

    X, y, noise = ds.assemble(cfg, train)
    norm = ds.Normalizer(X)
    scaler = ds.TargetScaler(y)

    model = Surrogate(device=args.device, seed=args.seed)
    model.fit(
        norm(X),
        scaler.forward(y),
        scaler.forward_variance(noise),
        iters=args.iters,
        lr=args.lr,
        log_every=max(args.iters // 6, 1),
        verbose=not args.quiet,
    )

    truth_by_key = {cfg.key(r.combo): r for r in held}
    xc_grid = np.linspace(0.0, 1.0, 200)
    results = []

    for combo in plan.targets:
        key = cfg.key(combo)
        Xq = ds.query_matrix(cfg, combo, xc_grid)
        mean, sigma = model.predict(norm(Xq))
        mean = scaler.inverse(mean)
        sigma = scaler.inverse_scale(sigma)

        truth_rec = truth_by_key.get(key)
        entry = {"key": key, "combo": combo, "surface": surface_label(surface)}

        if truth_rec is not None:
            Xt = ds.query_matrix(cfg, combo, truth_rec.xc)
            tmean, tsigma = model.predict(norm(Xt))
            tmean = scaler.inverse(tmean)
            tsigma = scaler.inverse_scale(tsigma)
            entry.update(
                metrics(
                    truth_rec.value,
                    tmean,
                    tsigma,
                    noise_var=(truth_rec.stderr**2) / 4.0,
                )
            )
            print(
                f"    {key:<28} RMSE {entry['rmse']:.4f}  R² {entry['r2']:+.3f}  "
                f"coverage {entry['coverage_latent']:.2f} latent / "
                f"{entry['coverage_predictive']:.2f} predictive"
            )
        else:
            entry["note"] = "no measurement withheld — interpolated query"
            print(f"    {key:<28} predicted (no truth available)")

        fig_path = os.path.join(outdir, f"{key}.{surface}.png")
        plots.prediction(
            fig_path,
            xc_grid,
            mean,
            sigma,
            truth_xc=truth_rec.xc if truth_rec else None,
            truth=truth_rec.value if truth_rec else None,
            title=f"{surface_label(surface)} surface — "
                  + ", ".join(f"{k}={v}" for k, v in combo.items()),
        )
        entry["figure"] = os.path.relpath(fig_path, ROOT)
        results.append(entry)

    diag_path = os.path.join(outdir, f"diagnostics.{plan.label}.{surface}.png")
    plots.diagnostics(diag_path, model.history, cfg.feature_names)

    return results, model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "config", "demo.yaml"))
    p.add_argument("--data", default=os.path.join(ROOT, "data", "synthetic.h5"))
    p.add_argument("--generate", action="store_true",
                   help="regenerate the synthetic dataset before running")
    p.add_argument("--surface", choices=["up", "low", "both"], default="both")
    p.add_argument("--campaign", type=int)
    p.add_argument("--custom", type=str)
    p.add_argument("--iters", type=int, default=700)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.campaign is None and args.custom is None:
        args.campaign = 1

    cfg = DoEConfig.load(args.config)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.data), exist_ok=True)

    if args.generate or not os.path.exists(args.data):
        synth.generate(args.data, cfg, seed=args.seed)
        print(synth.summarise(args.data))

    try:
        plan = exclusion.resolve(
            cfg,
            campaign=args.campaign,
            custom=parse_custom(args.custom) if args.custom else None,
        )
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    surfaces = ["up", "low"] if args.surface == "both" else [args.surface]
    all_results = []
    for surface in surfaces:
        results, _ = run_surface(cfg, args.data, surface, plan, args, args.outdir)
        all_results.extend(results)

    scored = [r for r in all_results if "rmse" in r]
    if scored:
        print("\n=== summary over withheld configurations ===")
        print(f"    mean RMSE      {np.mean([r['rmse'] for r in scored]):.4f}")
        print(f"    mean R²        {np.mean([r['r2'] for r in scored]):+.3f}")
        print(f"    mean coverage  {np.mean([r['coverage_latent'] for r in scored]):.3f}"
              f" latent / {np.mean([r['coverage_predictive'] for r in scored]):.3f}"
              " predictive   (nominal 0.95)")
        print(f"    mean sharpness {np.mean([r['sharpness_predictive'] for r in scored]):.4f}")

        plots.calibration(
            os.path.join(args.outdir, f"calibration.{plan.label}.png"),
            [{"label": f"{r['key']} · {r['surface']}",
              "coverage_latent": r["coverage_latent"],
              "coverage_predictive": r["coverage_predictive"]} for r in scored],
        )

    report = os.path.join(args.outdir, f"results.{plan.label}.json")
    with open(report, "w") as fh:
        json.dump({"plan": plan.mode, "label": plan.label, "results": all_results},
                  fh, indent=2)
    print(f"\nfigures and metrics written to {os.path.relpath(args.outdir, ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
