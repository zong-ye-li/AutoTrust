"""Transactional runner for the non-learning fixed-policy baseline extension.

It deliberately has no training or validation loop.  Each condition/baseline/
environment-seed episode is independently committed through the same atomic
``episode`` routine used by learned-policy evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiment_runner import (
    condition_specs, config_fingerprint, dump, episode, fingerprints, load_config,
    validate_config,
)
from fixed_policy_baselines import FrozenBaseline, baseline_names


def execute_baselines(out, config):
    """Run only frozen baseline episodes in a new append-only output root."""
    validate_config(config)
    if config.get("execution_kind") != "baseline-only":
        raise ValueError("Frozen baseline runner requires execution_kind='baseline-only'")
    names = baseline_names(config)
    if not names:
        raise ValueError("Baseline-only config must name at least one baseline")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / "runner.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = out / "manifest.json"
        signature = fingerprints()
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("config") != config or manifest.get("sha256") != signature:
                raise ValueError("Resume config/source/PMF mismatch")
        else:
            dump(manifest_path, dict(
                execution_kind="baseline-only", config=config,
                config_sha256=config_fingerprint(config), sha256=signature,
                python=sys.version, numpy=np.__version__, platform=platform.platform(),
                created=datetime.now(timezone.utc).isoformat(),
                seed_scheme="eval=3000000+j; frozen-baseline-policy=5000000+10000*baseline_index+j",
                resume="atomic episode boundary; incomplete episode is replayed",
            ))

        started = time.perf_counter()
        completed = 0
        for spec in condition_specs(config):
            cfg = dict(config, **spec)
            condition = out / spec["id"]
            for baseline_index, name in enumerate(names):
                for ev in range(config["evaluation_seeds"]):
                    policy_seed = 5000000 + 10000 * baseline_index + ev
                    episode(
                        condition / name / f"eval-{ev:05d}", cfg,
                        3000000 + ev, policy_seed,
                        FrozenBaseline(
                            name, policy_seed,
                            include_estimated_postattack_credence=config.get('include_estimated_postattack_credence', True),
                        ), False,
                    )
                    completed += 1
            dump(out / "status.json", dict(
                status="running", completed_episodes=completed,
                expected_episodes=len(condition_specs(config)) * len(names) * config["evaluation_seeds"],
                seconds=time.perf_counter() - started,
            ))

        summaries = [json.loads(path.read_text()) for path in out.glob("**/summary.json")
                     if ".partial" not in str(path)]
        total_epochs = sum(summary["epochs"] for summary in summaries)
        total_seconds = sum(summary["seconds"] for summary in summaries)
        size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        dump(out / "status.json", dict(
            status="complete", episodes=len(summaries), expected_episodes=completed,
            seconds_this_invocation=time.perf_counter() - started,
            total_simulation_seconds=total_seconds, epochs=total_epochs, bytes=size,
            free_bytes=shutil.disk_usage(out).free,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            seconds_per_epoch=total_seconds / max(1, total_epochs),
            decisions=sum(summary["decisions"] for summary in summaries),
            transitions=sum(summary["transitions"] for summary in summaries),
        ))
    except BaseException:
        dump(out / "status.json", dict(status="failed", traceback=__import__("traceback").format_exc()))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--readiness", type=Path,
                        help="Fresh readiness evidence required for a formal baseline run")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["mode"] == "formal":
        if args.readiness is None:
            parser.error("Formal baseline run requires --readiness")
        readiness = json.loads(args.readiness.read_text())
        if not readiness.get("passed") or readiness.get("sha256") != fingerprints():
            raise ValueError("Readiness gate missing or stale")
        if readiness.get("formal_config_sha256") != config_fingerprint(config):
            raise ValueError("Readiness config fingerprint is stale")
    execute_baselines(args.output, config)


if __name__ == "__main__":
    main()
