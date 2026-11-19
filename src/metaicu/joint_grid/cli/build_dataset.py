"""Build a joint grid with cohort-prefixed stay and subject IDs.

Each source pipeline extracts its own hourly grid. The joint build pads the
grids to a shared schema, fits scalers across the selected training cohorts,
and passes those scalers to each source's output writer. With one cohort the
writer fits its own scalers. The final assembly combines corresponding splits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from metaicu.cli.datasets import GRID_DATASETS, load_grid_module
from metaicu.common.grid.external_artifacts import train_values as _train_values
from metaicu.common.grid.impute import (
    capture_presence_mask,
    materialize_structural_zero_columns,
)
from metaicu.common.grid.pool_scale import (
    MIN_TRAIN_VALUES,
    compute_cohort_weights,
    pooled_fit_treatment,
    pooled_mean_std,
)
from metaicu.common.grid.scale import LOG_TRANSFORM_TAGS, _apply_log
from metaicu.common.grid.schema_union import (
    compute_union_matches,
    pad_matches_for_cohort,
)
from metaicu.joint_grid.assemble import write_joint_outputs
from metaicu.joint_grid.debug_subset import write_debug_subset

log = logging.getLogger(__name__)

STATIC_NUMERIC_TAGS = ("age", "weight", "height")


def _optional_path(value: Any) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    return Path(str(value)).expanduser()


def _pad_pre_scale_grid(name, pre_scale_grid, union_registry):
    """Pad a cohort's grid to the shared schema and update its observation masks."""
    n_before = len(pre_scale_grid.matches)
    padded_matches = pad_matches_for_cohort(pre_scale_grid.matches, union_registry)
    padded_matches_with_derived = {
        **padded_matches,
        **pre_scale_grid.derived_target_matches,
    }
    grid = materialize_structural_zero_columns(
        pre_scale_grid.grid, padded_matches_with_derived
    )
    grid, presence_mask_cols = capture_presence_mask(grid, padded_matches_with_derived)
    pre_scale_grid.matches = padded_matches
    pre_scale_grid.matches_with_derived = padded_matches_with_derived
    pre_scale_grid.grid = grid
    pre_scale_grid.presence_mask_cols = presence_mask_cols
    log.info(
        f"{name}: padded to {len(union_registry)} union tags "
        f"({len(union_registry) - n_before} newly added)"
    )


def _pooled_observation_scaler(per_cohort_values, weights, log_kind=None):
    """per_cohort_values: {cohort: 1D raw non-null train values}, already restricted to
    contributing cohorts (non-empty, non-structural-zero). Returns a scaler entry shaped like
    scale.py's own observation entries, or None if the pooled total is below MIN_TRAIN_VALUES."""
    total = sum(len(v) for v in per_cohort_values.values())
    if total < MIN_TRAIN_VALUES:
        return None
    transformed = {c: _apply_log(v, log_kind) for c, v in per_cohort_values.items()}
    stats = {
        c: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
        for c, v in transformed.items()
    }
    mean, std = pooled_mean_std(stats, weights)
    return {
        "type": "observation",
        "log": log_kind,
        "mean": mean,
        "std": std if std != 0.0 else 1.0,
    }


def _compute_pooled_scalers(pre_scale_by_cohort, weights, random_seed=42):
    """Returns one external_scalers dict (tag -> scaler entry) covering static (age/weight/
    height) and grid (direct_numeric/derived_output_rate/treatment_rate) tags, pooled across
    every cohort with real (non-structural-zero) train-split values for that tag -- passed
    identically to every cohort's finish_grid_dataset call. Only called when len(datasets) > 1.
    random_seed: passed to pooled_fit_treatment's QuantileTransformer -- a joint-level seed,
    independent of either cohort's own random_seed (this fit isn't tied to one cohort)."""
    pooled = {}

    for tag in STATIC_NUMERIC_TAGS:
        per_cohort_values = {
            name: _train_values(psg.admissions, psg.train_admission_ids, tag)
            for name, psg in pre_scale_by_cohort.items()
        }
        per_cohort_values = {c: v for c, v in per_cohort_values.items() if len(v) > 0}
        if not per_cohort_values:
            continue
        entry = _pooled_observation_scaler(per_cohort_values, weights, log_kind=None)
        if entry is not None:
            pooled[tag] = {**entry, "type": "static"}

    # Every cohort's matches_with_derived is identical post-padding -- any one of them is the
    # full joint tag registry.
    joint_matches = next(iter(pre_scale_by_cohort.values())).matches_with_derived
    for tag, info in joint_matches.items():
        rt = info["reconstruction_type"]
        if rt not in ("direct_numeric", "derived_output_rate", "treatment_rate"):
            continue
        per_cohort_values = {
            name: _train_values(psg.grid, psg.train_admission_ids, tag)
            for name, psg in pre_scale_by_cohort.items()
            if not psg.matches_with_derived.get(tag, {}).get("structural_zero")
        }
        per_cohort_values = {c: v for c, v in per_cohort_values.items() if len(v) > 0}
        if not per_cohort_values:
            continue

        if rt == "treatment_rate":
            qt = pooled_fit_treatment(
                per_cohort_values, weights, tag, random_seed=random_seed
            )
            if qt is not None:
                pooled[tag] = {"type": "treatment", "transformer": qt}
        else:
            entry = _pooled_observation_scaler(
                per_cohort_values, weights, log_kind=LOG_TRANSFORM_TAGS.get(tag)
            )
            if entry is not None:
                pooled[tag] = entry

    return pooled


@hydra.main(
    version_base=None, config_path="../configs", config_name="joint_grid_dataset"
)
def main(cfg: DictConfig) -> None:
    """Build a joint, pooled-statistics AUMCdb+MIMIC-IV (or single-dataset) hourly grid."""

    OmegaConf.resolve(cfg)
    datasets = list(OmegaConf.select(cfg, "datasets") or [])
    unknown = sorted(set(datasets) - set(GRID_DATASETS))
    if not datasets or unknown:
        raise ValueError(
            f"datasets must be a non-empty list drawn from {sorted(GRID_DATASETS)}, got {datasets!r}"
        )

    joint_output_dir = _optional_path(OmegaConf.select(cfg, "joint.output_dir"))
    joint_audit_dir = _optional_path(OmegaConf.select(cfg, "joint.audit_dir"))
    if joint_output_dir is None or joint_audit_dir is None:
        raise ValueError("joint.output_dir and joint.audit_dir are required")
    patients_per_file = int(
        OmegaConf.select(cfg, "joint.patients_per_file", default=1_000)
    )
    joint_random_seed = int(OmegaConf.select(cfg, "joint.random_seed", default=42))

    joint_audit_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=str(OmegaConf.select(cfg, "joint.log_level", default="INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(
                joint_audit_dir / "grid_build_joint_dataset.log", mode="w"
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )

    configs = {}
    workflows = {}
    for name in datasets:
        command = load_grid_module(name, "cli.grid_build_dataset")
        workflows[name] = load_grid_module(name, "build.build_workflow")
        sub_cfg = OmegaConf.select(cfg, f"dataset_configs.{name}")
        if sub_cfg is None:
            raise ValueError(
                f"dataset_configs.{name} is required when datasets includes {name!r}"
            )
        config = command.build_config(sub_cfg)
        if name == "aumcdb" and config.unit_of_analysis != "admission":
            raise ValueError(
                "grid_build_joint_dataset only supports split.unit_of_analysis='admission' -- "
                "subject-grain concatenation across cohorts is not implemented"
            )
        config.audit_dir.mkdir(parents=True, exist_ok=True)
        configs[name] = config

    log.info(f"Building pre-scale grids for {datasets}")
    pre_scale_by_cohort = {
        name: workflows[name].build_pre_scale_grid(configs[name]) for name in datasets
    }

    union_registry = compute_union_matches(
        {name: psg.matches for name, psg in pre_scale_by_cohort.items()}
    )
    for name, psg in pre_scale_by_cohort.items():
        _pad_pre_scale_grid(name, psg, union_registry)

    n_train_admissions = {
        name: len(psg.train_admission_ids) for name, psg in pre_scale_by_cohort.items()
    }
    weights = compute_cohort_weights(n_train_admissions)
    log.info(f"Cohort weights (normalized sqrt(n_train)): {weights}")

    external_scalers = None
    if len(datasets) > 1:
        external_scalers = _compute_pooled_scalers(
            pre_scale_by_cohort, weights, random_seed=joint_random_seed
        )
        log.info(f"Pooled statistics fit for {len(external_scalers)} tags")

    cohort_output_dirs = {}
    for name in datasets:
        workflows[name].finish_grid_dataset(
            configs[name], pre_scale_by_cohort[name], external_scalers=external_scalers
        )
        cohort_output_dirs[name] = configs[name].output_dir

    outputs = write_joint_outputs(
        cohort_output_dirs=cohort_output_dirs,
        joint_output_dir=joint_output_dir,
        joint_audit_dir=joint_audit_dir,
        patients_per_file=patients_per_file,
        weights=weights,
        n_train_admissions_by_cohort=n_train_admissions,
    )
    print(
        json.dumps(
            {name: str(path) for name, path in outputs.items()},
            indent=2,
            sort_keys=True,
        )
    )

    if bool(OmegaConf.select(cfg, "debug.enabled", default=False)):
        debug_output_dir = _optional_path(OmegaConf.select(cfg, "debug.output_dir"))
        debug_audit_dir = _optional_path(OmegaConf.select(cfg, "debug.audit_dir"))
        if debug_output_dir is None or debug_audit_dir is None:
            raise ValueError(
                "debug.output_dir and debug.audit_dir are required when debug.enabled=true"
            )
        debug_outputs = write_debug_subset(
            joint_output_dir=joint_output_dir,
            output_dir=debug_output_dir,
            audit_dir=debug_audit_dir,
            top_pct=float(OmegaConf.select(cfg, "debug.top_pct", default=0.01)),
            decile_frac=float(OmegaConf.select(cfg, "debug.decile_frac", default=0.1)),
            n_deciles=int(OmegaConf.select(cfg, "debug.n_deciles", default=10)),
            microbatch_size=int(
                OmegaConf.select(cfg, "debug.microbatch_size", default=64)
            ),
            seed=int(OmegaConf.select(cfg, "debug.seed", default=42)),
        )
        log.info(f"Debug subset written to {debug_output_dir}")
        print(
            json.dumps(
                {name: str(path) for name, path in debug_outputs.items()},
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
