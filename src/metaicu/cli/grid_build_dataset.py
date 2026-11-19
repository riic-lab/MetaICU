"""Build the hourly grid for the selected dataset."""

from __future__ import annotations

import json
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from metaicu.cli.datasets import load_grid_module


@hydra.main(version_base=None, config_path="configs", config_name="grid_dataset")
def main(cfg: DictConfig) -> None:
    """Build a split-aware iCareFM-style hourly grid for whichever dataset= is selected."""

    OmegaConf.resolve(cfg)
    dataset_name = str(OmegaConf.select(cfg, "dataset_name"))
    command = load_grid_module(dataset_name, "cli.grid_build_dataset")
    workflow = load_grid_module(dataset_name, "build.build_workflow")
    config = command.build_config(cfg)
    config.audit_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=str(OmegaConf.select(cfg, "run.log_level", default="INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(config.audit_dir / "grid_build_dataset.log", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    outputs = workflow.write_grid_dataset_outputs(config)
    print(
        json.dumps(
            {name: str(path) for name, path in outputs.items()},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
