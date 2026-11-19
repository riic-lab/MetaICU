"""Build the grid feature manifest for the selected dataset."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from metaicu.cli.datasets import load_grid_module


@hydra.main(version_base=None, config_path="configs", config_name="grid_manifest")
def main(cfg: DictConfig) -> None:
    """Build the grid feature manifest and audit files for whichever dataset= is selected."""

    OmegaConf.resolve(cfg)
    dataset_name = str(OmegaConf.select(cfg, "dataset_name"))
    command = load_grid_module(dataset_name, "cli.grid_build_manifest")
    manifest = load_grid_module(dataset_name, "manifest")
    outputs = manifest.write_grid_manifest_outputs(command.build_config(cfg))
    print(
        json.dumps(
            {name: str(path) for name, path in outputs.items()},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
