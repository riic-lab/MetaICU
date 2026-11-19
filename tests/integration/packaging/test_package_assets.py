"""Packaging contract for data required by installed MetaICU commands."""

from __future__ import annotations

import tomllib
import unittest
from importlib.resources import files
from pathlib import Path

from hydra import compose, initialize_config_module

from metaicu.cli.datasets import GRID_DATASETS, load_grid_module

from tests._paths import PROJECT_ROOT

EXPECTED_POLICY_FILES = {
    "generation_summary.json",
    "tier0_baseline_resolution.csv",
    "v4_curated_unmapped.csv",
    "v5_keep_drop_review.csv",
    "v6_semantic_contamination.csv",
    "v7_targeted_refinement.csv",
    "v8_device_outcome_refinement.csv",
    "v9_admission_respiratory_cleanup.csv",
    "v10_medication_atc.csv",
    "v11_listitem_values.csv",
    "v13_lab_consolidation.csv",
}


class PackageAssetTests(unittest.TestCase):
    def test_required_resources_are_present_and_selected_for_packaging(self):
        config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
        patterns = config["tool"]["setuptools"]["package-data"]
        expected = {
            "metaicu.aumcdb.tokenized.vocab_pipeline": {
                f"data/policy_manifests/{name}" for name in EXPECTED_POLICY_FILES
            },
            "metaicu.mimiciv.grid": {
                "data/mimic_grid_feature_manifest_review.md",
                "configs/grid_dataset.yaml",
                "configs/grid_manifest.yaml",
            },
            "metaicu.common.grid": {"data/icarefm_table_s3_features.csv"},
            "metaicu.cli": {f"configs/dataset/{name}.yaml" for name in GRID_DATASETS},
        }
        for package, assets in expected.items():
            with self.subTest(package=package):
                root = PROJECT_ROOT.joinpath("src", *package.split("."))
                selected = {
                    p.relative_to(root).as_posix()
                    for pattern in patterns[package]
                    for p in root.glob(pattern)
                    if p.is_file()
                }
                self.assertFalse(assets - selected, sorted(assets - selected))

    def test_registered_datasets_supply_grid_commands_and_configs(self) -> None:
        root = files("metaicu.cli").joinpath("configs/dataset")
        present = {entry.name for entry in root.iterdir() if entry.is_file()}
        expected = {f"{name}.yaml" for name in GRID_DATASETS}
        self.assertEqual(present, expected)

        workspace = Path("/example")
        with initialize_config_module(
            config_module="metaicu.cli.configs", version_base=None
        ):
            for name in GRID_DATASETS:
                with self.subTest(dataset=name):
                    overrides = [
                        f"dataset={name}",
                        f"paths.parent_dir={workspace}",
                        f"paths.raw_data_dir={workspace / 'raw'}",
                        f"paths.audit_dir={workspace / 'audit'}",
                    ]
                    grid = compose(config_name="grid_dataset", overrides=overrides)
                    manifest = compose(config_name="grid_manifest", overrides=overrides)
                    self.assertEqual(grid.dataset_name, name)
                    self.assertEqual(manifest.dataset_name, name)
                    grid_cli = load_grid_module(name, "cli.grid_build_dataset")
                    manifest_cli = load_grid_module(name, "cli.grid_build_manifest")
                    self.assertEqual(
                        grid_cli.build_config(grid).raw_data_dir, workspace / "raw"
                    )
                    self.assertEqual(
                        manifest_cli.build_config(manifest).audit_dir,
                        workspace / "audit",
                    )
                    workflow = load_grid_module(name, "build.build_workflow")
                    self.assertTrue(callable(workflow.write_grid_dataset_outputs))
                    self.assertTrue(callable(workflow.build_pre_scale_grid))
                    self.assertTrue(callable(workflow.finish_grid_dataset))
                    self.assertTrue(
                        callable(
                            load_grid_module(
                                name, "manifest"
                            ).write_grid_manifest_outputs
                        )
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
