# MetaICU
MetaICU builds hourly ICU grids from AmsterdamUMCdb and MIMIC-IV. It can also convert AmsterdamUMCdb records into pre-MEDS, MEDS-like events and tokenized timelines.
## Build an hourly grid
From a clone of this repository, use [uv](https://docs.astral.sh/uv/) to install the locked environment. Set `paths.parent_dir` to a workspace containing your input data under `data/raw/`. The [runbook](docs/user_runbook.md#grid-dataset) describes the expected source files and path overrides.
```bash
uv sync --locked
uv run grid_build_dataset dataset=aumcdb paths.parent_dir=/path/to/aumc_workspace
```
For MIMIC-IV, use `dataset=mimic_iv` and its workspace. Each build writes split Parquet shards, admission metadata, the feature schema and preprocessing artifacts under `data/grid/` by default. The integrity report goes under `audits/grid_dataset/`.
To fit scalers across cohorts and produce one dataset with source-prefixed IDs, follow the [joint grid workflow](docs/user_runbook.md#joint-grid-dataset-pooled-aumcdb--mimic-iv). For AmsterdamUMCdb event sequences, start with [vocabulary setup](docs/user_runbook.md#inputs) and follow the [pre-MEDS workflow](docs/user_runbook.md#pre-meds) through tokenization.
## Repository layout
- [aumcdb](src/metaicu/aumcdb) and [mimiciv](src/metaicu/mimiciv) own their source readers, clinical mappings and extraction rules.
- [common/grid](src/metaicu/common/grid) implements operations on the shared hourly representation, including patient splitting, target calculations, scaling, imputation and encoding.
- [cli](src/metaicu/cli) selects the dataset for manifest and standalone grid builds. [joint_grid](src/metaicu/joint_grid) fits pooled statistics, assembles the cohort outputs and writes optional debug subsets.
## Add a grid dataset
1. Use the [MIMIC-IV package](src/metaicu/mimiciv) as a working example. Keep raw table names, identifiers, source units and mapping decisions in the new dataset package. Its grid workflow produces a [PreScaleGrid](src/metaicu/common/grid/pre_scale.py) with raw values, observation masks and training admission IDs. The finishing step accepts fitted scalers and writes the grid and its sidecars.
2. Register the dataset token in the [CLI registry](src/metaicu/cli/datasets.py). Add its [standalone Hydra configuration](src/metaicu/cli/configs/dataset) and its staging configuration in the [joint build](src/metaicu/joint_grid/configs/joint_grid_dataset.yaml). Declare its configuration and manifest files in [pyproject.toml](pyproject.toml) so they ship with the package.
3. Exercise the new source with a small fixture and the public commands. Check patient split isolation, units, observation masks, categorical columns and schema parity with an existing cohort. Add tests for source-specific behavior alongside the dataset and cover shared operations once under `tests/unit/common/grid/`.
See the [test guide](tests/README.md) for the Ruff, unit and integration commands used by CI and the opt-in real-data regressions. The [policy record](docs/policy_decisions.md) documents reviewed AmsterdamUMCdb decisions.
