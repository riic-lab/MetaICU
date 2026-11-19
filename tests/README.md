# Tests
From the repository root, install the locked development environment with `uv sync --locked --extra dev`. GitHub Actions runs the following commands on pushes and pull requests:
```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run python -m unittest discover -s tests/unit -t . -v
uv run python -m unittest discover -s tests/integration -t . -v
```
## Where a test belongs
- `unit/` checks individual operations on in-memory inputs. Put shared grid behavior under `unit/common/grid/` and source-specific decisions under the dataset name.
- `integration/` checks workflows, packaging and output files using bounded fixtures from `fixtures/`. A new case should identify which behavior would fail, such as a patient's admissions crossing splits or a missing observation acquiring a positive mask.
- `real_data/` contains opt-in regressions that require access to the source datasets. See the requirements in the [AmsterdamUMCdb vocabulary](real_data/aumcdb/tokenized/vocab/test_vocab_build_regression.py), [MIMIC-IV grid](real_data/mimiciv/grid/test_grid_build_regression.py) and [joint grid](real_data/joint_grid/test_joint_grid_build_regression.py) tests.
To run a single module, pass its dotted name to unittest. For example:
```bash
uv run python -m unittest tests.unit.common.grid.test_derive_targets -v
```
After configuring the inputs for real-data regressions, run `uv run python -m unittest discover -s tests -t . -v` to include them. Unconfigured real-data tests skip.
