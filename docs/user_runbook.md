# MetaICU User Runbook

This runbook expands the README command sequence. The package currently supports:

- external resource setup
- reviewed supplied-vocabulary installation and optional raw-data rebuild
- iCareFM-style grid feature manifest
- iCareFM-style hourly grid dataset build
- subject-level train/val/test split creation
- source-preserving pre-MEDS extraction
- train-derived high-frequency numeric inventory
- causal mean-binned `numericitems_binned` outputs
- MEDS-like QC conversion for a supplied pre-MEDS directory
- ETHOS-style tokenized safetensor outputs

## Workspace Layout

Use one workspace per run:

```text
/path/to/aumc_workspace/
├── data/
│   ├── raw/              raw AmsterdamUMCdb CSV files
│   ├── raw_shards/       internal schema-cast parquet cache for large raw tables
│   ├── pre-MEDS/         pre-MEDS parquet outputs
│   ├── MEDS/             MEDS-like parquet outputs
│   ├── tokenized/        tokenized safetensor outputs
│   └── metadata/         splits, high-frequency inventory, quantile boundaries
├── externals/            GitHub external resources
│   └── omop_vocab/       manually downloaded Athena/OMOP CSV export
├── vocab/                supplied vocabulary artifact
└── audits/               audit JSON/CSV outputs
```

## Install

```bash
git clone https://github.com/AmirV97/MetaICU.git /path/to/MetaICU
cd /path/to/MetaICU
python -m pip install -e .
```

Main CLI entry points:

```bash
retrieve-aumc-externals --parent-dir /path/to/aumc_workspace
build-amsterdam-vocab paths.parent_dir=/path/to/aumc_workspace
grid_build_manifest paths.parent_dir=/path/to/aumc_workspace
build-aumc-premeds paths.parent_dir=/path/to/aumc_workspace
build-aumc-meds paths.parent_dir=/path/to/aumc_workspace
```

## Inputs

The reviewed Amsterdam vocabulary is bundled with MetaICU. Install it into a
workspace without raw data or external resources:

```bash
build-amsterdam-vocab paths.parent_dir=/path/to/aumc_workspace
```

Output:

```text
/path/to/aumc_workspace/vocab/aumc_supplied_vocab.csv
```

To install a different reviewed vocabulary file instead of the bundled artifact:

```bash
build-amsterdam-vocab \
  paths.parent_dir=/path/to/aumc_workspace \
  paths.supplied_vocab=/path/to/reviewed_vocab.csv
```

To rebuild and audit the vocabulary against local raw data, first retrieve the
external repositories:

```bash
retrieve-aumc-externals --parent-dir /path/to/aumc_workspace
```

Place or symlink the raw AmsterdamUMCdb CSVs into:

```text
/path/to/aumc_workspace/data/raw/
```

Download Athena/OMOP from `https://athena.ohdsi.org/vocabulary/list`. Select
SNOMED, LOINC, RxNorm, RxNorm Extension, ATC, UCUM, and OMOP Extension, then
extract the CSVs into:

```text
/path/to/aumc_workspace/externals/omop_vocab/
```

Run the optional rebuild:

```bash
build-amsterdam-vocab \
  paths.parent_dir=/path/to/aumc_workspace \
  run.mode=rebuild \
  run.overwrite=true
```

The rebuild extracts Latin-1 source tokens, generates evidence/candidate audits,
replays the reviewed mapping and policy manifests, and validates the result.
Candidate evidence is retained for audit; the reviewed baseline manifest owns
target selection. New uncovered source tokens fail by default. Use
`run.allow_unresolved_source_tokens=true` only for bounded or audit-only runs.

## Grid Feature Manifest

```bash
grid_build_manifest paths.parent_dir=/path/to/aumc_workspace
```

This writes:

```text
grid/aumc_grid_feature_manifest.csv
audits/grid_manifest/grid_manifest_summary.json
audits/grid_manifest/grid_manifest_unmatched_features.csv
audits/grid_manifest/grid_manifest_source_candidate_examples.csv
```

The manifest is stage 1 of the iCareFM-style hourly-grid fork. It has one row per packaged Table S3 feature tag and records broad source candidates from the source vocab, supplied vocab, and OpenICU AUMC mappings. It does not scan raw AUMC rows, decide unit conversion, or construct the grid dataset.

## Grid Dataset

```bash
grid_build_dataset dataset=aumcdb paths.parent_dir=/path/to/aumc_workspace
grid_build_dataset dataset=mimic_iv paths.parent_dir=/path/to/mimic_workspace
```

Use `grid_build_dataset` for ordinary single-dataset builds. Use `grid_build_joint_dataset`
(below) only when pooled cross-dataset fitting or one flat, namespaced output is required.

The AmsterdamUMCdb grid and tokenized pipelines share the Latin-1-preserving cache under
`data/raw_shards/`. The first pipeline to run builds schema-cast parquet shards
for `numericitems`, `listitems`, and `drugitems`; later grid or pre-MEDS runs
reuse them. Set `run.rebuild_raw_shards=true` only when the raw source files or
canonical raw schemas have changed.

For MIMIC-IV, point `paths.raw_data_dir` at the prepared MIMIC export containing `icu/` and `hosp/`. The reader expects the compressed event tables and `hosp/patients.parquet` with `subject_id`, `gender` and `year_of_birth`. Both datasets accept `paths.raw_shards_dir`, `paths.output_dir` and `paths.audit_dir`. Reusing a grid output directory requires `run.overwrite=true`, which removes previously generated grid artifacts.
Both pipelines harmonize units, filter implausible values and aggregate measurements by hour. They compute the P/F ratio and weight-normalized urine rate from unscaled values and record a binary observation mask for each continuous observation or derived-rate feature before imputation. They then fit scalers on training data, fill missing values and encode categories. Each hourly row also carries age, weight, height, sex, admission urgency, admission origin and ethnicity. See the [preprocessing reference](../src/metaicu/aumcdb/grid/data/icarefm_preprocessing_reference.md) for the feature-level rules.
The default inclusion criteria require an ICU stay of at least four hours, numeric observations in at least four distinct hours, and at most 48 hours between consecutive observed hours. Patients are split 80/10/10 across training, validation and test so that repeat admissions stay together. AmsterdamUMCdb's standalone build supports `split.unit_of_analysis=subject` to concatenate each patient's admissions chronologically. Both datasets default to one sample per admission.
The grid uses seed 42 by default. To change it via the standalone dispatcher, pass `+run.random_seed=7`. Its dataset configurations also list `run.seed` and `split.seed`, which the builders currently ignore. In a joint build, set `dataset_configs.<name>.run.random_seed` for each source and `joint.random_seed` for the pooled treatment fit.

Outputs under `paths.output_dir`:

```text
train|val|test/N.parquet              per-split hourly grid shards
metadata.csv                          one row per admission (or subject)
feature_schema.json                   per-column reconstruction type, unit, presence-mask column
tte_targets.json                      TTE targets and missing tags (35 targets in the full build)
scalers.pkl / scalers.summary.json    train-fitted scalers
categorical_encoding.csv              one-hot column layout (static + grid)
```

`feature_schema.json` also records each logical feature's `physical_columns`, including one-hot expansions and an empty list for a declared feature with no physical column. Every successful build writes `grid_integrity_summary.json` under `paths.audit_dir`; the build fails on split leakage, shard/metadata disagreement, duplicate or non-dense hour keys, schema-order drift, invalid masks/one-hot groups, non-finite values, or inconsistent TTE declarations.

## Joint Grid Dataset (pooled AUMCdb + MIMIC-IV)

```bash
grid_build_joint_dataset datasets=[mimic_iv,aumcdb] \
  dataset_configs.mimic_iv.paths.parent_dir=/path/to/mimic_workspace \
  dataset_configs.aumcdb.paths.parent_dir=/path/to/aumc_workspace \
  joint.output_dir=/path/to/joint/data joint.audit_dir=/path/to/joint/audits
```

`datasets` selects `[mimic_iv]`, `[aumcdb]`, or `[mimic_iv, aumcdb]`. Each `dataset_configs.<name>`
block is that cohort's own per-cohort **staging** build (same `paths`/`split`/`run` shape as the
standalone `grid_build_dataset`, written in full to its own `paths.output_dir`/`paths.audit_dir`).
With more than one dataset selected, every scaled tag (static age/weight/height and every grid
`direct_numeric`/`derived_output_rate`/`treatment_rate` tag with real data in at least one cohort)
uses pooled training data. Each cohort's weight is proportional to the square root of its
number of training admissions. For a given feature, contributing cohorts' weights sum to one.

The final joint output under `joint.output_dir` always uses one flat layout, regardless of how
many datasets were selected:

```text
train|val|test/N.parquet   re-sharded hourly grid shards, concatenated across cohorts
metadata.csv               one row per admission, plus a `source` column ("aumcdb"/"mimic_iv")
feature_schema.json        identical across cohorts by construction (data-content invariant)
tte_targets.json           union of every cohort's own TTE target list
scalers.pkl                fitted normalizers (z-score mean/std, QuantileTransformer objects) --
                            with >1 dataset selected, every pooled tag's entry here is the exact
                            fit applied to BOTH cohorts; keep this alongside the joint output to
                            preprocess a future inference-time subject the same way
scalers.summary.json       human-readable provenance for scalers.pkl (pooled tags, contributing
                            cohorts, weights)
```

`admissionid` and `subject_id` are source-prefixed strings, including in single-source joint builds. Use `source`, `native_admissionid` and `native_subject_id` in `metadata.csv` to recover the original IDs. Joint builds require admission-level rows.
Both cohorts report mortality for the admission. MIMIC-IV uses `hospital_expire_flag`. AmsterdamUMCdb assigns death only to the patient's last included admission, from admission through 24 hours after discharge. When discharge time is unavailable, the upper bound is open.

### Debug subset (small, pooled, LOS-stratified)

```bash
grid_build_joint_dataset datasets=[mimic_iv,aumcdb] \
  dataset_configs.mimic_iv.paths.parent_dir=/path/to/mimic_workspace \
  dataset_configs.aumcdb.paths.parent_dir=/path/to/aumc_workspace \
  joint.output_dir=/path/to/joint/data joint.audit_dir=/path/to/joint/audits \
  debug.enabled=true debug.output_dir=/path/to/joint/debug_data debug.audit_dir=/path/to/joint/debug_audits
```

`debug.enabled=true` builds a second, much smaller dataset from the just-written joint output
(`metaicu.joint_grid.debug_subset.write_debug_subset`) -- same output-contract shape as any other grid
dataset directory (`metadata.csv`, `feature_schema.json`, shards, sidecars), so it can be pointed
at by anything that reads a normal grid dataset. Meant for exercising the training pipeline
end-to-end (padding/masking edge cases, short- and long-sequence behavior, microbatch shapes)
without paying for a full-size dataset. Selection, computed **independently per `source` cohort**
(AUMCdb and MIMIC-IV admission-length distributions differ, so a single pooled threshold/decile
split would let the larger cohort's shape dictate both):

1. every admission at/above that cohort's own `p(1 - debug.top_pct)` `los_hours` quantile (the
   tail, always including that cohort's max -- a few hundred rows total across both cohorts at the
   default `top_pct=0.01`);
2. `debug.decile_frac` of the remainder, sampled independently within each of that cohort's own
   `debug.n_deciles` `los_hours` deciles, so short-admission buckets stay populated -- a random
   microbatch drawn from this debug set should still see short admissions, not only the LOS tail.

Both cohorts are sampled at the same `top_pct`/`decile_frac` rate (not to equal counts), so the
debug set's AUMCdb:MIMIC-IV admission ratio approximates the full joint dataset's ratio.
`debug.microbatch_size` (default 64) is purely diagnostic: `debug_sample_summary.json` (written to
`debug.audit_dir`) logs a warning if the smallest per-source decile sampled fewer admissions than
this, so an unusably-thin bucket is visible at build time rather than discovered later while
debugging training; it is not enforced. Defaults: `top_pct=0.01`, `decile_frac=0.1`,
`n_deciles=10`, `seed=42`. Can also be run standalone against an already-built joint output by
calling `write_debug_subset(joint_output_dir=..., output_dir=..., audit_dir=..., ...)` directly --
it only reads `joint_output_dir` from disk, it does not need the joint build to be in the same
process.

## Pre-MEDS

```bash
build-aumc-premeds paths.parent_dir=/path/to/aumc_workspace
```

Pre-MEDS creates or reuses `data/metadata/subject_splits.parquet` using the default 80/10/10 subject split, then writes:

```text
data/metadata/subject_splits.parquet   deterministic subject split
data/raw_shards/                       reusable raw parquet cache for large tables
data/pre-MEDS/                         combined source-preserving pre-MEDS
data/pre-MEDS/train|val|test/          split-specific pre-MEDS
data/pre-MEDS/train|val|test/numericitems_binned/
data/metadata/hf_numeric_inventory.csv
data/metadata/hf_numeric_binning_summary.json
audits/pre-MEDS/premeds_summary.json
```

The raw shard cache is source-preserving and shared with the grid pipeline: no patient filtering, vocab decisions, sentinel drops, or time transforms are applied there. The high-frequency inventory is fitted on train only. Causal mean-binning is then applied to each split using that frozen inventory. Rare-but-dense numeric variables, such as CRRT settings that appear in few stays but are very dense when present, are also flagged for binning; the default rare-dense gate requires 2 sampled admission/item groups plus a high row burden and a high-frequency CI above the threshold. Empty bins are not emitted.

Selected repeated categorical state streams are state-change deduplicated in pre-MEDS. Defaults include ventilation mode, CRRT modality, rhythm, oxygen/admin route, NIV program status, sputum state, pupil state, selected score components/final scores, ectopy, chest-drain state, and EVD state. The policy keeps the first value and later value changes per admission/item.

Useful QC run:

```bash
build-aumc-premeds paths.parent_dir=/path/to/aumc_workspace \
  run.num_patients=1000 run.max_rows=1000000
```

Full source-preservation and split-integrity audit:

```bash
audit-aumc-premeds --parent-dir /path/to/aumc_workspace
```

This performs streaming two-seed source-row fingerprint checks for the large
tables, reconciles documented exclusions and categorical state-change
deduplication, verifies subject-level split isolation and split-output parity,
and checks raw numeric passthrough plus causal hourly-bin invariants. Results
are written under `audits/pre-MEDS/full_integrity/`; required-check failures
produce a nonzero exit code.

## MEDS-Like QC Conversion

Split-aware MEDS conversion:

```bash
build-aumc-meds paths.parent_dir=/path/to/aumc_workspace
```

When split pre-MEDS folders exist, this writes `data/MEDS/{train,val,test}/`. Numeric quantile boundaries are fit on train only, saved to `data/metadata/numeric_quantile_boundaries.parquet`, and reused for all splits.

Death is assigned to at most one ICU admission. `MEDS_DEATH` is emitted for an in-ICU death or a death recorded within 24 hours after discharge. Later deaths remain in `data/metadata/subjects.parquet` and `data/metadata/admissions.parquet` without extending the token timeline.

Before fitting or applying those boundaries, raw numeric values go through `tokenized/meds/numeric_qc.py`, which reuses the grid pipeline's itemid-level corrections (`grid/build/unit_conversion_overrides.py`, `plausibility_bounds.py`, and the manifest's rejected duplicates like `pt`) so the same mislabeled units, device sentinels, and implausible readings don't corrupt either pipeline's numeric values. It's keyed by itemid, the only thing the two pipelines' vocabularies actually share -- grid pools itemids into physiology tags, tokenized keeps each itemid as its own token.

One-cohort QC is still available:

```bash
build-aumc-meds paths.parent_dir=/path/to/aumc_workspace run.split_outputs=false
```

Numeric MEDS conversion uses `numericitems_binned` when present, otherwise raw `numericitems`.

## Tokenization

```bash
build-aumc-tokenized paths.parent_dir=/path/to/aumc_workspace
```

This builds the token vocabulary from `data/MEDS/train` only and reuses that frozen vocabulary for `val` and `test`.
Use `vocab_scope=full_data` only when a descriptive vocabulary covering all
three splits is required:

```bash
build-aumc-tokenized paths.parent_dir=/path/to/aumc_workspace vocab_scope=full_data
```

The durable vocabulary copy is written under `mappings/` as
`aumc_token_vocab_train_only_tN.csv` or `aumc_token_vocab_full_tN.csv`.
Codes seen only in `val` or `test` are mapped to the frozen `UNK` token and recorded in the tokenization unknown-code audit.

Full safetensor integrity audit:
```bash
audit-aumc-tokenized --parent-dir /path/to/aumc_workspace
```
This reconciles tensor shapes, vocabulary IDs, timeline offsets and identities, timestamp ordering, terminal tokens, shared quantile-token adjacency, stored token counts, and deterministic sampled MEDS-to-token tracebacks. It also writes sequence-length and timeline-duration histograms under `audits/tokenization/full_integrity/`.

Outputs:

```text
data/tokenized/train|val|test/*.safetensors
data/tokenized/train/vocab_t*.csv
data/tokenized/train/vocab_decoded.csv
data/tokenized/train/code_counts.csv
data/tokenized/train/interval_estimates.json
data/tokenized/metadata/codes.parquet
data/tokenized/metadata/timeline_index.parquet
data/tokenized/metadata/subjects.parquet
data/tokenized/metadata/admissions.parquet
audits/tokenization/tokenization_summary.json
```

Each tokenized timeline is one ICU admission/stay, keyed by `(subject_id, hadm_id)`. Multi-admission patients therefore produce multiple timelines, but patient/admission identity is preserved through `patient_ids`, `hadm_id`, `icustay_id`, and `timeline_index.parquet`.

The tokenized unit of analysis is configurable:

```bash
build-aumc-tokenized paths.parent_dir=/path/to/aumc_workspace run.analysis_unit=stay
build-aumc-tokenized paths.parent_dir=/path/to/aumc_workspace run.analysis_unit=subject
```

`stay` is the default and writes one timeline per ICU admission/stay. `subject` writes one timeline per patient, concatenating that patient's admissions chronologically. Per-token `hadm_id` and `icustay_id` are still stored, so admission identity remains recoverable.

Medication ATC detail can be coarsened at tokenization time:

```bash
build-aumc-tokenized paths.parent_dir=/path/to/aumc_workspace run.medication_atc_depth=2
```

## Tests
See the [test guide](../tests/README.md) for the CI commands, synthetic fixtures and opt-in real-data regressions.
## Next Development Tasks

- Re-run bounded pre-MEDS -> MEDS -> tokenization QC after the aggressive rare-dense binning default. Confirm whether CRRT dilution/flow streams are now binned rather than raw-passthrough.
- Decide the policy for any remaining super-long ICU stays before full tokenization runs. Some stays are genuinely long, e.g. 37-95 ICU days, so hourly data alone can still produce long sequences.
