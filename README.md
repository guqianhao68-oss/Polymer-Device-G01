# QXU6028 Polymer Devices Virtual Lab

Group 01 (G01) · Academic year 2026–27

This repository presents our three virtual-lab analyses: carrier counting (L1), density-of-states model selection (L2), and spatial reconstruction of relative conductivity (L3). The integrated pipeline reproduces the **regular G01 dataset**. The timed blind-test submission is archived with its own results, run log, and frozen commit.

## Regular dataset results

The reference results are in [`results.json`](results.json), which records pipeline commit `5a10425b615263e8a9e66c71f66586e61e9b932c`.

| Phase | Finding |
| --- | --- |
| L1 · Carrier counting | The band-edge count is **2.568739×** the Fermi-referenced count at the operating point. The corrected carrier density is **3.405251 × 10¹¹ cm⁻²**, and the corrected mobility is **32,173.604 cm² V⁻¹ s⁻¹**. |
| L2 · Density of states | The selection rule favours the **cone** model. The selected gap is **0.00 kBT**, with a 16th–84th percentile interval of **[0.00, 0.00] kBT** under repeat resampling. The Fermi-level trajectory has **56** entries in acquisition order. The interval describes model-selection stability under resampling. |
| L3 · Conductivity uniformity | The reconstructed **10 × 10** zero-mean map has a peak absolute deviation of **4.1584%**. This is a **clear fail** at the **3.0%** decision threshold under the repeat-noise bound. The worst tested resolution is **1.4443 probe pitches**, and the 95th-percentile noise-only detection limit is **0.00334 percentage points**. |

## Repository contents

Keep `README.md`, `reproduce.py`, and the reference `results.json` at the repository root. Keep the three archive folders at the same level and preserve their original filenames. Publish from the existing analysis repository so the recorded commits remain traceable in its history.

## Reproduce the regular analysis

From a clean Git checkout of the repository root, use Python 3.10 or newer and install the analysis dependencies. NumPy 2.0 or newer is required for `numpy.trapezoid`.

```bash
python -m pip install "numpy>=2.0" pandas scipy matplotlib
```

Place the original regular G01 dataset under `data/`. Keep the filenames and file bytes intact because the script checks sizes and SHA-256 hashes against `manifest.json`. A data directory in another location can be passed via `--data`.

```text
data/
├── manifest.json
├── README_FIRST.txt
├── results_template.json
├── L1_data.csv
├── L1_params.json
├── L2_data.csv
├── L2_meta.json
├── L3_A.npy
├── L3_d.csv
└── L3_meta.json
```

Run the full L1–L3 analysis:

```bash
python reproduce.py --data data --output output --threshold 0.03
```

| Generated file | Contents |
| --- | --- |
| `output/results.json` | Main L1–L3 results in the supplied `QXU6028-results-v1` schema |
| `output/diagnostics.json` | Per-phase diagnostics, numerical settings, and dependency versions recorded by the script |
| `output/validation_checks.json` | Validation status, completed checks, and SHA-256 hashes of the result JSON and figures |
| `output/L1_correction_factor.png` | Carrier counts and correction factor across the sweep |
| `output/model_figure.png` | Hall and sheet-resistance model comparisons |
| `output/trajectory_figure.png` | Fermi-level trajectory and pointwise repeat-resampling intervals |
| `output/Figure_4_L3_map.png` | Reconstructed conductivity-deviation map |
| `output/Figure_5_L3_validation.png` | L3 reconstruction validation |

The script checks the dataset identity and manifest before analysis. It uses fixed seeds for L2 repeat resampling (200 fits, seed `6028`) and L3 noise-only reconstruction (500 trials, seed `20260924`). A successful run ends with `Completed G01/regular` and records `all_passed: true` in the validation JSON.

Compare the generated `L1`, `L2`, and `L3` blocks to the reference `results.json` using the same inputs and threshold. The script records the current Git `HEAD` as `pipeline_commit` and requires tracked files to be clean. For a source archive outside Git, it requires a matching `PIPELINE_VERSION.json` with `source_sha256` and `pipeline_commit`.

## Checkpoints and timed blind test

| Record | Contents | Recorded commit |
| --- | --- | --- |
| [L1 checkpoint](G01_L1_Checkpoint/) | Original sweep figure, phase result JSON, and brief note | `4b032eb539068fd75ae4b41d45043d46b5710fbf` |
| [L2 checkpoint](G01_L2_Checkpoint/) | Original model and trajectory figures, phase result JSON, and brief note | `9907f3ac0abb41760a75a4ebf42d224ed0288f90` |
| [Timed blind test](L3-BLINDTEST/) | Original blind output, run log, and frozen commit record | `81ef89e0c53c2c3af4b46901c99ab8b731313c32` |

The timed blind test has `dataset_set: blind`. The archived run logged a **clear pass** for the blind spatial map at the same 3.0% threshold, with peak absolute deviation plus the 95% noise-only detection limit equal to **2.99741%**. The regular-data command above accepts the regular dataset. The archived blind run details are in [`L3-BLINDTEST/run_log.txt`](L3-BLINDTEST/run_log.txt).
