# QXU6028 Polymer Devices · Group 01

This repository contains the regular G01 analysis for carrier counting, density-of-states model selection and conductivity mapping. The timed blind test is archived separately.

## Regular results

| Phase | Result |
| --- | --- |
| L1 | At 300 K the carrier-count correction is 2.568739. The corrected density is 3.40525 × 10¹¹ cm⁻² and the mobility is 32,173.6 cm² V⁻¹ s⁻¹. |
| L2 | The cone model was selected. The gap median and both ends of its 16th–84th percentile interval are 0 kBT across 200 resamples. The trajectory contains 56 Fermi levels. |
| L3 | The peak absolute deviation is 4.1584%. Even after subtracting the 95% repeat-noise detection limit, the deviation exceeds the provisional 3.0% threshold. The regular sample therefore fails this uniformity test. |

The [reference JSON and five figures](outputs/) contain the report results. The [run log](evidence/run_2_clean_directory.log) and [comparison record](evidence/reproduction_comparison.json) identify the analysis commit and environment.

## Reproduce from a clean clone

Use Python 3.12 and install the pinned packages from the repository root.

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

The script verifies the nine input files against the byte lengths and SHA-256 digests in `data/manifest.json`. It writes `results.json` and five PNG figures beside the script. Set `--output rerun` to write into a separate directory, or use `--data` for another copy of the original regular G01 inputs. The regular threshold is fixed in the script. The validation checks run before the results are written.

The JSON records the checked-out Git commit when the script matches the committed source. To reproduce the archived run with the same commit identifier, check out the analysis commit listed in the comparison record. A later commit that adds the run evidence will produce its own commit identifier. The script also carries a source digest and release identifier for use in a detached archive.

## Historical checkpoints

- [L1 checkpoint](G01_L1_Checkpoint/) includes the original correction-factor figure, phase JSON, note and recorded commit.
- [L2 checkpoint](G01_L2_Checkpoint/) includes both original figures, phase JSON, note and recorded commit.
- [Timed blind test](L3-BLINDTEST/) preserves the submitted JSON, frozen pipeline identifier and historical run log. The blind map passed the fixed-reference 3% test with a noise-adjusted upper value of 2.99741%.

The checkpoint commit identifiers are preserved as recorded by the group. The evidence files document the historical runs. The regular pipeline uses only the regular inputs in `data/`.
