# G01 QXU6028 regular-data reproduction

## Run the analysis

Use Python 3.11 or 3.12. From the directory containing `reproduce.py`, install the pinned dependencies once:

```sh
python -m pip install -r requirements.txt
```

Run the analysis and regenerate the results and all six PNGs:

```sh
python reproduce.py
```

The script reads the original G01 regular inputs from the adjacent `data/` folder and writes to `outputs/`. The run needs no network or prior outputs. Use `--data` or `--output` to override either directory. Raw files are validated against their original manifest before analysis.

## Generated files

| File in `outputs/` | Contents |
| --- | --- |
| `results.json` | QXU6028-results-v1 results for G01 regular data |
| `L1_correction_factor.png` | Carrier-count correction curve |
| `model_figure.png` | L2 cone and hard-gap model comparison |
| `trajectory_figure.png` | 56 assigned Fermi levels and the repeat-resampling band in eV |
| `Figure_4_L3_map.png` | 10 × 10 zero-mean conductivity-deviation map |
| `Figure_5_L3_validation.png` | Regularisation trade-off, peak sensitivity, two impulse responses and detection-limit simulation |
| `Supplementary_L3_singular_values.png` | Singular-value spectrum of the original L3 forward matrix |

## Analysis and verification

- L1 integrates the band-edge and Fermi-referenced carrier populations at 300 K, then calculates mobility at the exact metadata operating point.
- L2 uses paired repeats, signed Hall measurements and separate channel normalisation. The bootstrap uses 200 resamples with seed 6028.
- L3 uses first-difference regularisation and a zero-mean projector. Features are the maximum and minimum pixels. The reported resolution is the larger of the two positive-lobe FWHM estimates. The detection limit uses 500 repeat-noise simulations with seed 20260924.
- The current regular-data decision uses the 3.0% limit documented in the archived G01 L3 checkpoint. Confirm it against the Session 3 announcement. The 95% noise envelope covers repeat noise. The nominal decision is `pass: false` at this limit.
- The script checks the supplied results template, finite values, trajectory and map sizes, zero mean, feature coordinates and the decision. The `evidence/` folder records the clean-directory comparison. Last-digit floating-point differences can occur across computing environments.

## Version and report reconciliation

`PIPELINE_VERSION.json` records the local integration commit, the committed source digest and the distributed script digest. The detached script embeds its local release provenance and can run without Git. In the group repository, commit `reproduce.py` and its inputs, then run the script in that checkout to record the group repository's HEAD in `results.json`. The marker-facing repository link and group contribution history belong to the group's submission.

The Word draft previously checked needs to use the following values and units when reporting this run:

| Item | Value from the reproduced analysis |
| --- | --- |
| L1 correction at six decimal places | 2.568739 |
| L3 off-centre FWHM at (2, 7) | 1.303 pitches |
| L3 conservative FWHM | 1.444 pitches |
| L3 95th-percentile detection limit | 0.0000334054 as a fraction, or 0.00334 percentage points |
| L3 reconstructed map mean | Zero to numerical precision |
| Figure 3 vertical axis | eV |
