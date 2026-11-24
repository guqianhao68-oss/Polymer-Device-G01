# G01 QXU6028 – Regular L1/L2/L3 final integrated release

## Run on a clean computer

Python 3.11 or newer and the dependencies in `requirements.txt` are needed (tested with Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, Matplotlib 3.10.8). Install once:

```sh
python3 -m pip install -r requirements.txt
```

Then one command regenerates the five report PNGs, complete results.json, diagnostics.json and validation_checks.json from the original regular inputs:

```sh
python3 reproduce.py
```

All default paths are derived from the script's directory. No machine-specific absolute paths, manually edited paths, network access or historical results are used in the calculations. Outputs go to `output/`. The same command works from a fresh extraction, including an archive without Git installed: the release commit is identified from a source-hash-checked version record when Git is unavailable. In a Git checkout the actual HEAD is used and tracked edits are rejected until committed. The manifest validates all nine original input/template/README files. `--data`, `--output`, `--threshold` are optional overrides; the default threshold is 0.03, as documented in the archived G01 regular L3 checkpoint README, pending confirmation against the actual instructor announcement. `results.json` contains the nominal Boolean pass/fail; confidence separately describes the noise-only envelope.

## Sources and method

- `data/`: unchanged original regular G01 inputs retrieved from the archived G01_L3_Checkpoint.zip, verified against its original manifest SHA256. The three L3 files also match the three recently uploaded regular files byte-for-byte. The blank official output template is retained. No blind files are included in the calculation.
- L1: supplied `reproduce_L1(1).py`, finite-temperature integration on 200,001-point grids, with raw sweep cross-check and closed-form check. This matches the uploaded L1 checkpoint JSON exactly. The older combined checkpoint used 20,001 points and gives slightly different last digits.
- L2: supplied `reproduce L2`, same finite grids, paired repeat handling, model choice and 200 resamples (seed 6028). Matches the uploaded L2 checkpoint JSON exactly. The two figures preserve the original analysis, with 300 DPI output.
- L3: archived `G01_L3_Checkpoint.zip` projected Cholesky inversion, first differences, 1e-9 numerical stabiliser, local-extremum selection and 500 noise draws (seed 20260924). Retains all three detected extrema. Matches the archived map, feature positions/amplitudes, resolution and detection limit. Figures use correct physical row-to-y orientation (origin lower); the new Figure 5 combines six validation panels including singular spectrum, sensitivity, two PSFs and noise histogram.
- `PIPELINE_VERSION.json`: actual new integration commit and source hash, NOT the September checkpoint commits. The included Git repository records this new integration only, not student contributions or original group history. If transferring into a group repository, commit the source and inputs there and rerun; do not retain the old pipeline_commit field.
- `output/validation_checks.json`: local checks against the instructor-provided JSON template and physical gates. No instructor executable validator was supplied. Numerical reproducibility across platforms is subject to normal floating-point tolerance; the provided clean runs are compared exactly in the tested environment.
- `evidence/`: fresh execution logs, historical checkpoint numerical comparison, and Word report reconciliation notes. The synchronized report is based on the prior Word report and corrects its numerical/figure inconsistencies; student details, actual group contribution history and signatures must be completed by the group. Disclose AI-assisted integration in the coursework AI log.

## Report figures (five PNGs)
1. `L1_correction_factor.png`
2. `model_figure.png`
3. `trajectory_figure.png`
4. `Figure_4_L3_map.png`
5. `Figure_5_L3_validation.png`
