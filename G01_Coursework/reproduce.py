#!/usr/bin/env python3
"""Reproduce the QXU6028 G01 regular-data results and five report figures.

Place the original data/ directory next to this script, then run:
    python reproduce.py

The data files can also sit next to this script. Outputs go beside the script
by default. Use --data or --output to select a different directory.
Requires Python 3.10+, NumPy 2.x, pandas 2.x and Matplotlib 3.8+.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

# Limit BLAS threads for small matrices and reproducible reduction order.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
THRESHOLD = 0.03  # Provisional regular-data specification used in the G01 report.
FIGURES = (
    "L1_correction_factor.png",
    "model_figure.png",
    "trajectory_figure.png",
    "Figure_4_L3_map.png",
    "Figure_5_L3_validation.png",
)
INPUT_FILES = {
    "L1_data.csv", "L1_params.json", "L2_data.csv", "L2_meta.json",
    "L3_A.npy", "L3_d.csv", "L3_meta.json", "README_FIRST.txt",
    "results_template.json",
}


def check(name, condition, detail=""):
    if not condition:
        raise ValueError(f"{name} failed. {detail}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_inputs(data):
    manifest = read_json(data / "manifest.json")
    check("Regular G01 manifest", (manifest["group"], manifest["dataset_set"]) == ("G01", "regular"))
    entries = manifest["files"]
    check("Manifest file list", len(entries) == len(INPUT_FILES)
          and {entry["name"] for entry in entries} == INPUT_FILES)
    for entry in entries:
        raw = (data / entry["name"]).read_bytes()
        check(f"Manifest SHA-256 and length for {entry['name']}",
              len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"])
    for name in ("L1_params.json", "L2_meta.json", "L3_meta.json", "results_template.json"):
        obj = read_json(data / name)
        check(f"Regular G01 identity of {name}",
              (obj["group"], obj["dataset_set"]) == ("G01", "regular"))


def pipeline_commit():
    script = ROOT / "reproduce.py"
    source = script.read_bytes().replace(b"\r\n", b"\n")
    try:
        committed = subprocess.check_output(
            ["git", "show", "HEAD:./reproduce.py"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        if committed == source:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
            if re.fullmatch(r"[0-9a-f]{40}", commit):
                return commit
    except (OSError, subprocess.CalledProcessError):
        pass

    # A detached distribution contains the real local analysis commit in a
    # comment trailer. Its digest covers all executable source bytes.
    canonical, marker, trailer = source.rpartition(b"\n# Detached release metadata.\n")
    match = re.fullmatch(rb"# Commit: ([0-9a-f]{40})\n# SHA-256: ([0-9a-f]{64})\n", trailer)
    if marker and match and hashlib.sha256(canonical).hexdigest().encode() == match[2]:
        return match[1].decode()
    raise ValueError("Cannot identify the pipeline source. Commit reproduce.py to Git and rerun.")


KB_EV_K = 8.617333e-5
NA_CM2 = 3.816e15
E_CHARGE_C = 1.602176634e-19
EMAX_EV = 2.5


def fermi(E_eV: np.ndarray, EF_eV: float, T_K: float) -> np.ndarray:
    """Fermi-Dirac occupation, with energy in eV."""
    x = np.clip((E_eV - EF_eV) / (KB_EV_K * T_K), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(x))


def n_band(EF_eV: float, T_K: float, c: float) -> float:
    """Band-edge carrier count: integrate from E=0 to 2.5 eV."""
    E = np.linspace(0.0, EMAX_EV, 200_001)
    return float(NA_CM2 * np.trapezoid(c * E * fermi(E, EF_eV, T_K), E))


def n_fermi(EF_eV: float, T_K: float, c: float) -> float:
    """Fermi-referenced carrier count: integrate from E=EF to 2.5 eV."""
    E = np.linspace(EF_eV, EMAX_EV, 200_001)
    return float(NA_CM2 * np.trapezoid(c * E * fermi(E, EF_eV, T_K), E))


def n_fermi_closed(EF_eV: float, T_K: float, c: float) -> float:
    """Closed-form check for the Fermi-referenced integral."""
    kT = KB_EV_K * T_K
    return float(
        NA_CM2 * c * kT
        * (EF_eV * math.log(2.0) + (math.pi**2 / 12.0) * kT)
    )


def crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """Linearly interpolate the first x position where y crosses level."""
    i = np.where(np.diff(np.sign(y - level)) != 0)[0][0]
    return float(np.interp(level, y[i : i + 2], x[i : i + 2]))


def run_l1(data: Path, out: Path):
    params = read_json(data / "L1_params.json")
    raw = pd.read_csv(data / "L1_data.csv")
    check("L1 input columns", raw.columns.tolist() == params["columns"])
    check("L1 raw sweep finite and ordered", np.isfinite(raw.to_numpy()).all() and np.all(np.diff(raw.EF_eV)>0))
    check("L1 sheet resistance matches metadata", np.allclose(raw.rho_ohm_sq, params["rho_ohm_sq"], rtol=1e-10))

    c = float(params["c"])
    T = float(params["T"])
    EF_op = float(params["EF_op_eV"])
    rho = float(params["rho_ohm_sq"])

    calculated_sweep = np.array([n_band(x, T, c) for x in raw.EF_eV])
    check("L1 supplied band-edge sweep agrees with model", np.allclose(calculated_sweep, raw.n_band_cm2, rtol=1e-5))
    # Published L1 gates.
    gates = {
        ef: n_band(ef, T, c) / n_fermi(ef, T, c)
        for ef in (0.0, 0.073, 0.100, 0.200)
    }
    for energy, expected, tolerance in ((0, 1, 2e-5), (0.073, 2, 0.03),
                                         (0.100, 2.60, 0.04), (0.200, 5.10, 0.06)):
        check(f"L1 gate at {energy:g} eV", abs(gates[energy] - expected) < tolerance)

    n_f_op = n_fermi(EF_op, T, c)
    n_f_closed = n_fermi_closed(EF_op, T, c)
    check("L1 closed-form carrier density", abs(n_f_op / n_f_closed - 1.0) < 2e-7)

    correction = n_band(EF_op, T, c) / n_f_op
    mobility = 1.0 / (E_CHARGE_C * n_f_op * rho)

    EF = np.linspace(0.0, 0.25, 501)
    C = np.array([n_band(x, T, c) / n_fermi(x, T, c) for x in EF])
    ef2 = crossing(EF, C, 2.0)
    ef4 = crossing(EF, C, 4.0)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 160})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(EF, C, color="#176B87", linewidth=2.4,
            label=r"$C(E_F)=n_{band}/n_F$")
    ax.axhline(2, color="#999999", linewidth=0.9, linestyle="--")
    ax.axhline(4, color="#999999", linewidth=0.9, linestyle="--")
    ax.scatter([EF_op], [correction], color="#C0392B", s=45, zorder=3,
               label="Exact operating point")
    ax.scatter([ef2, ef4], [2, 4], color="#E67E22", s=30, zorder=3)
    ax.annotate(
        f"Operating point\n$E_F$={EF_op:.4f} eV, C={correction:.3f}",
        (EF_op, correction), xytext=(38, -52), textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#C0392B"},
    )
    ax.annotate(f"C=2 at {ef2:.3f} eV", (ef2, 2),
                xytext=(-51, 15), textcoords="offset points")
    ax.annotate(f"C=4 at {ef4:.3f} eV", (ef4, 4),
                xytext=(6, -19), textcoords="offset points")
    ax.set(
        xlabel=r"Fermi level $E_F$ (eV)",
        ylabel="Carrier-count correction factor C",
        title="L1 carrier-count correction at 300 K",
    )
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0.8, max(6.2, float(C.max()) * 1.04))
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(out / "L1_correction_factor.png", dpi=300)
    plt.close(fig)

    check("L1 correction monotonic", np.all(np.diff(C) > 0))
    print(f"L1 at {T:g} K, EF={EF_op:.9f} eV, rho={rho:.6f} ohm/square, "
          f"C={correction:.9f}, n_F={n_f_op:.9g} cm^-2, mobility={mobility:.6f} cm^2/Vs")
    print(f"L1 C=2 at {ef2:.6f} eV and C=4 at {ef4:.6f} eV")
    return {
        "correction_factor": correction,
        "n_fermi_op_cm2": n_f_op,
        "mobility_corrected_cm2_Vs": mobility,
    }


def normalize_channels(pairs):
    scale = np.array([np.max(np.abs(pairs[:, 0])), np.max(pairs[:, 1])])
    check("L2 channel normalisation", np.all(scale > 0))
    return pairs / scale


def run_l2(data: Path, out: Path):
    meta = read_json(data / "L2_meta.json")
    raw = pd.read_csv(data / "L2_data.csv")
    check("L2 input columns", raw.columns.tolist() == meta["columns"])
    check("L2 finite raw data", np.isfinite(raw.to_numpy()).all())
    check("L2 positive longitudinal readings", (raw.R_xx_raw > 0).all())
    indices = raw.measurement_index.drop_duplicates().to_numpy()
    check("L2 acquisition order", np.array_equal(indices, np.arange(56)))
    ordered = raw.sort_values(["measurement_index", "repeat"])
    check("L2 three distinct paired repeats at each index",
          np.array_equal(ordered.measurement_index, np.repeat(indices, 3))
          and np.array_equal(ordered.repeat, np.tile([1, 2, 3], len(indices))))
    repeats = ordered[["R_H_raw", "R_xx_raw"]].to_numpy().reshape(len(indices), 3, 2)
    means = repeats.mean(axis=1)
    crossings = np.flatnonzero(means[:-1, 0] * means[1:, 0] < 0)
    check("L2 Hall sign crossing", crossings.size > 0)
    observations = normalize_channels(means)
    check("L2 Hall signs preserved",
          np.array_equal(np.sign(observations[:, 0]), np.sign(means[:, 0])))

    kbt = KB_EV_K * meta["T_K"]
    energy = np.linspace(-2.5, 2.5, 4000)
    ef_grid = np.linspace(-8 * kbt, 9 * kbt, 400)
    delta = energy[None, :] - ef_grid[:, None]
    occupation = 1.0 / (np.exp(np.clip(delta / kbt, -60, 60)) + 1.0)
    gaps = np.r_[0, np.arange(0.5, 8.001, 0.25)]
    curves = []
    for gap in gaps:
        # The common DoS prefactor cancels under per-channel normalisation.
        dos = np.maximum(np.abs(energy) - gap * kbt / 2, 0)
        electrons = NA_CM2 * np.trapezoid(
            np.where(delta >= 0, dos * occupation, 0), energy, axis=1
        )
        holes = NA_CM2 * np.trapezoid(
            np.where(delta <= 0, dos * (1 - occupation), 0), energy, axis=1
        )
        total = holes + electrons
        response = np.column_stack(((holes - electrons) / total**2, 1 / total))
        curves.append(normalize_channels(response))
    curves = np.asarray(curves)
    check("L2 finite model library", curves.shape == (32, 400, 2) and np.isfinite(curves).all())

    def fit(points):
        difference = points[None, :, None, :] - curves[:, None, :, :]
        squared_distance = (difference**2).sum(axis=-1)
        nearest = squared_distance.argmin(axis=2)
        rms = np.sqrt(squared_distance.min(axis=2).mean(axis=1))
        best_gap = 1 + rms[1:].argmin()
        chosen = best_gap if rms[best_gap] <= 0.85 * rms[0] and gaps[best_gap] >= 1 else 0
        return chosen, nearest[chosen], rms, best_gap

    chosen, nearest, rms, best_gap = fit(observations)
    check("L2 trajectory within fit grid", np.all((nearest > 0) & (nearest < len(ef_grid) - 1)))
    trajectory = ef_grid[nearest]
    model = "cone" if chosen == 0 else "hard_gap"

    rng = np.random.default_rng(6028)
    trials = 200
    bootstrap_gaps = np.empty(trials)
    bootstrap_ef = np.empty((trials, len(indices)))
    for trial in range(trials):
        # Draw the two measured channels together from the same repeat.
        readings = repeats[indices, rng.integers(0, 3, size=len(indices)), :]
        selected, assignment, _, _ = fit(normalize_channels(readings))
        bootstrap_gaps[trial] = gaps[selected]
        bootstrap_ef[trial] = ef_grid[assignment]
    gap_lo, gap_hi = np.percentile(bootstrap_gaps, [16, 84])
    ef_lo, ef_hi = np.percentile(bootstrap_ef, [16, 84], axis=0)

    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(7.6, 5.4), layout="constrained")
        ax.plot(*curves[chosen].T, color="#007b87", lw=1.8,
                label=f"Selected {model}: RMS {rms[chosen]:.4f}")
        comparison = best_gap if chosen == 0 else 0
        comparison_name = f"Best hard gap: {gaps[best_gap]:g} kBT" if chosen == 0 else "Cone"
        ax.plot(*curves[comparison].T, color="gray", ls="--", lw=1.2,
                label=f"{comparison_name}; RMS {rms[comparison]:.4f}")
        points = ax.scatter(*observations.T, c=indices, cmap="viridis", s=26,
                            edgecolors="white", linewidths=0.35, zorder=3)
        fig.colorbar(points, ax=ax, label="Measurement index")
        ax.set(xlabel="Normalised Hall channel", ylabel="Normalised sheet-resistance channel",
               title=f"G01 L2 model comparison | selected: {model}")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
        ax.grid(alpha=0.2)
        fig.savefig(out / "model_figure.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
        ax.fill_between(indices, ef_lo, ef_hi, color="#007b87", alpha=0.22,
                        label="Bootstrap 16th–84th percentiles")
        ax.plot(indices, trajectory, "o-", ms=3, lw=1.3, color="#005b68",
                label="Fit to repeat means")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(xlabel="Measurement index", ylabel="Assigned Fermi level (eV)",
               title="G01 L2 Fermi-level trajectory")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.savefig(out / "trajectory_figure.png", dpi=300)
        plt.close(fig)

    print(f"L2 {model}, Eg/kBT={gaps[chosen]:g}, bootstrap p16–p84=[{gap_lo:g}, {gap_hi:g}], "
          f"hard-gap selection={np.mean(bootstrap_gaps > 0):.1%}")
    print(f"L2 cone RMS={rms[0]:.6f}, best hard-gap RMS={rms[best_gap]:.6f} "
          f"at {gaps[best_gap]:g} kBT, EF range=[{trajectory.min():.6f}, {trajectory.max():.6f}] eV")
    print("L2 Hall zero crossings:", [(int(i), int(i + 1)) for i in crossings])
    return {
        "model": model,
        "Eg_kT": float(gaps[chosen]),
        "Eg_uncertainty_kT": float((gap_hi - gap_lo) / 2),
        "EF_trajectory_eV": trajectory.tolist(),
    }


def build_smoothness_matrix(npix: int) -> np.ndarray:
    pixel_count = npix * npix
    rows = []
    for r in range(npix):
        for c in range(npix):
            i = r * npix + c
            if c + 1 < npix:
                v = np.zeros(pixel_count)
                v[i] = 1.0
                v[i + 1] = -1.0
                rows.append(v)
            if r + 1 < npix:
                v = np.zeros(pixel_count)
                v[i] = 1.0
                v[i + npix] = -1.0
                rows.append(v)
    return np.asarray(rows)


def reconstruction_operator(A: np.ndarray, L: np.ndarray, weight: float) -> np.ndarray:
    pixel_count = A.shape[1]
    P = np.eye(pixel_count) - np.ones((pixel_count, pixel_count)) / pixel_count  # zero-mean projector
    system = A.T @ A + weight * (L.T @ L) + 1e-9 * np.eye(pixel_count)
    return P @ np.linalg.solve(system, A.T)


def feature_points_from_map(m: np.ndarray, coords: np.ndarray):
    npix = int(np.sqrt(m.size))
    pos = int(m.argmax())
    neg = int(m.argmin())
    features = []
    for sign, idx in [("positive", pos), ("negative", neg)]:
        r, c = divmod(idx, npix)
        features.append({
            "sign": sign,
            "row": int(r),
            "column": int(c),
            "x_probe_pitch": float(coords[c]),
            "y_probe_pitch": float(coords[r]),
            "amplitude": float(m[idx]),
        })
    return features


def psf_fwhm(B: np.ndarray, A: np.ndarray, npix: int, rc: tuple[int, int], coords: np.ndarray):
    pixel_count = npix * npix
    r, c = rc
    impulse = np.full(pixel_count, -1.0 / pixel_count)
    impulse[r * npix + c] += 1.0
    psf = (B @ A @ impulse).reshape(npix, npix)

    X, Y = np.meshgrid(coords, coords)
    w = np.maximum(psf, 0.0)
    xc = coords[c]
    yc = coords[r]
    sigma = np.sqrt((w * ((X - xc) ** 2 + (Y - yc) ** 2)).sum() / w.sum() / 2.0)
    fwhm = float(2.355 * sigma)
    return psf, fwhm


def run_l3(data_dir: Path, out_dir: Path):
    threshold = THRESHOLD
    noise_trials = 500
    seed = 20260924
    meta = read_json(data_dir / "L3_meta.json")
    npix = int(meta["npix"])
    pixel_count = npix * npix

    A = np.load(data_dir / "L3_A.npy", allow_pickle=False)
    d = np.loadtxt(data_dir / "L3_d.csv", delimiter=",")
    if A.shape != (128, pixel_count):
        raise ValueError(f"Unexpected A shape {A.shape}; expected (128, {pixel_count})")
    if d.shape != (3, 128):
        raise ValueError(f"Unexpected d shape {d.shape}; expected (3, 128)")
    if not np.isfinite(A).all() or not np.isfinite(d).all():
        raise ValueError("Found non-finite entries in A or d")

    row_sum_max = float(np.abs(A.sum(axis=1)).max())
    if row_sum_max >= 1e-10:
        raise ValueError(f"Forward matrix row sums are too large: {row_sum_max}")

    L = build_smoothness_matrix(npix)
    eta0 = 0.01 * np.linalg.norm(A, 2) ** 2 / np.linalg.norm(L, 2) ** 2
    B = reconstruction_operator(A, L, eta0)

    dbar = d.mean(axis=0)
    m = B @ dbar
    coords = -3.0 + (np.arange(npix) + 0.5) * 6.0 / npix

    features = feature_points_from_map(m, coords)
    peak = float(np.max(np.abs(m)))
    map_mean = float(m.mean())

    # Point-spread tests.
    psf_center, fwhm_center = psf_fwhm(B, A, npix, (5, 5), coords)
    psf_off, fwhm_off = psf_fwhm(B, A, npix, (2, 7), coords)
    resolution = max(fwhm_center, fwhm_off)

    # Noise-only detection limit.
    sd = d.std(axis=0, ddof=1)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(A.shape[0], noise_trials)) * sd[:, None]
    noise_peaks = np.max(np.abs(B @ noise), axis=0)
    delta95 = float(np.percentile(noise_peaks, 95))

    # Pass/fail decision with confidence.
    if peak + delta95 < threshold:
        decision_text = (
            f"Clear pass because max|m| + δ95 = {(peak + delta95) * 100:.3f}% < {threshold * 100:.1f}%."
        )
    elif peak - delta95 > threshold:
        decision_text = (
            f"Clear fail because max|m| - δ95 = {(peak - delta95) * 100:.3f}% > {threshold * 100:.1f}%."
        )
    else:
        decision_text = (
            f"Borderline because the noise envelope overlaps the {threshold * 100:.1f}% threshold."
        )

    # Regularisation sensitivity.
    regularisation = []
    for mult in [0.01, 0.1, 1.0, 10.0, 100.0]:
        Bm = reconstruction_operator(A, L, eta0 * mult)
        mm = Bm @ dbar
        regularisation.append({
            "multiplier": mult,
            "peak_fraction": float(np.max(np.abs(mm))),
            "residual_norm": float(np.linalg.norm(A @ mm - dbar)),
            "roughness_norm": float(np.linalg.norm(L @ mm)),
        })

    decision_text += " The 95% noise envelope reflects repeat noise only."
    decision_text += " The 3.0% regular-data specification is provisional."

    # Figure 4. Row zero is at the lower edge of the map.
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    vmax = peak * 100.0
    im = ax.imshow(
        m.reshape(npix, npix) * 100.0,
        origin="lower",
        extent=(-3, 3, -3, 3),
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative conductivity deviation (%)")
    ax.set_xlabel("x (probe pitches)")
    ax.set_ylabel("y (probe pitches)")
    ax.set_title("G01 L3 reconstructed zero-mean conductivity map")
    for feat in features:
        x = feat["x_probe_pitch"]
        y = feat["y_probe_pitch"]
        amp_pct = feat["amplitude"] * 100.0
        ax.scatter([x], [y], s=280, facecolors="none", edgecolors="black", linewidths=2.2, zorder=3)
        label = f"{amp_pct:+.3f}%"
        xytext = (13, -27) if feat["sign"] == "negative" else (-60, 28)
        ax.annotate(label, xy=(x, y), xytext=xytext, textcoords="offset points",
                    bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none", "alpha": 0.82},
                    arrowprops={"arrowstyle": "-", "lw": 1.6, "color": "black"}, fontsize=11)
    fig.savefig(out_dir / "Figure_4_L3_map.png", dpi=300)
    plt.close(fig)

    # --- Figure 5: 4-panel validation figure ---
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # (a) residual-roughness trade-off
    ax = fig.add_subplot(gs[0, 0])
    res = np.array([x["residual_norm"] for x in regularisation])
    rough = np.array([x["roughness_norm"] for x in regularisation])
    mults = [x["multiplier"] for x in regularisation]
    ax.loglog(res, rough, marker="o", linewidth=2)
    for r, q, mult in zip(res, rough, mults):
        label = rf"{mult:g}$\eta_0$"
        offset = (-6, 8) if mult > 1 else (5, -15)
        ax.annotate(label, (r, q), textcoords="offset points", xytext=offset,
                    ha="right" if mult > 1 else "left", fontsize=9)
    # highlight reference eta0
    ref = mults.index(1.0)
    ax.scatter([res[ref]], [rough[ref]], s=260, facecolors="none", edgecolors="#C05A2B", linewidths=2.2,
               label=r"Reference $\eta_0$")
    ax.set_xlabel(r"Residual norm $\|Am-\bar d\|_2$")
    ax.set_ylabel(r"Roughness norm $\|Lm\|_2$")
    ax.set_title("(a) Regularisation trade-off")
    ax.legend(frameon=False, loc="lower left")

    # (b) peak vs eta
    ax = fig.add_subplot(gs[0, 1])
    peak_pct = 100.0 * np.array([x["peak_fraction"] for x in regularisation])
    ax.semilogx(mults, peak_pct, marker="o", linewidth=2, label="Reconstructed maximum")
    ax.axhline(threshold * 100.0, linestyle="--", linewidth=2, label="3% specification")
    ax.scatter([1.0], [peak_pct[ref]], s=260, facecolors="none", edgecolors="black", linewidths=2.2)
    ax.set_xlabel(r"Regularisation multiplier $\eta / \eta_0$")
    ax.set_ylabel("Maximum absolute deviation (%)")
    ax.set_title("(b) Peak amplitude sensitivity")
    ax.legend(frameon=False, loc="center left")

    # (c) PSF tests
    sub = gs[1, 0].subgridspec(1, 2, wspace=0.02)
    ax1 = fig.add_subplot(sub[0, 0])
    vmax_psf = max(np.abs(psf_center).max(), np.abs(psf_off).max())
    im1 = ax1.imshow(psf_center, origin="lower", extent=(-3, 3, -3, 3), cmap="RdBu_r",
                     vmin=-vmax_psf, vmax=vmax_psf, interpolation="nearest")
    ax1.scatter([coords[5]], [coords[5]], marker="+", s=180, c="black", linewidths=2.2)
    ax1.set_title(f"(c1) Impulse (5, 5)\nFWHM = {fwhm_center:.3f} pitches")
    ax1.set_xlabel("x (probe pitches)")
    ax1.set_ylabel("y (probe pitches)")
    ax2 = fig.add_subplot(sub[0, 1])
    im2 = ax2.imshow(psf_off, origin="lower", extent=(-3, 3, -3, 3), cmap="RdBu_r",
                     vmin=-vmax_psf, vmax=vmax_psf, interpolation="nearest")
    ax2.scatter([coords[7]], [coords[2]], marker="+", s=180, c="black", linewidths=2.2)
    ax2.set_title(f"(c2) Impulse (2, 7)\nFWHM = {fwhm_off:.3f} pitches")
    ax2.set_xlabel("x (probe pitches)")
    ax2.set_ylabel("y (probe pitches)")
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.88)
    cbar.set_label("Reconstructed impulse amplitude")

    # (d) noise histogram
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(noise_peaks * 100.0, bins=22, edgecolor="black")
    ax.axvline(delta95 * 100.0, linestyle="--", linewidth=2,
               label=rf"$\delta_{{95}}$ = {delta95*100:.4f}%")
    ax.set_xlabel(r"Noise-only $\max |m|$ (%)")
    ax.set_ylabel("Count")
    ax.set_title("(d) Detection-limit simulation (500 trials)")
    ax.legend(frameon=False)

    fig.suptitle("G01 L3 validation and decision evidence", fontsize=16)
    fig.savefig(out_dir / "Figure_5_L3_validation.png", dpi=300)
    plt.close(fig)

    print(f"L3 row-sum maximum={row_sum_max:.3g}, map mean={map_mean:.3g}, eta0={eta0:.9g}")
    print(f"L3 FWHM at (5, 5)={fwhm_center:.6f}, at (2, 7)={fwhm_off:.6f} pitches")
    print(f"L3 peak={peak:.9g}, detection limit={delta95:.9g}. {decision_text}")
    return {
        "map": m.tolist(),
        "features": features,
        "resolution_probe_pitch": resolution,
        "detection_limit_fraction": delta95,
        "pass": bool(peak <= threshold),
        "confidence": decision_text,
    }


def validate(result, template):
    check("Course schema top-level keys", set(result) == set(template))
    for phase in ("L1", "L2", "L3"):
        check(f"Course schema {phase} keys", set(result[phase]) == set(template[phase]))
    for field in ("schema_version", "group", "dataset_set"):
        check(f"Course schema {field}", result[field] == template[field])

    def finite_number(value):
        return type(value) in (int, float) and math.isfinite(value)

    l1, l2, l3 = (result[phase] for phase in ("L1", "L2", "L3"))
    check("L1 positive finite outputs", all(finite_number(v) and v > 0 for v in l1.values()))
    check("L2 model", l2["model"] in ("cone", "hard_gap"))
    check("L2 band gap and uncertainty",
          all(finite_number(l2[key]) and l2[key] >= 0 for key in ("Eg_kT", "Eg_uncertainty_kT")))
    trajectory = l2["EF_trajectory_eV"]
    check("L2 56 finite Fermi levels", len(trajectory) == 56 and all(map(finite_number, trajectory)))

    pixels = l3["map"]
    check("L3 100 finite zero-mean pixels",
          len(pixels) == 100 and all(map(finite_number, pixels)) and abs(np.mean(pixels)) < 1e-12)
    check("L3 positive resolution and detection limit", all(
        finite_number(l3[key]) and l3[key] > 0
        for key in ("resolution_probe_pitch", "detection_limit_fraction")
    ))
    check("L3 one feature of each sign", [f["sign"] for f in l3["features"]] == ["positive", "negative"])
    for feature, extremum in zip(l3["features"], (max(pixels), min(pixels))):
        check("L3 feature keys", set(feature) == {
            "sign", "row", "column", "x_probe_pitch", "y_probe_pitch", "amplitude",
        })
        row, column = feature["row"], feature["column"]
        check("L3 feature indices",
              type(row) is int and type(column) is int and 0 <= row < 10 and 0 <= column < 10)
        check("L3 feature amplitude", feature["amplitude"] == pixels[10 * row + column] == extremum)
        check("L3 feature coordinates",
              np.isclose(feature["x_probe_pitch"], -2.7 + 0.6 * column)
              and np.isclose(feature["y_probe_pitch"], -2.7 + 0.6 * row))
    check("L3 decision", type(l3["pass"]) is bool
          and l3["pass"] == (max(map(abs, pixels)) <= THRESHOLD))
    check("L3 confidence statement", isinstance(l3["confidence"], str) and bool(l3["confidence"]))


def main():
    local_data = ROOT / "data"
    default_data = ROOT if (ROOT / "manifest.json").is_file() else local_data
    if (local_data / "manifest.json").is_file():
        default_data = local_data
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data, help="directory with G01 regular data")
    parser.add_argument("--output", type=Path, default=ROOT, help="directory for results.json and figures")
    args = parser.parse_args()

    if not (args.data / "manifest.json").is_file():
        parser.error("manifest.json not found. Place the original G01 regular data/ beside reproduce.py.")
    verify_inputs(args.data)
    template = read_json(args.data / "results_template.json")
    commit = pipeline_commit()
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".reproduce-", dir=args.output) as temp:
        staging = Path(temp)
        result = {
            "schema_version": "QXU6028-results-v1",
            "group": "G01",
            "dataset_set": "regular",
            "pipeline_commit": commit,
            "L1": run_l1(args.data, staging),
            "L2": run_l2(args.data, staging),
            "L3": run_l3(args.data, staging),
        }
        validate(result, template)
        check("All five figures generated", all((staging / name).stat().st_size > 0 for name in FIGURES))
        results_file = staging / "results.json"
        results_file.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        check("JSON round-trip", read_json(results_file) == result)
        for name in (*FIGURES, "results.json"):
            (staging / name).replace(args.output / name)

    print(f"Validated G01 regular results and five report figures in {args.output}")


if __name__ == "__main__":
    main()

# Detached release metadata.
# Commit: b07056ec051bd2805d5e2fdbec00d7aa155059aa
# SHA-256: 605f87fa5c01a8ff224c035f40243a5f82bda5def3003f47397fc2f3530e7854
