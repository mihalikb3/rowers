#!/usr/bin/env python3
"""
Two modes of operation:

1. SWEEP MODE (default)
   Sweeps a gradient parameter (linear or geometric) and plots how the
   natural frequency shifts.  An ensemble of quiet runs per gradient value
   provides error bars on the peak frequency.

   Single mode, with error bars:
     python check_natural_frequency.py --mode linear --d_avg 10 --runs 5 --cores 8

   Both modes on one comparison figure:
     python check_natural_frequency.py --both --runs 5 --cores 8

2. CONFIG MODE  (--bo_json / --metro_json)
   Loads the best configuration from a BO or Metropolis JSON, runs a quiet
   (F0=0) simulation with those exact bead distances, and computes the
   natural frequency from the PSD.  The result is compared against:
     • f_drive  = 1 / T_signal
     • f_observed = (crossings_per_cycle) × f_drive
       where crossings_per_cycle comes from the forward runs in the JSON
       (best eval's asymmetry entry — requires a verify_summary.json in the
       same directory, or falls back to the JSON crossings field if present).

   python check_natural_frequency.py --bo_json path/to/bo_history.json
   python check_natural_frequency.py --metro_json path/to/metropolis_history.json
"""

import os
import json
import hashlib
import shutil
import subprocess
import argparse
import warnings
from multiprocessing import Pool
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from bead_analysis import BeadDataAnalyzer

warnings.filterwarnings("ignore")

THERMALISATION_TIME = 10.0   # quiet sims need a longer warm-up (no driving)

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
})


def _compute_distances(mode, d_avg, g, num_gaps):
    if mode == "linear":
        return [max(3.1, d_avg + (i - (num_gaps - 1) / 2.0) * g)
                for i in range(num_gaps)]
    else:  # geometric
        L   = num_gaps * d_avg
        d0  = L * (g - 1.0) / (g**num_gaps - 1.0) if abs(g - 1.0) > 1e-9 else d_avg
        return [max(3.1, d0 * g**i) for i in range(num_gaps)]


def _sweep_worker(task):

    exe, template, mode, d_avg, g, total_time, run_idx, return_psd = task

    with open(template) as fh:
        params = fh.readline().strip().split()

    num_gaps  = int(params[0]) - 1
    distances = _compute_distances(mode, d_avg, g, num_gaps)

    params        = list(params)
    params[5]     = str(total_time)
    params[13]    = "50"     # write_sampling
    params[19]    = "0.0"    # F0 = 0 (quiet)
    for i in range(num_gaps):
        params[22 + i] = f"{distances[i]:.6f}"

    label      = f"quiet_{mode}_{g:.4f}_r{run_idx}".replace(".", "p")
    h          = hashlib.md5(label.encode()).hexdigest()[:8]
    run_id     = f"{label}_{h}"
    input_file = f"{run_id}.input"

    try:
        with open(input_file, "w") as fh:
            fh.write(" ".join(params) + "\n")

        subprocess.run(f"{exe} -C -R {input_file}", shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        dat_files = [f for f in os.listdir(run_id) if f.endswith(".dat")]
        if not dat_files:
            return (mode, g, run_idx, None, None, None)

        analyzer = BeadDataAnalyzer()
        analyzer.read_data_file(os.path.join(run_id, dat_files[0]))

        mask = analyzer.time_series > THERMALISATION_TIME
        t    = analyzer.time_series[mask]
        pos  = analyzer.positions[mask]
        fs   = 1.0 / (t[1] - t[0])

        all_psds = []
        for i in range(analyzer.num_beads):
            x = pos[:, i, 0]
            freqs, psd = welch(x - np.mean(x), fs=fs, nperseg=4096)
            all_psds.append(psd)

        avg_psd   = np.mean(all_psds, axis=0)
        mask_rng  = (freqs > 0.1) & (freqs < 20.0)
        peak_f    = freqs[mask_rng][np.argmax(avg_psd[mask_rng])]

        freqs_out = freqs.tolist() if return_psd else None
        psd_out   = avg_psd.tolist() if return_psd else None

        shutil.rmtree(run_id)
        os.remove(input_file)
        return (mode, g, run_idx, float(peak_f), freqs_out, psd_out)

    except Exception as exc:
        for path in [run_id, input_file]:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass
        print(f"[Worker error] mode={mode}, g={g:.4f}, run={run_idx}: {exc}")
        return (mode, g, run_idx, None, None, None)


class NaturalFrequencyChecker:
    def __init__(self, executable="./Beads",
                 template="BeadsV_6Beads_0Signal_UPDlambda2.input",
                 mode="linear"):
        self.executable = executable
        self.template   = template
        self.mode       = mode
        if not os.path.exists(self.executable):
            if os.path.exists("./modern/modern_Beads"):
                self.executable = "./modern/modern_Beads"
            else:
                raise FileNotFoundError(f"Executable {self.executable} not found.")

    def _read_template(self):
        with open(self.template) as f:
            return f.readline().strip().split()

    # quiet simulation

    def run_quiet_sim_from_distances(self, distances, total_time=300):
        """Config-mode: use explicit distance list."""
        params = self._read_template()
        return self._run_quiet_from_distances(distances, params, total_time,
                                              label="quiet_config")

    def _run_quiet_from_distances(self, distances, params, total_time, label):
        params     = list(params)
        params[5]  = str(total_time)
        params[13] = "50"
        params[19] = "0.0"

        num_gaps = int(params[0]) - 1
        for i in range(num_gaps):
            params[22 + i] = f"{distances[i]:.6f}"

        h          = hashlib.md5(label.encode()).hexdigest()[:8]
        run_id     = f"{label}_{h}"
        input_file = f"{run_id}.input"

        with open(input_file, "w") as f:
            f.write(" ".join(params) + "\n")

        subprocess.run(f"{self.executable} -C -R {input_file}",
                       shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        dat_files = [f for f in os.listdir(run_id) if f.endswith(".dat")]
        if not dat_files:
            raise FileNotFoundError("Simulation failed to produce .dat file.")

        analyzer = BeadDataAnalyzer()
        analyzer.read_data_file(os.path.join(run_id, dat_files[0]))

        shutil.rmtree(run_id)
        os.remove(input_file)
        return analyzer


    def get_psd_data(self, analyzer):
        """Return (freqs, mean_psd, per_bead_psds, peak_freq)."""
        mask = analyzer.time_series > THERMALISATION_TIME
        t    = analyzer.time_series[mask]
        pos  = analyzer.positions[mask]
        fs   = 1.0 / (t[1] - t[0])

        all_psds = []
        for i in range(analyzer.num_beads):
            x = pos[:, i, 0]
            freqs, psd = welch(x - np.mean(x), fs=fs, nperseg=4096)
            all_psds.append(psd)

        avg_psd  = np.mean(all_psds, axis=0)
        mask_rng = (freqs > 0.1) & (freqs < 20.0)
        peak_f   = freqs[mask_rng][np.argmax(avg_psd[mask_rng])]

        return freqs, avg_psd, all_psds, peak_f

    def get_per_bead_peaks(self, analyzer):
        """Return list of per-bead peak frequencies."""
        mask = analyzer.time_series > THERMALISATION_TIME
        t    = analyzer.time_series[mask]
        pos  = analyzer.positions[mask]
        fs   = 1.0 / (t[1] - t[0])

        peaks = []
        for i in range(analyzer.num_beads):
            x = pos[:, i, 0]
            freqs, psd = welch(x - np.mean(x), fs=fs, nperseg=4096)
            rng = (freqs > 0.1) & (freqs < 20.0)
            peaks.append(freqs[rng][np.argmax(psd[rng])])
        return peaks


def _run_ensemble(exe, template, mode, d_avg, grad_range, total_time,
                  n_runs, n_cores):

    mid_idx   = len(grad_range) // 2
    mid_g     = grad_range[mid_idx]

    tasks = []
    for g in grad_range:
        for r in range(n_runs):
            want_psd = (g == mid_g and r == 0)
            tasks.append((exe, template, mode, d_avg, g, total_time, r, want_psd))

    print(f"  [{mode}] Running {len(tasks)} simulations "
          f"({len(grad_range)} points × {n_runs} runs) on {n_cores} cores …")

    with Pool(n_cores) as pool:
        results = pool.map(_sweep_worker, tasks)

    peaks_by_g   = defaultdict(list)
    mid_freqs    = None
    mid_psd      = None

    for (m, g, r, pk, freqs, psd) in results:
        if pk is not None:
            peaks_by_g[g].append(pk)
        if freqs is not None:
            mid_freqs = np.array(freqs)
            mid_psd   = np.array(psd)

    peaks_mean = np.array([np.mean(peaks_by_g[g]) if peaks_by_g[g] else np.nan
                           for g in grad_range])
    peaks_sem  = np.array([
        (np.std(peaks_by_g[g], ddof=1) / np.sqrt(len(peaks_by_g[g]))
         if len(peaks_by_g[g]) > 1 else 0.0)
        for g in grad_range
    ])

    return peaks_mean, peaks_sem, mid_freqs, mid_psd


def _add_freq_panel(ax, grad_range, means, sems, mode, n_runs, d_avg):
    color     = "#1f77b4" if mode == "linear" else "#d62728"
    ref_val   = 0.0 if mode == "linear" else 1.0
    grad_label = (r"Distance gradient $\Delta d$ (µm / bead)"
                  if mode == "linear"
                  else r"Geometric ratio $g = d_{i+1}/d_i$")
    title = (f"Linear chain"
             if mode == "linear"
             else f"Geometric chain")

    ax.plot(grad_range, means, "o-", color=color, linewidth=2, markersize=5,
            label="Natural frequency")
    ax.axvline(ref_val, color="grey", linestyle=":", linewidth=1.2, alpha=0.8,
               label="Uniform spacing")

    ax.set_xlabel(grad_label)
    ax.set_ylabel("Dominant frequency (Hz)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def plot_single_mode(grad_range, means, sems, mode, n_runs, d_avg, filename):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    _add_freq_panel(ax, grad_range, means, sems, mode, n_runs, d_avg)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[Plot] → {filename}")


def plot_freq_single(grad_range, means, sems, mode, n_runs, d_avg, filename):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    _add_freq_panel(ax, grad_range, means, sems, mode, n_runs, d_avg)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[Plot] → {filename}")


def plot_spectrum_single(freqs, psd, peak, mode, mid_g, filename):
    g_label = (fr"$\Delta d = {mid_g:.2f}$ µm/bead"
               if mode == "linear"
               else fr"$g = {mid_g:.3f}$")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.semilogy(freqs, psd, color="black", linewidth=1.5, alpha=0.8)
    ax.axvline(peak, color="red", linestyle="--", linewidth=1.5,
               label=f"Peak: {peak:.3f} Hz")
    ax.set_xlim(0.1, 10)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (position variance / Hz)")
    ax.set_title(f"{mode.capitalize()} chain ({g_label})",
                 fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[Plot] → {filename}")


def plot_psd_sample(freqs, psd, peak_f, d_grad, mode, filename):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.semilogy(freqs, psd, color="black", linewidth=1.5, alpha=0.8)
    ax.axvline(peak_f, color="red", linestyle="--",
               label=f"Peak: {peak_f:.3f} Hz")
    grad_label = r"$\Delta d$" if mode == "linear" else r"$g$"
    ax.set_title(f"Thermal PSD ({mode} mode, {grad_label}={d_grad:.3f})",
                 fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (Position variance / Hz)")
    ax.set_xlim(0.1, 10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[Plot] → {filename}")


def plot_config_psd(freqs, avg_psd, per_bead_psds, per_bead_peaks,
                    f_natural, f_drive, f_observed,
                    distances, T_signal, output_dir):
    """2-panel figure for config mode."""
    N      = len(per_bead_psds)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, N))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax = axes[0]
    ax.semilogy(freqs, avg_psd, color="black", linewidth=1.8,
                label="Mean PSD (all beads)")
    ax.axvline(f_natural, color="red",    linestyle="--", linewidth=1.5,
               label=f"Natural frequency: {f_natural:.3f} Hz  "
                     f"(T = {1/f_natural:.3f} s)")
    ax.axvline(f_drive,   color="blue",   linestyle=":",  linewidth=1.5,
               label=f"Drive frequency: {f_drive:.3f} Hz  "
                     f"(T = {T_signal:.3f} s)")
    if f_observed is not None:
        ax.axvline(f_observed, color="green", linestyle="-.", linewidth=1.5,
                   label=f"Observed fwd freq: {f_observed:.3f} Hz  "
                         f"(ratio = {f_observed/f_drive:.2f}×)")
    ax.set_xlim(0.1, 20)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (position variance / Hz)")
    ax.set_title("Thermal PSD — quiet simulation\n(F₀ = 0, best-config distances)",
                 fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    for i, (psd_i, pk) in enumerate(zip(per_bead_psds, per_bead_peaks)):
        ax.semilogy(freqs, psd_i, color=colors[i], linewidth=1.3, alpha=0.85,
                    label=f"Bead {i+1}  (peak {pk:.2f} Hz)")
    ax.axvline(f_natural, color="red",  linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axvline(f_drive,   color="blue", linestyle=":",  linewidth=1.2, alpha=0.6)
    ax.set_xlim(0.1, 20)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (position variance / Hz)")
    ax.set_title("Per-bead thermal PSD", fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    path = os.path.join(output_dir, "natural_frequency.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Plot] → {path}")


def run_config_mode(args):
    if args.bo_json:
        json_path = args.bo_json
        with open(json_path) as f:
            h = json.load(f)
        best_i    = int(np.argmax(h["asymmetry"]))
        distances = h["distances"][best_i]
        T_signal  = h["T_signal"][best_i]
        F0        = h["F0"][best_i]
        best_asym = h["asymmetry"][best_i]
        source_dir = os.path.dirname(json_path)
        label      = "BO"
    else:
        json_path = args.metro_json
        with open(json_path) as f:
            h = json.load(f)
        best_i    = int(np.argmax(h["asymmetry"]))
        distances = h["distances"][best_i]
        T_signal  = h["T_signal"][best_i]
        F0        = h["F0"][best_i]
        best_asym = h["asymmetry"][best_i]
        source_dir = os.path.dirname(json_path)
        label      = "Metropolis"

    f_drive = 1.0 / T_signal

    print(f"\n{'='*60}")
    print(f"Natural frequency check — {label} best config")
    print(f"  Distances (µm) : {[round(d, 3) for d in distances]}")
    print(f"  T_signal       : {T_signal:.4f} s  →  f_drive = {f_drive:.4f} Hz")
    print(f"  F0             : {F0:.4f}")
    print(f"  Best ΔC        : {best_asym:.4f}")
    print(f"{'='*60}\n")

    f_observed = None
    verify_json = os.path.join(source_dir, "verify_summary.json")
    if os.path.exists(verify_json):
        with open(verify_json) as f:
            vs = json.load(f)
        fwd_cross = vs.get("fwd_crossings_per_cycle", None)
        if fwd_cross:
            fwd_cross_per_cycle = fwd_cross[-1]
            f_observed = fwd_cross_per_cycle * f_drive
            print(f"[Verify] Loaded crossing data from {verify_json}")
            print(f"  Fwd crossings/cycle (last bead): {fwd_cross_per_cycle:.3f}")
            print(f"  → f_observed = {f_observed:.4f} Hz  "
                  f"(ratio to f_drive: {f_observed/f_drive:.2f}×)\n")
    else:
        print(f"[Verify] No verify_summary.json found in {source_dir}.")
        print( "         Run verify_best_config.py first to add observed frequency.\n")

    checker  = NaturalFrequencyChecker(executable=args.exe, template=args.template)
    print(f"[Sim] Running quiet simulation (F₀=0, {args.time} s)…")
    analyzer = checker.run_quiet_sim_from_distances(distances, total_time=args.time)

    freqs, avg_psd, per_bead_psds, f_natural = checker.get_psd_data(analyzer)
    per_bead_peaks = checker.get_per_bead_peaks(analyzer)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Natural frequency (PSD peak) : {f_natural:.4f} Hz  "
          f"(period = {1/f_natural:.4f} s)")
    print(f"  Drive frequency              : {f_drive:.4f} Hz  "
          f"(period = {T_signal:.4f} s)")
    print(f"  Ratio f_natural / f_drive    : {f_natural/f_drive:.3f}")
    if f_observed is not None:
        print(f"  Observed fwd frequency       : {f_observed:.4f} Hz  "
              f"(ratio = {f_observed/f_drive:.2f}× f_drive)")
        print(f"  f_observed vs f_natural      : {f_observed:.4f} vs "
              f"{f_natural:.4f} Hz  (diff = {abs(f_observed-f_natural):.4f} Hz)")
    print(f"\n  Per-bead natural frequencies:")
    for i, pk in enumerate(per_bead_peaks):
        print(f"    Bead {i+1}: {pk:.4f} Hz")
    print(f"{'='*60}\n")

    output_dir = source_dir if source_dir else "."
    plot_config_psd(freqs, avg_psd, per_bead_psds, per_bead_peaks,
                    f_natural, f_drive, f_observed,
                    distances, T_signal, output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check natural frequency via quiet (F0=0) simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--bo_json",    type=str, default=None,
                     help="Path to bo_history.json — uses the best configuration.")
    src.add_argument("--metro_json", type=str, default=None,
                     help="Path to metropolis_history.json — uses the best configuration.")

    # wweep mode: single-mode parameters
    parser.add_argument("--mode",     type=str, choices=["linear", "geometric"],
                        default="linear",
                        help="Parameterisation when not using --both.")
    parser.add_argument("--d_avg",    type=float, default=10.0,
                        help="Mean inter-bead spacing (µm).")
    parser.add_argument("--grad_min", type=float, default=None,
                        help="Minimum gradient value (default: −2 linear / 1.0 geometric).")
    parser.add_argument("--grad_max", type=float, default=None,
                        help="Maximum gradient value (default: +2 linear / 1.5 geometric).")
    parser.add_argument("--n_grad",   type=int,   default=11,
                        help="Number of gradient steps.")

    # wweep mode: both-modes parameters
    parser.add_argument("--both",     action="store_true",
                        help="Run both linear and geometric sweeps and produce comparison plots.")
    parser.add_argument("--lin_min",  type=float, default=-2.0,
                        help="Linear mode: minimum d_grad (µm/bead).")
    parser.add_argument("--lin_max",  type=float, default=2.0,
                        help="Linear mode: maximum d_grad (µm/bead).")
    parser.add_argument("--geo_min",  type=float, default=1.0,
                        help="Geometric mode: minimum g.")
    parser.add_argument("--geo_max",  type=float, default=1.5,
                        help="Geometric mode: maximum g.")

    # ensemble
    parser.add_argument("--runs",  type=int, default=5,
                        help="Number of independent quiet runs per gradient value (for error bars).")
    parser.add_argument("--cores", type=int, default=4,
                        help="Parallel simulation processes.")

    # shared
    parser.add_argument("--time",     type=int, default=300,
                        help="Quiet simulation duration (s). Longer → better frequency resolution.")
    parser.add_argument("--exe",      type=str, default="./modern/modern_Beads")
    parser.add_argument("--template", type=str,
                        default="BeadsV_6Beads_0Signal_UPDlambda2.input")

    args = parser.parse_args()

    # config mode
    if args.bo_json or args.metro_json:
        run_config_mode(args)

    # sweep mode
    elif args.both:
        lin_grads = np.linspace(args.lin_min, args.lin_max, args.n_grad)
        geo_grads = np.linspace(args.geo_min, args.geo_max, args.n_grad)

        print(f"=== Natural-frequency comparison sweep ===")
        print(f"  d_avg = {args.d_avg} µm,  {args.runs} runs/point,  "
              f"{args.time} s each,  {args.cores} cores\n")

        lin_means, lin_sems, lin_freqs, lin_psd = _run_ensemble(
            args.exe, args.template, "linear",
            args.d_avg, lin_grads, args.time, args.runs, args.cores)

        geo_means, geo_sems, geo_freqs, geo_psd = _run_ensemble(
            args.exe, args.template, "geometric",
            args.d_avg, geo_grads, args.time, args.runs, args.cores)

        lin_mid_g = lin_grads[len(lin_grads) // 2]
        geo_mid_g = geo_grads[len(geo_grads) // 2]

        def _peak_from_psd(freqs, psd):
            if freqs is None:
                return np.nan
            freqs, psd = np.array(freqs), np.array(psd)
            rng = (freqs > 0.1) & (freqs < 20.0)
            return float(freqs[rng][np.argmax(psd[rng])])

        lin_peak = _peak_from_psd(lin_freqs, lin_psd)
        geo_peak = _peak_from_psd(geo_freqs, geo_psd)

        # summary
        print(f"\nLinear   midpoint (Δd={lin_mid_g:.2f}):  "
              f"f = {lin_means[len(lin_grads)//2]:.3f} ± "
              f"{lin_sems[len(lin_grads)//2]:.3f} Hz")
        print(f"Geometric midpoint (g={geo_mid_g:.3f}):  "
              f"f = {geo_means[len(geo_grads)//2]:.3f} ± "
              f"{geo_sems[len(geo_grads)//2]:.3f} Hz")

        plot_freq_single(lin_grads, lin_means, lin_sems,
                         "linear", args.runs, args.d_avg,
                         "natural_frequency_linear.png")
        plot_freq_single(geo_grads, geo_means, geo_sems,
                         "geometric", args.runs, args.d_avg,
                         "natural_frequency_geometric.png")

        if lin_freqs is not None:
            plot_spectrum_single(np.array(lin_freqs), np.array(lin_psd),
                                 lin_peak, "linear", lin_mid_g,
                                 "natural_spectrum_linear.png")
        if geo_freqs is not None:
            plot_spectrum_single(np.array(geo_freqs), np.array(geo_psd),
                                 geo_peak, "geometric", geo_mid_g,
                                 "natural_spectrum_geometric.png")
        if lin_freqs is None or geo_freqs is None:
            print("[Warning] PSD data for one or both spectrum plots was not "
                  "captured; skipping that figure.")

        print("\n[Done]  natural_frequency_linear.png  "
              "natural_frequency_geometric.png  "
              "natural_spectrum_linear.png  "
              "natural_spectrum_geometric.png")

    else:
        if args.grad_min is None:
            args.grad_min = -2.0 if args.mode == "linear" else 1.0
        if args.grad_max is None:
            args.grad_max =  2.0 if args.mode == "linear" else 1.5

        grad_range = np.linspace(args.grad_min, args.grad_max, args.n_grad)
        print(f"=== Natural frequency sweep: {args.mode} mode ===")
        print(f"  d_avg = {args.d_avg} µm,  {args.runs} runs/point,  "
              f"{args.time} s each,  {args.cores} cores\n")

        means, sems, mid_freqs, mid_psd = _run_ensemble(
            args.exe, args.template, args.mode,
            args.d_avg, grad_range, args.time, args.runs, args.cores)

        mid_idx  = len(grad_range) // 2
        mid_g    = grad_range[mid_idx]
        mid_peak = float(means[mid_idx]) if not np.isnan(means[mid_idx]) else 0.0
        if mid_freqs is not None:
            mf = np.array(mid_freqs); mp = np.array(mid_psd)
            rng = (mf > 0.1) & (mf < 20.0)
            mid_peak = float(mf[rng][np.argmax(mp[rng])])

        print(f"\nSummary  ({args.mode}):")
        for g, m, s in zip(grad_range, means, sems):
            print(f"  {args.mode[:3]}={g:.3f}  f = {m:.3f} ± {s:.3f} Hz")

        plot_single_mode(grad_range, means, sems, args.mode,
                         args.runs, args.d_avg,
                         "resonance_frequency_shift.png")

        if mid_freqs is not None:
            plot_psd_sample(np.array(mid_freqs), np.array(mid_psd),
                            mid_peak, mid_g, args.mode,
                            "resonance_spectrum_midpoint.png")
            print("[Done]  resonance_frequency_shift.png  "
                  "resonance_spectrum_midpoint.png")
        else:
            print("[Done]  resonance_frequency_shift.png  (spectrum unavailable)")
