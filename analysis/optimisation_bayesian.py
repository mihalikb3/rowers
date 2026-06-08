#!/usr/bin/env python3


import os
import hashlib
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import json
import shutil
import multiprocessing
import datetime
import warnings

from scipy.stats import norm as scipy_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from bead_analysis import BeadDataAnalyzer, MultiBeadCoherenceAnalyzer

warnings.filterwarnings("ignore")

MIN_DISTANCE = 3.1   # hard lower bound between adjacent beads

# json encoder
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


class BeadBayesianoptimiser:
    def __init__(self, executable="./modern/modern_Beads",
                 template_input="BeadsV_6Beads_0Signal_UPDlambda2.input"):
        self.executable     = executable
        self.template_input = template_input
        if not os.path.exists(self.executable):
            if os.path.exists("./Beads"):
                self.executable = "./Beads"
            else:
                raise FileNotFoundError(f"Executable '{self.executable}' not found.")
        self._read_template()


    def _read_template(self):
        with open(self.template_input) as fh:
            self._template_params = fh.readline().strip().split()
        self.N               = int(self._template_params[0])
        self.num_inner       = self.N - 2
        self.default_lambda  = float(self._template_params[21])
        num_gaps             = self.N - 1
        template_dists       = [float(self._template_params[22 + i]) for i in range(num_gaps)]
        self.chain_length    = sum(template_dists)
        print(f"[Template] N={self.N}, chain_length={self.chain_length:.3f} µm")

    def _positions_to_distances(self, inner_positions):
        all_pos = [0.0] + list(inner_positions) + [self.chain_length]
        return [all_pos[i + 1] - all_pos[i] for i in range(self.N - 1)]

    def _is_valid(self, inner_positions):
        return all(d >= MIN_DISTANCE for d in self._positions_to_distances(inner_positions))

    def _build_params(self, inner_positions, T_signal, F0, total_time, reverse=False):
        params     = list(self._template_params[:22])
        params[5]  = str(total_time)
        params[13] = "50"
        params[19] = f"{F0:.6f}"
        params[20] = f"{T_signal:.6f}"
        distances  = self._positions_to_distances(inner_positions)
        if reverse:
            distances = distances[::-1]
        params += [f"{d:.6f}" for d in distances]
        params += [f"{self.default_lambda:.6f}"] * self.N
        return params


    def run_single_sim(self, inner_positions, T_signal, F0,
                       reverse=False, seed_offset=0, total_time=100):
        run_id = input_file = "unknown"
        try:
            params = self._build_params(inner_positions, T_signal, F0, total_time, reverse)
            state_str  = (f"{seed_offset}_{','.join(f'{p:.4f}' for p in inner_positions)}"
                          f"_T{T_signal:.6f}_F{F0:.6f}_{'r' if reverse else 'f'}")
            h          = hashlib.md5(state_str.encode()).hexdigest()[:10]
            run_id     = f"bo_{os.getpid()}_{h}"
            input_file = f"{run_id}.input"

            with open(input_file, "w") as fh:
                fh.write(" ".join(params) + "\n")

            subprocess.run(f"{self.executable} -C -R {input_file}",
                           shell=True, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            output_dir = run_id
            if not os.path.exists(output_dir):
                return np.nan
            dat_files = [f for f in os.listdir(output_dir) if f.endswith(".dat")]
            if not dat_files:
                shutil.rmtree(output_dir); os.remove(input_file)
                return np.nan

            analyzer = BeadDataAnalyzer()
            analyzer.read_data_file(os.path.join(output_dir, dat_files[0]))
            if analyzer.positions is None or len(analyzer.positions) < 10:
                shutil.rmtree(output_dir); os.remove(input_file)
                return np.nan

            mask = analyzer.time_series >= 5.0
            if np.any(mask):
                coh_an    = MultiBeadCoherenceAnalyzer(
                    analyzer.time_series[mask], analyzer.positions[mask])
                coherence = coh_an.coherence_windowed_real_sum(ref_bead=0)
                res       = float(coherence[-1])
            else:
                res = 0.0

            shutil.rmtree(output_dir); os.remove(input_file)
            return res if not np.isnan(res) else 0.0

        except Exception:
            for path in [input_file, run_id]:
                if path != "unknown":
                    try:
                        if os.path.isdir(path):   shutil.rmtree(path)
                        elif os.path.isfile(path): os.remove(path)
                    except Exception: pass
            return np.nan

    def evaluate_state(self, inner_positions, T_signal, F0,
                       n_runs=3, total_time=100, num_cores=4):
        """Evaluate a single configuration; used during the BO phase."""
        results = self._run_tasks(
            [(inner_positions, T_signal, F0, rev, r, total_time)
             for r in range(n_runs) for rev in (False, True)],
            num_cores,
        )
        return self._parse_fwd_rev(results, n_runs)

    def _evaluate_batch(self, configs, n_runs, total_time, num_cores):

        tasks = [
            (pos, T, F0, rev, cfg_i * n_runs + r, total_time)
            for cfg_i, (pos, T, F0) in enumerate(configs)
            for r in range(n_runs)
            for rev in (False, True)
        ]
        flat = self._run_tasks(tasks, num_cores)

        # each config occupies a contiguous block of 2*n_runs results
        # within that block, even indices = fwd, odd = rev (matching task order)
        sims_per_cfg = 2 * n_runs
        return [
            self._parse_fwd_rev(flat[i * sims_per_cfg:(i + 1) * sims_per_cfg], n_runs)
            for i in range(len(configs))
        ]

    def _run_tasks(self, tasks, num_cores):
        with multiprocessing.Pool(processes=min(num_cores, len(tasks))) as pool:
            return pool.starmap(self.run_single_sim, tasks)

    @staticmethod
    def _parse_fwd_rev(results, n_runs):
        fwd_c = [results[2 * r]     for r in range(n_runs) if not np.isnan(results[2 * r])]
        rev_c = [results[2 * r + 1] for r in range(n_runs) if not np.isnan(results[2 * r + 1])]
        f = float(np.mean(fwd_c)) if fwd_c else 0.0
        r = float(np.mean(rev_c)) if rev_c else 0.0
        return f, r, f - r


    def _make_bounds(self, T_range, F0_range):
        lo = np.array([0.0] * self.num_inner + [T_range[0],  F0_range[0]])
        hi = np.array([self.chain_length] * self.num_inner + [T_range[1], F0_range[1]])
        return lo, hi

    def _normalise(self, X, lo, hi):
        return (np.asarray(X) - lo) / (hi - lo)

    def _denormalise(self, X_norm, lo, hi):
        return np.asarray(X_norm) * (hi - lo) + lo

    def _sample_valid_configs(self, n, T_range, F0_range):

        N_gaps = self.N - 1
        excess = self.chain_length - N_gaps * MIN_DISTANCE
        if excess <= 0:
            raise ValueError(
                f"Chain length {self.chain_length:.2f} µm is too short for N={self.N} "
                f"beads with MIN_DISTANCE={MIN_DISTANCE} µm."
            )
        # uniform simplex sampling via exponential normalisation
        raw    = np.random.exponential(scale=1.0, size=(n, N_gaps))
        fracs  = raw / raw.sum(axis=1, keepdims=True)
        gaps   = fracs * excess + MIN_DISTANCE       # (n, N_gaps), all >= MIN_DISTANCE

        inner  = np.cumsum(gaps, axis=1)[:, :-1]     # (n, num_inner)

        T_samp  = np.random.uniform(T_range[0],  T_range[1],  (n, 1))
        F0_samp = np.random.uniform(F0_range[0], F0_range[1], (n, 1))
        return np.hstack([inner, T_samp, F0_samp])   # (n, ndim)


    def _build_gp(self, ndim):

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(ndim),
                     length_scale_bounds=(0.05, 10.0),
                     nu=2.5)
            + WhiteKernel(noise_level=0.01,
                          noise_level_bounds=(1e-5, 0.5))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimiser=5,
            normalize_y=True,
            random_state=0,
        )

    def _acq_ucb(self, X_norm, gp, kappa):
        mu, sigma = gp.predict(X_norm, return_std=True)
        return mu + kappa * sigma

    def _acq_ei(self, X_norm, gp, f_best, xi=0.01):
        mu, sigma = gp.predict(X_norm, return_std=True)
        Z    = (mu - f_best - xi) / (sigma + 1e-9)
        return (mu - f_best - xi) * scipy_norm.cdf(Z) + sigma * scipy_norm.pdf(Z)

    def _next_candidate(self, gp, lo, hi, T_range, F0_range,
                        acquisition, kappa, f_best, n_candidates=50_000):

        candidates = self._sample_valid_configs(n_candidates, T_range, F0_range)
        cand_norm  = self._normalise(candidates, lo, hi)

        if acquisition == "ucb":
            acq_vals = self._acq_ucb(cand_norm, gp, kappa)
        else:
            acq_vals = self._acq_ei(cand_norm, gp, f_best)

        best_idx = int(np.argmax(acq_vals))
        return candidates[best_idx], float(acq_vals[best_idx])

    # warm start loader
    def _load_warm_start(self, json_path, T_range, F0_range):

        with open(json_path) as fh:
            h = json.load(fh)
        X, y = [], []
        for pos, T, F0, asym in zip(h["positions"], h["T_signal"],
                                    h["F0"],         h["asymmetry"]):
            if np.isnan(asym): continue
            if not (T_range[0]  <= T  <= T_range[1]):  continue
            if not (F0_range[0] <= F0 <= F0_range[1]): continue
            if not self._is_valid(pos): continue
            X.append(list(pos) + [T, F0])
            y.append(float(asym))
        print(f"[Warm start] Loaded {len(X)} valid points from {json_path}")
        return np.array(X), np.array(y)

    # history helpers
    def _record(self, history, eval_num, pos, T, F0,
                f, r, asym, phase, best_so_far, acq_val):
        history["eval_num"].append(int(eval_num))
        history["positions"].append([float(p) for p in pos])
        history["distances"].append([float(d) for d in self._positions_to_distances(pos)])
        history["T_signal"].append(float(T))
        history["F0"].append(float(F0))
        history["fwd_coh"].append(float(f))
        history["rev_coh"].append(float(r))
        history["asymmetry"].append(float(asym))
        history["phase"].append(phase)
        history["best_so_far"].append(float(best_so_far))
        history["acq_value"].append(float(acq_val) if acq_val is not None else None)

    def save_history(self, history, output_dir):
        path = os.path.join(output_dir, "bo_history.json")
        with open(path, "w") as fh:
            json.dump(history, fh, indent=4, cls=_NumpyEncoder)


    def optimise(self,
                 chain_length   = None,
                 n_initial      = None,
                 n_iterations   = 50,
                 init_positions = None,
                 init_T         = 0.3,
                 init_F0        = 2.5,
                 acquisition    = "ucb",
                 kappa          = 2.0,
                 T_range        = (0.01, 2.0),
                 F0_range       = (0.5, 10.0),
                 n_runs         = 3,
                 total_time     = 100,
                 num_cores      = 4,
                 warm_start_json = None,
                 output_dir     = "bo_opt"):

        os.makedirs(output_dir, exist_ok=True)

        if chain_length is not None:
            self.chain_length = float(chain_length)

        ndim      = self.num_inner + 2
        n_initial = n_initial if n_initial is not None else max(8, 2 * ndim)
        lo, hi    = self._make_bounds(T_range, F0_range)

        if init_positions is not None:
            init_pos = np.array(init_positions, dtype=float)
        else:
            spacing  = self.chain_length / (self.N - 1)
            init_pos = np.array([spacing * i for i in range(1, self.N - 1)])

        if len(init_pos) != self.num_inner:
            raise ValueError(f"Expected {self.num_inner} positions, got {len(init_pos)}.")
        if not self._is_valid(init_pos):
            raise ValueError(f"Initial positions violate MIN_DISTANCE. "
                             f"Gaps: {self._positions_to_distances(init_pos)}")

        print(f"\n{'='*66}")
        print(f"Bayesian Optimisation  —  {ndim}-D parameter space")
        print(f"  Beads N={self.N}  |  free inner: {self.num_inner}  "
              f"|  chain length: {self.chain_length:.2f} µm")
        print(f"  Acquisition: {acquisition.upper()}  (kappa={kappa})")
        print(f"  Initial random evals: {n_initial}  |  BO iterations: {n_iterations}")
        print(f"  n_runs={n_runs}  |  total_time={total_time}s  |  cores={num_cores}")
        print(f"{'='*66}\n")

        X_data, y_data = [], []
        history = {
            "eval_num":    [],
            "positions":   [],
            "distances":   [],
            "T_signal":    [],
            "F0":          [],
            "fwd_coh":     [],
            "rev_coh":     [],
            "asymmetry":   [],
            "phase":       [],
            "best_so_far": [],
            "acq_value":   [],
            "config": {
                "N":            self.N,
                "chain_length": self.chain_length,
                "T_range":      list(T_range),
                "F0_range":     list(F0_range),
                "n_initial":    n_initial,
                "n_iterations": n_iterations,
                "acquisition":  acquisition,
                "kappa":        kappa,
                "n_runs":       n_runs,
                "timestamp":    datetime.datetime.now().isoformat(),
            },
        }

        if warm_start_json:
            Xw, yw = self._load_warm_start(warm_start_json, T_range, F0_range)
            X_data.extend(Xw.tolist()); y_data.extend(yw.tolist())


        random_inits = self._sample_valid_configs(n_initial - 1, T_range, F0_range)
        init_configs = (
            [(init_pos.tolist(), float(init_T), float(init_F0))]
            + [(cand[:self.num_inner].tolist(),
                float(cand[self.num_inner]),
                float(cand[self.num_inner + 1]))
               for cand in random_inits]
        )

        total_sims = n_initial * 2 * n_runs
        print(f"[Initial phase] Launching {n_initial} configs "
              f"({total_sims} simulations) across {num_cores} cores...")

        batch_results = self._evaluate_batch(init_configs, n_runs, total_time, num_cores)

        eval_count = 0
        for (pos, T, F0), (f, r, asym) in zip(init_configs, batch_results):
            X_data.append(list(pos) + [T, F0])
            y_data.append(asym)
            self._record(history, eval_count, pos, T, F0,
                         f, r, asym, "initial", max(y_data), None)
            print(f"  [{eval_count+1}/{n_initial}]  "
                  f"pos={[f'{p:.2f}' for p in pos]}  T={T:.4f}  F0={F0:.4f}  "
                  f"→ Fwd={f:.4f}  Rev={r:.4f}  Asym={asym:.4f}")
            eval_count += 1

        self.save_history(history, output_dir)
        self.plot_history(history, output_dir)
        print(f"\n[BO start]  Best after initial phase: {max(y_data):.4f}\n")

        gp = self._build_gp(ndim)

        for bo_iter in range(1, n_iterations + 1):
            X_arr  = np.array(X_data)
            y_arr  = np.array(y_data)
            X_norm = self._normalise(X_arr, lo, hi)
            gp.fit(X_norm, y_arr)
            f_best = float(np.max(y_arr))

            next_cand, acq_val = self._next_candidate(
                gp, lo, hi, T_range, F0_range,
                acquisition, kappa, f_best,
            )
            pos = next_cand[:self.num_inner]
            T   = float(next_cand[self.num_inner])
            F0  = float(next_cand[self.num_inner + 1])

            print(f"[BO {bo_iter}/{n_iterations}]  "
                  f"pos={[f'{p:.2f}' for p in pos]}  T={T:.4f}  F0={F0:.4f}  "
                  f"(acq={acq_val:.4f})")
            f, r, asym = self.evaluate_state(pos, T, F0, n_runs, total_time, num_cores)
            new_best   = max(f_best, asym)
            print(f"  → Fwd={f:.4f}  Rev={r:.4f}  Asym={asym:.4f}  "
                  f"best={new_best:.4f}  {'↑ NEW BEST' if asym > f_best else ''}")

            X_data.append(next_cand.tolist()); y_data.append(asym)
            self._record(history, eval_count, pos, T, F0,
                         f, r, asym, "bo", float(np.max(y_data)), acq_val)
            eval_count += 1

            if bo_iter % 5 == 0:
                self.save_history(history, output_dir)
                self.plot_history(history, output_dir)

        # summary
        best_idx  = int(np.argmax(y_data))
        best_X    = X_data[best_idx]
        best_pos  = best_X[:self.num_inner]
        best_T    = best_X[self.num_inner]
        best_F0   = best_X[self.num_inner + 1]
        best_asym = y_data[best_idx]

        print(f"\n{'='*66}")
        print(f"Finished!  Best asymmetry: {best_asym:.4f}")
        print(f"  positions (µm): {[f'{p:.3f}' for p in best_pos]}")
        print(f"  distances (µm): {[f'{d:.3f}' for d in self._positions_to_distances(best_pos)]}")
        print(f"  T_signal={best_T:.4f} s,  F0={best_F0:.4f}")
        print(f"  Total evaluations (this run): {eval_count}"
              f"  (+{len(y_data) - eval_count} warm-start)")
        print(f"{'='*66}")

        self.save_history(history, output_dir)
        self.plot_history(history, output_dir)

        X_norm = self._normalise(np.array(X_data), lo, hi)
        gp.fit(X_norm, np.array(y_data))
        self._plot_gp_slice(gp, lo, hi, best_pos, T_range, F0_range, output_dir)

        return history

    # plots
    def plot_history(self, history, output_dir):
        eval_nums = history["eval_num"]
        positions = np.array(history["positions"])
        distances = np.array(history["distances"])
        T_vals    = history["T_signal"]
        F0_vals   = history["F0"]
        asym_vals = history["asymmetry"]
        fwd_vals  = history["fwd_coh"]
        rev_vals  = history["rev_coh"]
        best_vals = history["best_so_far"]
        phases    = history["phase"]
        n_initial = history["config"]["n_initial"]

        num_inner = positions.shape[1]
        num_gaps  = distances.shape[1]
        bo_mask   = np.array([p == "bo" for p in phases])

        plt.rcParams.update({
            "font.family": "serif", "font.size": 10,
            "axes.labelsize": 12,   "axes.titlesize": 12,
            "text.usetex": False,
        })

        fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)

        # (0,0)
        ax = axes[0, 0]
        col = ["#5060c0" if p == "initial" else "#c04030" for p in phases]
        ax.scatter(eval_nums, asym_vals, c=col, s=22, zorder=3, alpha=0.75,
                   label="Initial / BO evaluations")
        ax.plot(eval_nums, best_vals, "k-", linewidth=2.0, label="Best so far", zorder=4)
        ax.axvline(n_initial - 1, color="grey", linestyle="--",
                   linewidth=0.9, label="BO starts")
        ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Evaluation number")
        ax.set_ylabel(r"Asymmetry $\Delta C$")
        ax.set_title("Optimisation progress", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (0, 1)
        ax = axes[0, 1]
        sc = ax.scatter(T_vals, F0_vals, c=asym_vals, cmap="viridis",
                        s=25, alpha=0.85, zorder=3)
        plt.colorbar(sc, ax=ax, label=r"Asymmetry $\Delta C$")
        if bo_mask.any():
            ax.scatter(np.array(T_vals)[bo_mask], np.array(F0_vals)[bo_mask],
                       s=55, facecolors="none", edgecolors="red",
                       linewidths=1.0, zorder=4, label="BO selected")
        best_idx = int(np.argmax(asym_vals))
        ax.scatter(T_vals[best_idx], F0_vals[best_idx],
                   c="yellow", s=120, marker="*",
                   edgecolors="black", zorder=5, label="Best")
        ax.set_xlabel(r"$T_\mathrm{signal}$ (s)")
        ax.set_ylabel(r"$F_0$")
        ax.set_title(r"Evaluated points in ($T$, $F_0$) space", fontweight="bold")
        ax.legend(fontsize=8)

        # (1, 0)        
        ax = axes[1, 0]
        colors_pos = plt.cm.plasma(np.linspace(0.1, 0.9, num_inner))
        for k in range(num_inner):
            ax.plot(eval_nums, positions[:, k], color=colors_pos[k],
                    linewidth=1.3, label=fr"$x_{k+1}$")
        L = history["config"]["chain_length"]
        ax.axhline(0, color="black", linestyle=":",  linewidth=0.8, label=r"$x_0 = 0$")
        ax.axhline(L, color="black", linestyle="--", linewidth=0.8,
                   label=fr"$x_{{N-1}}={L:.1f}$ µm")
        ax.axvline(n_initial - 1, color="grey", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Evaluation number")
        ax.set_ylabel("Position (µm)")
        ax.set_title("Inner bead positions", fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # (1, 1)
        ax = axes[1, 1]
        colors_gap = plt.cm.cool(np.linspace(0.1, 0.9, num_gaps))
        for k in range(num_gaps):
            ax.plot(eval_nums, distances[:, k], color=colors_gap[k],
                    linewidth=1.3, label=fr"$d_{k}$")
        ax.axhline(MIN_DISTANCE, color="red", linestyle="--",
                   linewidth=0.9, label=f"min = {MIN_DISTANCE} µm")
        ax.axvline(n_initial - 1, color="grey", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Evaluation number")
        ax.set_ylabel("Distance (µm)")
        ax.set_title("Inter-bead distances", fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # (2, 0) 
        ax = axes[2, 0]
        ax.plot(eval_nums, fwd_vals, "b-", linewidth=1.3, label="Forward")
        ax.plot(eval_nums, rev_vals, "r-", linewidth=1.3, label="Reverse")
        ax.axvline(n_initial - 1, color="grey", linestyle="--",
                   linewidth=0.9, label="BO starts")
        ax.set_xlabel("Evaluation number")
        ax.set_ylabel("Coherence")
        ax.set_title("Coherence components", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2, 1))
        ax  = axes[2, 1]
        ax2 = ax.twinx()
        p1, = ax.plot( eval_nums, T_vals,  "m-", linewidth=1.3,
                       label=r"$T_\mathrm{signal}$")
        p2, = ax2.plot(eval_nums, F0_vals, "c-", linewidth=1.3, label=r"$F_0$")
        ax.axvline(n_initial - 1, color="grey", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Evaluation number")
        ax.set_ylabel(r"$T_\mathrm{signal}$ (s)", color="m")
        ax2.set_ylabel(r"$F_0$", color="c")
        ax.set_title("Driving parameters", fontweight="bold")
        ax.legend(handles=[p1, p2], fontsize=8)

        plt.savefig(os.path.join(output_dir, "bo_progress.png"), dpi=300)
        plt.close()

    def _plot_gp_slice(self, gp, lo, hi, best_pos, T_range, F0_range, output_dir):

        n_grid  = 80
        T_grid  = np.linspace(T_range[0],  T_range[1],  n_grid)
        F0_grid = np.linspace(F0_range[0], F0_range[1], n_grid)
        TT, FF  = np.meshgrid(T_grid, F0_grid)

        pos_block = np.tile(best_pos, (n_grid * n_grid, 1))
        X_slice   = np.hstack([pos_block, TT.ravel()[:, None], FF.ravel()[:, None]])
        mu, sigma = gp.predict(self._normalise(X_slice, lo, hi), return_std=True)
        mu        = mu.reshape(n_grid, n_grid)
        sigma     = sigma.reshape(n_grid, n_grid)

        ext = [T_range[0], T_range[1], F0_range[0], F0_range[1]]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        im1 = ax1.imshow(mu,    origin="lower", extent=ext, aspect="auto", cmap="viridis")
        fig.colorbar(im1, ax=ax1, label=r"GP mean $\hat{\mu}(\Delta C)$")
        ax1.set_xlabel(r"$T_\mathrm{signal}$ (s)")
        ax1.set_ylabel(r"$F_0$")
        ax1.set_title("GP posterior mean\n(positions fixed at best)", fontweight="bold")

        im2 = ax2.imshow(sigma, origin="lower", extent=ext, aspect="auto", cmap="magma")
        fig.colorbar(im2, ax=ax2, label=r"GP std $\hat{\sigma}$")
        ax2.set_xlabel(r"$T_\mathrm{signal}$ (s)")
        ax2.set_ylabel(r"$F_0$")
        ax2.set_title("GP posterior std\n(unexplored regions are brighter)",
                      fontweight="bold")

        plt.savefig(os.path.join(output_dir, "gp_surrogate_slice.png"), dpi=300)
        plt.close()
        print(f"[INFO] GP surrogate slice → {output_dir}/gp_surrogate_slice.png")



# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Bayesian Optimisation of bead asymmetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # chain geometry
    parser.add_argument("--chain_length",    type=float, required=True,
                        help="Total chain length (µm) — distance from first to last bead.")
    # optimisation settings
    parser.add_argument("--n_initial",       type=int,   default=None,
                        help="Random initial evaluations before BO "
                             "(default: max(8, 2*ndim)).")
    parser.add_argument("--n_iterations",    type=int,   default=50,
                        help="Number of BO-guided evaluations.")
    parser.add_argument("--acquisition",     type=str,   default="ucb",
                        choices=["ucb", "ei"],
                        help="Acquisition function: UCB (explore/exploit balance) "
                             "or EI (expected improvement).")
    parser.add_argument("--kappa",           type=float, default=2.0,
                        help="UCB exploration weight κ (higher = more exploration).")
    # initial point
    parser.add_argument("--init_positions",  type=float, nargs="+", default=None,
                        help="Starting inner bead positions (µm, N-2 values). "
                             "Default: equally spaced.")
    parser.add_argument("--T_init",          type=float, default=0.3)
    parser.add_argument("--F0_init",         type=float, default=2.5)
    # parameter bounds
    parser.add_argument("--T_min",           type=float, default=0.01)
    parser.add_argument("--T_max",           type=float, default=2.0)
    parser.add_argument("--F0_min",          type=float, default=0.5)
    parser.add_argument("--F0_max",          type=float, default=10.0)
    # simulation
    parser.add_argument("--runs",            type=int,   default=3)
    parser.add_argument("--time",            type=int,   default=100)
    parser.add_argument("--cores",           type=int,   default=6)
    # engine / template
    parser.add_argument("--exe",             type=str,
                        default="./src/rowers")
    parser.add_argument("--template",        type=str,
                        default="./templates/6beads.input")
    # warm start and output name
    parser.add_argument("--warm_start",      type=str,   default=None,
                        metavar="JSON",
                        help="Path to a previous metropolis_history.json or "
                             "bo_history.json to seed the GP.")
    parser.add_argument("--name",            type=str,   default=None)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = f"bo_results_{timestamp}"
    if args.name:
        out_dir += f"_{args.name}"

    optimiser = BeadBayesianoptimiser(
        executable=args.exe, template_input=args.template
    )
    optimiser.optimise(
        chain_length    = args.chain_length,
        n_initial       = args.n_initial,
        n_iterations    = args.n_iterations,
        init_positions  = args.init_positions,
        init_T          = args.T_init,
        init_F0         = args.F0_init,
        acquisition     = args.acquisition,
        kappa           = args.kappa,
        T_range         = (args.T_min, args.T_max),
        F0_range        = (args.F0_min, args.F0_max),
        n_runs          = args.runs,
        total_time      = args.time,
        num_cores       = args.cores,
        warm_start_json = args.warm_start,
        output_dir      = out_dir,
    )
