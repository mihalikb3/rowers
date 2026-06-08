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
from bead_analysis import BeadDataAnalyzer, MultiBeadCoherenceAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

MIN_DISTANCE = 3.1   # hard lower bound between any two adjacent beads 


class BeadMetropolisoptimiser:
    def __init__(self, executable="./modern/modern_Beads",
                 template_input="BeadsV_6Beads_0Signal_UPDlambda2.input"):
        self.executable = executable
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
        self.N          = int(self._template_params[0])
        self.num_inner  = self.N - 2 # number of free position parameters
        self.default_lambda = float(self._template_params[21])

        num_gaps = self.N - 1
        self._template_distances = [
            float(self._template_params[22 + i]) for i in range(num_gaps)
        ]
        self.chain_length = sum(self._template_distances)

        cumpos = np.cumsum([0.0] + self._template_distances)
        self._template_inner_positions = cumpos[1:-1].tolist()   # x_1 … x_{N-2}

        print(f"[Template] N={self.N}, chain_length={self.chain_length:.3f} µm")
        print(f"[Template] inner bead positions: "
              f"{[f'{p:.3f}' for p in self._template_inner_positions]}")

    def _positions_to_distances(self, inner_positions):
        all_pos = [0.0] + list(inner_positions) + [self.chain_length]
        return [all_pos[i + 1] - all_pos[i] for i in range(self.N - 1)]

    def _is_valid(self, inner_positions):
        dists = self._positions_to_distances(inner_positions)
        return all(d >= MIN_DISTANCE for d in dists)

    def _build_params(self, inner_positions, T_signal, F0, total_time, reverse=False):
        params = list(self._template_params[:22])
        params[5]  = str(total_time)
        params[13] = "50"
        params[19] = f"{F0:.6f}"
        params[20] = f"{T_signal:.6f}"

        distances = self._positions_to_distances(inner_positions)
        if reverse:
            distances = distances[::-1]

        params += [f"{d:.6f}" for d in distances]
        params += [f"{self.default_lambda:.6f}"] * self.N
        return params


    # single sim
    def run_single_sim(self, inner_positions, T_signal, F0,
                       reverse=False, seed_offset=0, total_time=100):
        run_id     = "unknown"
        input_file = "unknown"
        try:
            params = self._build_params(inner_positions, T_signal, F0, total_time, reverse)

            # compact unique ID using a hash
            state_str  = (f"{seed_offset}_{','.join(f'{p:.4f}' for p in inner_positions)}"
                          f"_T{T_signal:.6f}_F{F0:.6f}_{'r' if reverse else 'f'}")
            h          = hashlib.md5(state_str.encode()).hexdigest()[:10]
            run_id     = f"metro_{os.getpid()}_{h}"
            input_file = f"{run_id}.input"

            with open(input_file, "w") as fh:
                fh.write(" ".join(params) + "\n")

            subprocess.run(
                f"{self.executable} -C -R {input_file}",
                shell=True, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

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

            # discard first 5 s transient
            mask = analyzer.time_series >= 5.0
            if np.any(mask):
                coh_an   = MultiBeadCoherenceAnalyzer(
                    analyzer.time_series[mask], analyzer.positions[mask]
                )
                coherence = coh_an.coherence_windowed_real_sum(ref_bead=0)
                res       = float(coherence[-1])   # coherence of the last bead
            else:
                res = 0.0

            shutil.rmtree(output_dir)
            os.remove(input_file)
            return res if not np.isnan(res) else 0.0

        except Exception:
            for path in [input_file, run_id]:
                if path != "unknown":
                    try:
                        if os.path.isdir(path):   shutil.rmtree(path)
                        elif os.path.isfile(path): os.remove(path)
                    except Exception:
                        pass
            return np.nan


    def evaluate_state(self, inner_positions, T_signal, F0,
                       n_runs=3, total_time=100, num_cores=4):
        tasks = []
        for r in range(n_runs):
            tasks.append((inner_positions, T_signal, F0, False, r, total_time))
            tasks.append((inner_positions, T_signal, F0, True,  r, total_time))

        with multiprocessing.Pool(processes=min(num_cores, len(tasks))) as pool:
            results = pool.starmap(self.run_single_sim, tasks)

        fwd = [results[i] for i in range(0, len(results), 2)]
        rev = [results[i] for i in range(1, len(results), 2)]

        fwd_clean = [x for x in fwd if not np.isnan(x)]
        rev_clean = [x for x in rev if not np.isnan(x)]

        f_mean = np.mean(fwd_clean) if fwd_clean else 0.0
        r_mean = np.mean(rev_clean) if rev_clean else 0.0
        return f_mean, r_mean, f_mean - r_mean


    def optimise(self,
                 chain_length=None,
                 init_positions=None,
                 init_T=0.3,
                 init_F0=2.5,
                 # proposal steps
                 pos_step=0.5,
                 T_step=0.02,
                 F0_step=0.2,
                 # parameter bounds
                 T_range=(0.01, 2.0),
                 F0_range=(0.5, 10.0),
                 # optimisation settings
                 n_iterations=50,
                 temperature=0.05,
                 n_runs=3,
                 total_time=100,
                 num_cores=4,
                 output_dir="metro_opt"):

        os.makedirs(output_dir, exist_ok=True)

        if chain_length is not None:
            self.chain_length = float(chain_length)

        if init_positions is not None:
            current_pos = np.array(init_positions, dtype=float)
        else:
            # equal spacing
            spacing = self.chain_length / (self.N - 1)
            current_pos = np.array(
                [spacing * i for i in range(1, self.N - 1)], dtype=float
            )
        current_T   = float(np.clip(init_T,   T_range[0],  T_range[1]))
        current_F0  = float(np.clip(init_F0, F0_range[0], F0_range[1]))

        if len(current_pos) != self.num_inner:
            raise ValueError(
                f"Expected {self.num_inner} inner positions for N={self.N}, "
                f"got {len(current_pos)}."
            )
        if not self._is_valid(current_pos):
            raise ValueError(
                f"Initial positions violate MIN_DISTANCE={MIN_DISTANCE} µm. "
                f"Gaps: {self._positions_to_distances(current_pos)}"
            )

        ndim = self.num_inner + 2   # positions + T + F0
        print(f"\n{'='*62}")
        print(f"Metropolis optimisation  —  {ndim}-D parameter space")
        print(f"  Beads: N={self.N}  |  inner (free): {self.num_inner}  "
              f"|  chain length: {self.chain_length:.2f} µm")
        print(f"  Initial positions (µm): {[f'{p:.3f}' for p in current_pos]}")
        print(f"  Initial distances (µm): {[f'{d:.3f}' for d in self._positions_to_distances(current_pos)]}")
        print(f"  Initial T={current_T:.4f} s,  F0={current_F0:.4f}")
        print(f"  Iterations: {n_iterations}  |  n_runs: {n_runs}  |  cores: {num_cores}")
        print(f"  Step sizes:  pos={pos_step} µm,  T={T_step} s,  F0={F0_step}")
        print(f"{'='*62}\n")

        f_curr, r_curr, asym_curr = self.evaluate_state(
            current_pos, current_T, current_F0,
            n_runs, total_time, num_cores,
        )
        current_energy = -asym_curr
        print(f"[Iter 0]  Fwd={f_curr:.4f}  Rev={r_curr:.4f}  Asym={asym_curr:.4f}")

        history = {
            "iteration": [0],
            "positions": [current_pos.tolist()],
            "distances": [self._positions_to_distances(current_pos)],
            "T_signal":  [current_T],
            "F0":        [current_F0],
            "fwd_coh":   [f_curr],
            "rev_coh":   [r_curr],
            "asymmetry": [asym_curr],
            "accepted":  [True],
            "config": {
                "N":            self.N,
                "chain_length": self.chain_length,
                "T_range":      list(T_range),
                "F0_range":     list(F0_range),
                "temperature":  temperature,
                "n_runs":       n_runs,
                "pos_step":     pos_step,
                "T_step":       T_step,
                "F0_step":      F0_step,
                "timestamp":    datetime.datetime.now().isoformat(),
            },
        }

        best_asym  = asym_curr
        best_state = (current_pos.copy(), current_T, current_F0)
        auto_rejected = 0

        for i in range(1, n_iterations + 1):
            # proposal
            next_pos = current_pos + np.random.normal(0, pos_step, size=self.num_inner)
            next_T   = float(np.clip(current_T  + np.random.normal(0, T_step),
                                     T_range[0],  T_range[1]))
            next_F0  = float(np.clip(current_F0 + np.random.normal(0, F0_step),
                                     F0_range[0], F0_range[1]))

            print(f"\n--- Iteration {i}/{n_iterations} ---")

            # auto rejections
            if not self._is_valid(next_pos):
                auto_rejected += 1
                min_gap = min(self._positions_to_distances(next_pos))
                print(f"AUTO-REJECTED (min gap {min_gap:.3f} µm < {MIN_DISTANCE} µm, "
                      f"{auto_rejected} total)")
                history["iteration"].append(i)
                history["positions"].append(current_pos.tolist())
                history["distances"].append(self._positions_to_distances(current_pos))
                history["T_signal"].append(current_T)
                history["F0"].append(current_F0)
                history["fwd_coh"].append(f_curr)
                history["rev_coh"].append(r_curr)
                history["asymmetry"].append(asym_curr)
                history["accepted"].append(False)
                if i % 5 == 0:
                    self.save_history(history, output_dir)
                    self.plot_history(history, output_dir)
                continue

            print(f"Proposing:  pos={[f'{p:.2f}' for p in next_pos]}  "
                  f"T={next_T:.4f}  F0={next_F0:.4f}")

            f_next, r_next, asym_next = self.evaluate_state(
                next_pos, next_T, next_F0,
                n_runs, total_time, num_cores,
            )
            next_energy = -asym_next
            delta_E     = next_energy - current_energy

            # acceptance criterion
            if delta_E < 0:
                accepted = True
            else:
                p_acc    = np.exp(-delta_E / temperature)
                accepted = np.random.random() < p_acc

            if accepted:
                print(f"ACCEPTED   Asym {asym_curr:.4f} → {asym_next:.4f}  "
                      f"(ΔE={delta_E:+.4f})")
                current_pos, current_T, current_F0 = next_pos.copy(), next_T, next_F0
                current_energy = next_energy
                asym_curr, f_curr, r_curr = asym_next, f_next, r_next
                if asym_curr > best_asym:
                    best_asym  = asym_curr
                    best_state = (current_pos.copy(), current_T, current_F0)
            else:
                p_acc_str = f"{np.exp(-delta_E / temperature):.4f}"
                print(f"REJECTED   Asym stays {asym_curr:.4f}  "
                      f"(ΔE={delta_E:+.4f}, p_acc={p_acc_str})")

            history["iteration"].append(i)
            history["positions"].append(current_pos.tolist())
            history["distances"].append(self._positions_to_distances(current_pos))
            history["T_signal"].append(current_T)
            history["F0"].append(current_F0)
            history["fwd_coh"].append(f_curr)
            history["rev_coh"].append(r_curr)
            history["asymmetry"].append(asym_curr)
            history["accepted"].append(accepted)

            if i % 5 == 0:
                self.save_history(history, output_dir)
                self.plot_history(history, output_dir)

        # print summary
        best_pos, best_T, best_F0 = best_state
        print(f"\n{'='*62}")
        print(f"Optimisation finished!  Best asymmetry: {best_asym:.4f}")
        print(f"  positions (µm): {[f'{p:.3f}' for p in best_pos]}")
        print(f"  distances (µm): {[f'{d:.3f}' for d in self._positions_to_distances(best_pos)]}")
        print(f"  T_signal={best_T:.4f} s,  F0={best_F0:.4f}")
        print(f"  Auto-rejected proposals: {auto_rejected}/{n_iterations}")
        print(f"{'='*62}")

        self.save_history(history, output_dir)
        self.plot_history(history, output_dir)
        return history


    # json dump
    def save_history(self, history, output_dir):
        with open(os.path.join(output_dir, "metropolis_history.json"), "w") as fh:
            json.dump(history, fh, indent=4, cls=_NumpyEncoder)

    # plots
    def plot_history(self, history, output_dir):
        iters     = history["iteration"]
        positions = np.array(history["positions"])
        distances = np.array(history["distances"])
        T_vals    = history["T_signal"]
        F0_vals   = history["F0"]
        asym_vals = history["asymmetry"]
        fwd_vals  = history["fwd_coh"]
        rev_vals  = history["rev_coh"]
        accepted  = np.array(history["accepted"])
        cfg       = history.get("config", {})

        num_inner = positions.shape[1]
        num_gaps  = distances.shape[1]

        plt.rcParams.update({
            "font.family":     "serif",
            "font.size":       10,
            "axes.labelsize":  12,
            "axes.titlesize":  12,
            "text.usetex":     False,
        })

        fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)

        # (0, 0)
        ax = axes[0, 0]
        sc = ax.scatter(T_vals, F0_vals, c=asym_vals, cmap="viridis", s=20, alpha=0.7, zorder=3)
        ax.plot(T_vals, F0_vals, "k-", alpha=0.25, linewidth=0.8)
        plt.colorbar(sc, ax=ax, label=r"Asymmetry $\Delta C$")
        ax.scatter(T_vals[0],  F0_vals[0],  c="yellow", s=100, marker="*",
                   edgecolors="black", zorder=5, label="Start")
        ax.scatter(T_vals[-1], F0_vals[-1], c="blue",   s=100, marker="X",
                   edgecolors="white",  zorder=5, label="End")
        ax.set_xlabel(r"$T_\mathrm{signal}$ (s)")
        ax.set_ylabel(r"$F_0$")
        ax.set_title(r"Path in ($T_\mathrm{signal}$, $F_0$) space", fontweight="bold")
        ax.legend(fontsize=8)

        # (0, 1)
        ax = axes[0, 1]
        ax.plot(iters, asym_vals, color="green", linewidth=1.5)
        acc_idx = [k for k in range(len(iters)) if accepted[k]]
        ax.scatter([iters[k] for k in acc_idx],
                   [asym_vals[k] for k in acc_idx],
                   c="green", s=18, zorder=3, label="Accepted")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Asymmetry $\Delta C$")
        ax.set_title("Asymmetry history", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (1, 0)
        ax = axes[1, 0]
        colors_pos = plt.cm.plasma(np.linspace(0.1, 0.9, num_inner))
        for k in range(num_inner):
            ax.plot(iters, positions[:, k], color=colors_pos[k],
                    linewidth=1.3, label=fr"$x_{{{k+1}}}$")
        L = cfg.get("chain_length", self.chain_length)
        ax.axhline(0, color="black", linestyle=":",  linewidth=0.9, label=r"$x_0$ (fixed)")
        ax.axhline(L, color="black", linestyle="--", linewidth=0.9, label=fr"$x_{{N-1}}$ = {L:.1f} µm")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Position (µm)")
        ax.set_title("Inner bead positions", fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # (1, 1)
        ax = axes[1, 1]
        colors_gap = plt.cm.cool(np.linspace(0.1, 0.9, num_gaps))
        for k in range(num_gaps):
            ax.plot(iters, distances[:, k], color=colors_gap[k],
                    linewidth=1.3, label=fr"$d_{k}$")
        ax.axhline(MIN_DISTANCE, color="red", linestyle="--", linewidth=0.9,
                   label=f"min = {MIN_DISTANCE} µm")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Distance (µm)")
        ax.set_title("Inter-bead distances", fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # (2, 0)
        ax = axes[2, 0]
        ax.plot(iters, fwd_vals, "b-", linewidth=1.3, label="Forward")
        ax.plot(iters, rev_vals, "r-", linewidth=1.3, label="Reverse")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Coherence")
        ax.set_title("Coherence components", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2, 1)
        ax   = axes[2, 1]
        ax2  = ax.twinx()
        p1,  = ax.plot( iters, T_vals,  "m-", linewidth=1.3, label=r"$T_\mathrm{signal}$")
        p2,  = ax2.plot(iters, F0_vals, "c-", linewidth=1.3, label=r"$F_0$")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$T_\mathrm{signal}$ (s)", color="m")
        ax2.set_ylabel(r"$F_0$", color="c")
        ax.set_title("Driving parameters", fontweight="bold")
        ax.legend(handles=[p1, p2], fontsize=8)

        plt.savefig(os.path.join(output_dir, "metropolis_plots.png"), dpi=300)
        plt.close()


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Metropolis optimisation of bead asymmetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # optimisation settings
    parser.add_argument("--iterations",    type=int,   default=50)
    parser.add_argument("--temp",          type=float, default=0.02,
                        help="Metropolis temperature (dimensionless, ~0.01–0.1)")
    parser.add_argument("--runs",          type=int,   default=3,
                        help="Ensemble runs per evaluation point")
    parser.add_argument("--cores",         type=int,   default=6)
    parser.add_argument("--time",          type=int,   default=100,
                        help="Simulation duration (s)")
    # chain geometry
    parser.add_argument("--chain_length",   type=float, required=True,
                        help="Total chain length (µm) — distance from the first to the last bead.")
    # initial state
    parser.add_argument("--init_positions", type=float, nargs="+", default=None,
                        help="Initial inner bead positions in µm "
                             "(must be N-2 values). Defaults to equally spaced.")
    parser.add_argument("--T_init",        type=float, default=0.3)
    parser.add_argument("--F0_init",       type=float, default=2.5)
    # step sizes
    parser.add_argument("--pos_step",      type=float, default=0.5,
                        help="Std of Gaussian position proposal (µm)")
    parser.add_argument("--T_step",        type=float, default=0.02)
    parser.add_argument("--F0_step",       type=float, default=0.2)
    # parameter bounds
    parser.add_argument("--T_min",         type=float, default=0.01)
    parser.add_argument("--T_max",         type=float, default=2.0)
    parser.add_argument("--F0_min",        type=float, default=0.5)
    parser.add_argument("--F0_max",        type=float, default=10.0)
    # engine / template
    parser.add_argument("--exe",           type=str,   default="./src/rowers")
    parser.add_argument("--template",      type=str,
                        default="./templates/6beads.input")
    parser.add_argument("--name",          type=str,   default=None)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name  = f"metro_results_{timestamp}"
    if args.name:
        dir_name += f"_{args.name}"

    optimiser = BeadMetropolisoptimiser(
        executable=args.exe, template_input=args.template
    )
    optimiser.optimise(
        chain_length   = args.chain_length,
        init_positions = args.init_positions,
        init_T         = args.T_init,
        init_F0        = args.F0_init,
        pos_step       = args.pos_step,
        T_step         = args.T_step,
        F0_step        = args.F0_step,
        T_range        = (args.T_min,  args.T_max),
        F0_range       = (args.F0_min, args.F0_max),
        n_iterations   = args.iterations,
        temperature    = args.temp,
        n_runs         = args.runs,
        total_time     = args.time,
        num_cores      = args.cores,
        output_dir     = dir_name,
    )
