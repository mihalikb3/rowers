#!/usr/bin/env python3


import os
import hashlib
import sys
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import json
import shutil
import multiprocessing
import datetime
import warnings
from functools import partial
from bead_analysis import BeadDataAnalyzer, MultiBeadCoherenceAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

MIN_DISTANCE = 3.1  # µm — hard lower bound between adjacent beads


# each entry is either:
#   {'idx': int,        'desc': str}  — directly sets params[idx]
#   {'special': str,    'desc': str}  — handled by _apply_overrides() below
PARAM_DEFS = {
    'F0':          {'idx': 19,             'desc': 'Signal force amplitude'},
    'T_signal':    {'idx': 20,             'desc': 'Signal period (s)'},
    'amplitude':   {'idx':  1,             'desc': 'Trap oscillation amplitude'},
    'viscosity':   {'idx':  6,             'desc': 'Fluid viscosity (Pas)'},
    'kx':          {'idx':  8,             'desc': 'x-trap stiffness'},
    'ky':          {'idx': 10,             'desc': 'y-trap stiffness'},
    'temperature': {'idx': 11,             'desc': 'Thermal temperature'},
    'epsilon':     {'idx': 14,             'desc': 'Coupling parameter epsilon'},
    'KS':          {'idx': 16,             'desc': 'Spring constant KS'},
    'd_avg':       {'special': 'd_avg',    'desc': 'Mean inter-bead distance (µm)'},
    'd_grad':      {'special': 'd_grad',   'desc': 'Linear gradient of inter-bead distances (µm/bead)'},
    'g_factor':    {'special': 'g_factor', 'desc': 'Geometric ratio of successive inter-bead distances'},
}

_DIST_PARAMS = {'d_avg', 'd_grad', 'g_factor'}


def _apply_overrides(params, overrides):
    num_beads = int(params[0])
    num_gaps  = num_beads - 1

    dist_overrides   = {k: float(v) for k, v in overrides.items() if k in _DIST_PARAMS}
    scalar_overrides = {k: float(v) for k, v in overrides.items() if k not in _DIST_PARAMS}

    # scalar params
    for name, value in scalar_overrides.items():
        p = PARAM_DEFS.get(name)
        if p and 'idx' in p:
            params[p['idx']] = str(value)

    # composite distance params
    if dist_overrides:
        template_dists = [float(params[22 + i]) for i in range(num_gaps)]
        d_avg = dist_overrides.get('d_avg', float(np.mean(template_dists)))

        if 'g_factor' in dist_overrides:
            g = dist_overrides['g_factor']
            L = num_gaps * d_avg
            if abs(g - 1.0) < 1e-9:
                distances = [d_avg] * num_gaps
            else:
                d0 = L * (g - 1.0) / (g ** num_gaps - 1.0)
                distances = [d0 * (g ** i) for i in range(num_gaps)]
        else:
            d_grad = dist_overrides.get('d_grad', 0.0)
            distances = [d_avg + (i - (num_gaps - 1) / 2.0) * d_grad
                         for i in range(num_gaps)]

        for i in range(num_gaps):
            params[22 + i] = str(max(MIN_DISTANCE, distances[i]))

    return params


class BeadAsymmetrySweep:
    def __init__(self, executable="./Beads",
                 template_input="BeadsV_6Beads_0Signal_UPDlambda2.input"):
        self.executable     = executable
        self.template_input = template_input
        if not os.path.exists(self.executable):
            raise FileNotFoundError(f"Executable '{self.executable}' not found.")

    # Single simulation
    def run_single_sim(self, overrides, reverse=False, seed_offset=0, total_time=100):
        run_id = input_file = "unknown"
        try:
            with open(self.template_input) as fh:
                params = fh.readline().strip().split()

            params[5]  = str(total_time)
            params[13] = "50"            # write_sampling

            _apply_overrides(params, overrides)

            # reverse distance array for the mirror geometry
            num_beads = int(params[0])
            num_gaps  = num_beads - 1
            if reverse:
                dists = [params[22 + i] for i in range(num_gaps)]
                for i in range(num_gaps):
                    params[22 + i] = dists[num_gaps - 1 - i]

            state_str  = (f"{seed_offset}_{'r' if reverse else 'f'}_"
                          + "_".join(f"{k}={v:.6g}" for k, v in sorted(overrides.items())))
            h          = hashlib.md5(state_str.encode()).hexdigest()[:10]
            run_id     = f"sweep_{os.getpid()}_{h}"
            input_file = f"{run_id}.input"

            with open(input_file, 'w') as fh:
                fh.write(" ".join(params) + "\n")

            subprocess.run(
                f"{self.executable} -C -R {input_file}",
                shell=True, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

            output_dir = run_id
            if not os.path.exists(output_dir):
                return np.nan, np.nan
            dat_files = [f for f in os.listdir(output_dir) if f.endswith('.dat')]
            if not dat_files:
                return np.nan, np.nan

            analyzer = BeadDataAnalyzer()
            analyzer.read_data_file(os.path.join(output_dir, dat_files[0]))
            if analyzer.positions is None or len(analyzer.positions) < 10:
                return np.nan, np.nan

            mask = analyzer.time_series >= 5.0
            if np.any(mask):
                coh_an    = MultiBeadCoherenceAnalyzer(
                    analyzer.time_series[mask], analyzer.positions[mask])
                coherence = coh_an.coherence_windowed_real_sum(ref_bead=0)
                res       = float(coherence[-1])
            else:
                res = 0.0

            shutil.rmtree(output_dir)
            os.remove(input_file)
            return res if not np.isnan(res) else 0.0

        except Exception:
            for path in [input_file, run_id]:
                if path != "unknown":
                    try:
                        if os.path.isdir(path):    shutil.rmtree(path)
                        elif os.path.isfile(path): os.remove(path)
                    except Exception:
                        pass
            return np.nan

    # ensemble worker (called inside multiprocessing pool)
    def worker_ensemble(self, task, n_runs=1, total_time=100):
        task_idx, overrides = task
        fwd = [self.run_single_sim(overrides, reverse=False,
                                   seed_offset=r, total_time=total_time)
               for r in range(n_runs)]
        rev = [self.run_single_sim(overrides, reverse=True,
                                   seed_offset=r, total_time=total_time)
               for r in range(n_runs)]

        fwd_coh = [x for x in fwd if not np.isnan(x)]
        rev_coh = [x for x in rev if not np.isnan(x)]

        asym_per_seed = [f - r
                         for f, r in zip(fwd, rev)
                         if not np.isnan(f) and not np.isnan(r)]

        def _stats(data):
            if not data:
                return np.nan, np.nan
            mean = float(np.mean(data))
            sem  = float(np.std(data) / np.sqrt(len(data))) if len(data) > 1 else 0.0
            return mean, sem

        return (task_idx, {
            'fwd':  _stats(fwd_coh),
            'rev':  _stats(rev_coh),
            'asym': _stats(asym_per_seed),
        })

   
    # main sweep
    def run_sweep_ensemble(self,
                           param1_name, param1_range,
                           param2_name, param2_range,
                           fixed_overrides=None,
                           num_cores=4, n_runs=3, total_time=100,
                           output_dir="results"):

        if fixed_overrides is None:
            fixed_overrides = {}

        # build the flat task list: every (v1, v2) combination
        task_list = []
        for v1 in param1_range:
            for v2 in param2_range:
                overrides = dict(fixed_overrides)
                overrides[param1_name] = float(v1)
                overrides[param2_name] = float(v2)
                task_list.append(overrides)

        tasks       = list(enumerate(task_list))
        total_tasks = len(tasks)

        p1_desc = PARAM_DEFS.get(param1_name, {}).get('desc', param1_name)
        p2_desc = PARAM_DEFS.get(param2_name, {}).get('desc', param2_name)

        print(f"\n{'='*64}")
        print(f"Ensemble Sweep: {total_tasks} grid points "
              f"({len(param1_range)} × {len(param2_range)})")
        print(f"  param1 ({param1_name}): {param1_range[0]:.4g} → {param1_range[-1]:.4g}")
        print(f"  param2 ({param2_name}): {param2_range[0]:.4g} → {param2_range[-1]:.4g}")
        if fixed_overrides:
            print(f"  fixed:  {fixed_overrides}")
        print(f"  n_runs={n_runs}  |  total_time={total_time}s  |  cores={num_cores}")
        print(f"{'='*64}\n")

        results   = [None] * total_tasks
        completed = 0
        worker_fn = partial(self.worker_ensemble, n_runs=n_runs, total_time=total_time)

        with multiprocessing.Pool(processes=num_cores) as pool:
            for res in pool.imap_unordered(worker_fn, tasks):
                task_idx, data = res
                completed += 1
                results[task_idx] = res
                print(f"[{completed:4d}/{total_tasks}] point {task_idx:4d}"
                      f" | Fwd={data['fwd'][0]:.3f}"
                      f" | Rev={data['rev'][0]:.3f}"
                      f" | Asym={data['asym'][0]:.3f}")

        n_p1, n_p2 = len(param1_range), len(param2_range)
        grids = {k: np.full((n_p1, n_p2), np.nan)
                 for k in ('fwd_m', 'fwd_s', 'rev_m', 'rev_s', 'asym_m', 'asym_s')}

        for task_idx, data in results:
            i = task_idx // n_p2
            j = task_idx %  n_p2
            grids['fwd_m'][i, j],  grids['fwd_s'][i, j]  = data['fwd']
            grids['rev_m'][i, j],  grids['rev_s'][i, j]  = data['rev']
            grids['asym_m'][i, j], grids['asym_s'][i, j] = data['asym']

        self.results_data = {
            'param1_name':    param1_name,
            'param1_range':   param1_range.tolist(),
            'param2_name':    param2_name,
            'param2_range':   param2_range.tolist(),
            'fixed_overrides': {k: float(v) for k, v in fixed_overrides.items()},
            'grids':          {k: v.tolist() for k, v in grids.items()},
            'metadata': {
                'command_line': ' '.join(sys.argv),
                'n_runs':       n_runs,
                'total_time':   total_time,
                'timestamp':    datetime.datetime.now().isoformat(),
            },
        }

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'sweep_data.json'), 'w') as fh:
            json.dump(self.results_data, fh, indent=4)
        self.plot_results(output_dir)

    # Plots
    def plot_results(self, output_dir):
        res   = self.results_data
        p1    = np.array(res['param1_range'])
        p2    = np.array(res['param2_range'])
        grids = {k: np.array(v) for k, v in res['grids'].items()}

        p1_name = res['param1_name']
        p2_name = res['param2_name']
        p1_desc = PARAM_DEFS.get(p1_name, {}).get('desc', p1_name)
        p2_desc = PARAM_DEFS.get(p2_name, {}).get('desc', p2_name)

        plt.rcParams.update({
            "font.family":     "serif",
            "font.size":       12,
            "axes.labelsize":  14,
            "axes.titlesize":  16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "text.usetex":     False,
        })

        extent = [p2[0], p2[-1], p1[0], p1[-1]]

        def save_heatmap(data_list, titles, cmaps, vm_list, filename, figsize):
            fig, axes = plt.subplots(1, len(data_list), figsize=figsize,
                                     constrained_layout=True)
            if len(data_list) == 1:
                axes = [axes]
            for ax, data, title, cmap, vms in zip(axes, data_list, titles, cmaps, vm_list):
                im = ax.imshow(data, origin='lower', extent=extent,
                               aspect='auto', cmap=cmap, vmin=vms[0], vmax=vms[1])
                fig.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title(title, fontweight='bold', pad=10)
                ax.set_xlabel(p2_desc)
            axes[0].set_ylabel(p1_desc)
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()

        save_heatmap(
            [grids['fwd_m'], grids['rev_m'], grids['asym_m']],
            [r'Forward Coherence $\langle C_\mathrm{fwd} \rangle$',
             r'Reverse Coherence $\langle C_\mathrm{rev} \rangle$',
             r'Asymmetry $\langle \Delta C \rangle$'],
            ['viridis', 'viridis', 'RdBu_r'],
            [(0, 1), (0, 1), (-0.5, 0.5)],
            'coherence_asymmetry.png', (18, 5),
        )
        save_heatmap(
            [grids['fwd_s'], grids['rev_s'], grids['asym_s']],
            [r'SEM($C_\mathrm{fwd}$)', r'SEM($C_\mathrm{rev}$)', r'SEM($\Delta C$)'],
            ['magma', 'magma', 'magma'],
            [(0, None), (0, None), (0, None)],
            'standard_error.png', (18, 5),
        )
        print(f"\n[INFO] Plots saved to: {output_dir}")


# CLI
def _print_param_table():
    print("\nAvailable parameters for --param1 / --param2 / --set:\n")
    print(f"  {'Name':<14} {'Type':<10} {'Description'}")
    print(f"  {'-'*14} {'-'*10} {'-'*42}")
    for name, p in PARAM_DEFS.items():
        kind = 'composite' if 'special' in p else f"index {p['idx']}"
        print(f"  {name:<14} {kind:<10} {p['desc']}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # swept parameters
    sweep = parser.add_argument_group('Swept parameters (required)')
    sweep.add_argument('--param1',     required=True, metavar='NAME',
                       help='First swept parameter (see --list_params)')
    sweep.add_argument('--param1_min', required=True, type=float, metavar='V')
    sweep.add_argument('--param1_max', required=True, type=float, metavar='V')
    sweep.add_argument('--n_param1',   required=True, type=int,   metavar='N',
                       help='Number of grid points along param1')
    sweep.add_argument('--param2',     required=True, metavar='NAME',
                       help='Second swept parameter')
    sweep.add_argument('--param2_min', required=True, type=float, metavar='V')
    sweep.add_argument('--param2_max', required=True, type=float, metavar='V')
    sweep.add_argument('--n_param2',   required=True, type=int,   metavar='N',
                       help='Number of grid points along param2')

    # overrides
    parser.add_argument('--set', nargs=2, metavar=('PARAM', 'VALUE'),
                        action='append', default=[],
                        help='Fix a parameter at VALUE for the whole sweep '
                             '(can be repeated, e.g. --set d_avg 7.5 --set F0 2.5)')

    # simulation settings
    sim = parser.add_argument_group('Simulation settings')
    sim.add_argument('--cores', type=int,   default=4,
                     help='Number of parallel worker processes')
    sim.add_argument('--runs',  type=int,   default=5,
                     help='Ensemble runs per grid point')
    sim.add_argument('--time',  type=int,   default=100,
                     help='Simulation duration (s)')
    sim.add_argument('--name',  type=str,   default='sweep',
                     help='Label appended to the output directory name')

    # engine/template
    eng = parser.add_argument_group('Engine and template')
    eng.add_argument('--exe',      type=str, default='./src/rowers',
                     help='Path to the simulation executable')
    eng.add_argument('--template', type=str,
                     default='./templates/6beads.input',
                     help='Template .input file (provides all fixed physics parameters)')

    # utility
    parser.add_argument('--list_params', action='store_true',
                        help='Print the table of sweepable parameters and exit')

    if '--list_params' in sys.argv:
        _print_param_table()
        sys.exit(0)

    args = parser.parse_args()

    # v alidate parameter names
    unknown = [n for n in (args.param1, args.param2) if n not in PARAM_DEFS]
    if unknown:
        parser.error(f"Unknown parameter name(s): {unknown}. "
                     f"Run with --list_params to see valid names.")
    if args.param1 == args.param2:
        parser.error("--param1 and --param2 must be different parameters.")

    # parse fixed overrides
    fixed_overrides = {}
    for name, value in args.set:
        if name not in PARAM_DEFS:
            parser.error(f"Unknown parameter '{name}' in --set. "
                         f"Run with --list_params to see valid names.")
        fixed_overrides[name] = float(value)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = f"results_{timestamp}_{args.name}"

    param1_range = np.linspace(args.param1_min, args.param1_max, args.n_param1)
    param2_range = np.linspace(args.param2_min, args.param2_max, args.n_param2)

    runner = BeadAsymmetrySweep(executable=args.exe, template_input=args.template)
    runner.run_sweep_ensemble(
        param1_name    = args.param1,
        param1_range   = param1_range,
        param2_name    = args.param2,
        param2_range   = param2_range,
        fixed_overrides = fixed_overrides,
        num_cores      = args.cores,
        n_runs         = args.runs,
        total_time     = args.time,
        output_dir     = out_dir,
    )
