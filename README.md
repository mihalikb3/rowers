# Rowers

Simulation of hydrodynamically coupled oscillators (cilia-like "rowers") to study directional asymmetry in bead arrays. A C++ engine runs the Langevin dynamics; Python scripts drive parameter sweeps and optimisation.

## 1. Build the simulation engine

The engine is written in C++ and needs to be compiled once before anything else. On most Linux servers a compiler is already installed. From the repo root:

```bash
cd src
make
cd ..
```

This downloads a header-only linear algebra library (Eigen) automatically and produces the executable `src/rowers`. You only need to do this once, or again after pulling updates to the C++ code.

## 2. Running a single simulation directly

The engine takes an input file and writes its output to a directory with the same name. The basic command is:

```bash
./src/rowers -C -R templates/6beads.input
```

The first flag sets the geometry and the second sets the hydrodynamic tensor.

Geometry options:
- `-C` — **chain**: beads in a line, the usual case for cilia-like systems
- `-R` — **ring**: beads arranged in a closed loop (default if no flag given)
- `-A` — **arbitrary**: bead positions read directly from the input file, for any custom arrangement (not yet tested)

Tensor options:
- `-R` — **Rotne-Prager-Yamakawa** (recommended): Contains short-range corrections. This guarantees the mobility matrix is positive definite for any bead configuration, which is required for the Langevin dynamics to remain physically valid.
- `-O` — **Oseen**: Point-particle approximation, faster but unphysical at short range and can make the mobility matrix indefinite when beads are close.

After the run you will find a directory called `6beads/` containing three files:

- `*.dat` — bead positions over time
- `*.trap` — trap positions over time
- `*.forces` — forces over time

**Input files** are plain text with one simulation per line. Each line is a space-separated list of parameters in a fixed order. See section 7 for the full parameter reference.

You can run multiple input files in one command, the engine processes them sequentially:

```bash
./src/rowers -C -R run1.input run2.input run3.input
```

In practice the Python scripts create and manage input files automatically, so you rarely need to edit them by hand. The templates in `templates/` serve as the starting point that the scripts modify.

## 3. Python environment

The scripts require NumPy, SciPy, Matplotlib, and scikit-learn. Install them into a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib scikit-learn
```

Activate the environment (`source .venv/bin/activate`) at the start of each session.

## 4. Run a parameter sweep

All scripts are run from the **repo root** (not from inside `analysis/`).

The sweep script measures directional asymmetry across a 2D grid of any two simulation parameters. You must specify both axes explicitly:

```bash
python analysis/sweep.py \
    --param1 d_grad   --param1_min -1.0 --param1_max 1.0  --n_param1 21 \
    --param2 T_signal --param2_min  0.2 --param2_max 0.5  --n_param2 21 \
    --set d_avg 7.5 \
    --cores 8 --runs 5 --time 100 --name my_sweep
```

To see all available parameters:

```bash
python analysis/sweep.py --list_params
```

Results (heatmap PNGs and a JSON data file) are saved to a timestamped directory.

Key options:

| Flag | Description |
|---|---|
| `--param1/2` | Parameter to sweep (see `--list_params`) |
| `--param1/2_min/max` | Range of values |
| `--n_param1/2` | Number of grid points |
| `--set NAME VALUE` | Fix a parameter at a value (repeatable) |
| `--cores` | Parallel workers (blade1 has 48 cores) |
| `--runs` | Ensemble repeats per grid point (more = lower noise) |
| `--time` | Simulation duration in seconds |
| `--template` | Input template file (default: `templates/6beads.input`) |

## 5. Optimisation

Two optimisation scripts search for the bead geometry and driving parameters that maximise asymmetry.

**Metropolis–Hastings**:
```bash
python analysis/optimisation_metropolis.py \
    --chain_length 40.0 --iterations 100 --cores 8
```

**Bayesian optimisation**:
```bash
python analysis/optimisation_bayesian.py \
    --chain_length 40.0 --n_iterations 60 --cores 8
```

Both scripts save a history JSON and diagnostic plots at every checkpoint.

## 6. Other scripts

`analysis/check_natural_frequency.py` — measures the natural oscillation frequency of the chain as a function of geometry. Useful for choosing a sensible `T_signal` range before running a sweep.

## 7. Input file parameter reference

Input files are plain text. Each line defines one simulation as a space-separated list of values in a fixed order. The templates in `templates/` are the starting point; the Python scripts modify individual fields and pass the result to the engine.

The two provided templates are both chain geometries:
- **6beads.input** — 6 beads with non-uniform spacing (6.0, 7.1, 8.0, 9.0, 9.9 µm), encoding an asymmetric geometry from previous optimisation work.
- **8beads.input** — 8 beads with uniform spacing (7.5 µm per gap), a neutral symmetric starting point.

The full parameter list, in the order they appear in the file:

| Field | Name | Template value | Description |
|---|---|---|---|
| 1 | `N` | 6 | Number of beads. Change by using a different template. |
| 2 | `amplitude` | 3 | Trap half-width (µm) — the spatial extent of each bead's oscillation. |
| 3 | `base_distance` | 6.0 | Reference inter-bead distance (µm), used for ring geometry and for normalising the position-dependent stiffness in chain mode. |
| 4 | `bead_radius` | 1.5 | Bead radius (µm), enters the hydrodynamic mobility tensor. |
| 5 | `dt` | 0.00005 | Integration time step (s). Reduce if the simulation becomes numerically unstable. |
| 6 | `totaltime` | 30 | Total simulation duration (s). ★ |
| 7 | `viscosity` | 0.006 | Fluid viscosity (Pa*s). |
| 8 | `alpha` | 0.5 | Exponent of the x-direction trap force. |
| 9 | `kx` | 10 | x-trap stiffness prefactor. |
| 10 | `beta` | 1.5 | Exponent of the y-direction trap force (same role as alpha, for y). |
| 11 | `ky` | 10 | y-trap stiffness prefactor. |
| 12 | `temperature` | 70 | Effective temperature for thermal noise (sets noise amplitude via k_B T). Set to 0 to run without noise. ★ |
| 13 | `sampling_feedback` | 50 | Steps between trap-switching checks. Controls the feedback loop timing; rarely needs changing. |
| 14 | `sampling_write` | 200 | Steps between writing to output files. Increase to reduce output file size. |
| 15 | `epsilon` | 0.3 | Small gap (µm) added to the trap boundary when switching direction, to prevent immediate re-switching. |
| 16 | `velox` | 0.0 | Background flow velocity in x (µm/s). |
| 17 | `ks` | 0 | Inter-bead repulsion spring constant. 0 = disabled. |
| 18 | `fl0` | 0.25 | Natural length for the inter-bead spring (µm), used only if ks > 0. |
| 19 | `veloy` | 0.0 | Background flow velocity in y (µm/s). |
| 20 | `F0` | 2.5 | Signal force amplitude. Square-wave drive applied to the signal beads (see below) after the 5 s transient. ★ |
| 21 | `Tsignal` | 0.1847 | Signal period (s). ★ |
| 22 | `default_lambda` | 0.06 | Default per-bead coupling coefficient λ, which controls how trap stiffness varies with position in the chain. |
| 23… | distances | 6.0 7.1 … | N−1 inter-bead distances (µm), one per gap. The key geometry parameter. ★ |
| last N | lambdas | 0.06 × N | Per-bead λ values, one per bead. Overrides `default_lambda` individually if needed. |

★ Parameters you are most likely to want to change. All others can be left at their template values unless you have a specific reason to modify them.

### Optional per-bead configuration blocks

Three optional blocks can be appended after the lambda values to specify heterogeneous trap stiffness and control which beads receive the signal. All three are absent from the provided templates, in which case the engine falls back to the global `kx`/`ky` for every bead and applies the signal to bead 0 only.

The blocks must appear in this order if used, and each block requires the preceding one to be present:

**Block 1 — per-bead x-stiffness** (N values):
```
<kx_0> <kx_1> ... <kx_{N-1}>
```
Replaces the global `kx` for each bead individually. The value is still scaled by the same distance-based H factor used in the uniform case, so relative heterogeneity combines with the geometry-driven correction.

**Block 2 — per-bead y-stiffness** (N values, requires Block 1):
```
<ky_0> <ky_1> ... <ky_{N-1}>
```
Same as Block 1 but for the y direction.

**Block 3 — signal bead list** (requires Blocks 1 and 2):
```
<n_signal> <bead_0> <bead_1> ... <bead_{n-1}>
```
The first value is the count of beads that receive the square-wave drive, followed by their 0-based indices. For example, `2 0 3` applies the force to beads 0 and 3.

**Example** — 6-bead chain with increasing x-stiffness, uniform y-stiffness, and the signal applied to beads 0 and 5:
```
6 3 6.0 1.5 0.00005 100 0.006 0.5 10 1.5 10 70 50 200 0.3 0.0 0 0.25 0.0 2.5 0.18 0.06  6.0 7.1 8.0 9.0 9.9  0.06 0.06 0.06 0.06 0.06 0.06  8 9 10 11 12 13  10 10 10 10 10 10  2 0 5
```

When running a sweep or optimisation, if per-bead blocks are present in the template the scripts handle them automatically: sweeping `kx` or `ky` updates all per-bead values uniformly, and the mirror-geometry runs reverse the per-bead arrays along with the inter-bead distances.
