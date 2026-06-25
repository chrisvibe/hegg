# Boolean Reservoir Computing with Random Boolean Networks

> **Paper:** *[Title]* — [Authors] — ALIFE 2026
> **Hosted dashboards:** https://work.tubiformis.work/hegg/dashboard/

A research framework for **Reservoir Computing (RC)** using **Random Boolean Networks (RBNs)** as the dynamical reservoir. The RBN projects inputs into a high-dimensional binary state space; a simple linear readout (ridge regression) is trained on top. No backpropagation through the reservoir is needed.

**No GPU required.** Reservoir dynamics are boolean — computation runs on CPU via a Julia engine. Everything else is NumPy.

---

## Quick Start

```bash
# Generate grid search data (article configs)
python code/project/temporal/code/train_model.py config/temporal/density/grid_search/design_choices/all_heterogeneous.yaml
python code/project/path_integration/code/train_model.py config/path_integration/1D/grid_search/design_choices/continuous.yaml

# Explore results locally
python code/dashboard/run.py temporal          # http://localhost:8050
python code/dashboard/run.py path_integration
python code/dashboard/run.py polar --port 8051
```

Or browse the results we host: https://work.tubiformis.work/hegg/dashboard/

---

## Architecture

```
code/
├── benchmark/      # Shared infrastructure: physics, datasets, parameter models
├── project/
│   ├── boolean_reservoir/   # Core library — BooleanReservoir, LUT, encoding, graph
│   ├── path_integration/    # Path integration task
│   ├── temporal/            # Temporal density and parity tasks
│   └── parallel_grid_search/  # Multi-process grid search runner (submodule)
├── config/         # YAML experiment configurations
└── dashboard/      # Plotly/Dash visualisation apps
```

Results land in `out/` (parquet), datasets in `data/` (npz, auto-generated on first run).

---

## Tasks

### Path Integration
The reservoir receives velocity or displacement steps and must predict the agent's final position. Trajectory physics include momentum, quadratic drag, and boundary forces (Verlet integration).

- **Displacement** (null test): trivial sum — tests memory only
- **Velocity** (primary test): genuine integration through nonlinear physics

### Temporal Tasks
Binary bit-stream tasks on a sliding window:
- **Density**: more 1s than 0s in the window?
- **Parity**: odd number of 1s?

### Capacity Metrics (KQ / GR)
Task-independent reservoir diagnostics:
- **KQ** (Kernel Quality): fully random inputs — measures input separability
- **GR** (Generalization Rank): inputs with `τ` shared bits — measures fading memory
- **KQGR = KQ − GR**: proxy for the compute/memory trade-off

---

## Configuration

YAML configs map 1:1 to Pydantic `Params` models. Any field set to a list triggers a cartesian-product grid search:

```yaml
model:
  variables:
    R_k_avg: [1, 2, 3, 4, 5, 6, 7, 8]
    R_mode: [homogeneous, heterogeneous]
```

Article configs are in `code/config/` under `temporal/` and `path_integration/`.

---

## Installation

### Option A — DevContainer (recommended)

Requires [VS Code](https://code.visualstudio.com/) and [Docker](https://docs.docker.com/get-started/).

```bash
docker pull chrisvibe/boolean_reservoir:4.1
```

Open the repo in VS Code → *Reopen in Container*. The image (~6 GB) includes all dependencies and the Julia engine.

To build locally:
```bash
cd docker && docker build -t boolean_reservoir:4.1 .
```

### Option B — Conda

```bash
git clone --recursive https://github.com/chrisvibe/hegg.git
cd hegg
conda env create -f docker/src/environment.lock.yaml
conda activate boolean_reservoir
export PYTHONPATH="$PWD/code:$PWD/code/dashboard"
```

**Dependencies:** Python 3.12, NumPy, SciPy, scikit-learn, pandas, pyarrow, NetworkX, Pydantic 2, Plotly, Dash, pyjuliacall.

---

## Reproducibility

All random structures (graph topology, LUTs, initial states) are seeded via NumPy. Ridge regression is closed-form. No GPU, no stochastic training.

- `R.seed` → reservoir topology and LUTs
- `D.seed` → dataset generation
