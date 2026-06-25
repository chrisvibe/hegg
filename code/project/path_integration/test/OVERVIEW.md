# Path Integration Test Overview

## What is being tested
Path integration: given velocity inputs over S steps, predict final position.
Non-linearity arises when the boundary **clips position but not velocity** — the model
must internally integrate position and detect wall hits. This is the hard boundary case.

## Linearity axis (5 named profiles in `config/{1D,2D}/grid_search/linearity/library.yaml`)

| universe             | boundary      | baseline | regime                        |
|----------------------|---------------|----------|-------------------------------|
| mother               | none          | 100%     | pure integration, no boundary |
| `soft_boundary`      | soft (k=0.5)  | 100%     | **current production config** |
| `hard_boundary_low_acc` | hard, low acc | ~99%  | transition point              |
| `hard_boundary`      | hard, full acc| ~9–14%   | clearly non-linear            |
| `displacement`       | any           | 100%     | always linear (telescoping)   |

Production configs (`config/path_integration/*/design_choices/`) use `soft_boundary`.
The `old_non_linear_boundary/` subfolders preserve the previous hard-boundary configs.

## Normalization and thresholds
- **y** is z-scored (`StandardScaler`) in `project/path_integration/code/dataset_init.py`.
  Min-max was rejected because the soft boundary concentrates paths near centre,
  compressing most probability mass into the middle of [0,1].
- **Continuous threshold**: 0.1σ (set in all `continuous*.yaml` configs)
- **Discrete threshold**: 0.2σ — brackets the output grid:
  1D half-step = 0.125σ < **0.2σ** < 0.25σ = nearest wrong bin
  2D half-diagonal = 0.177σ < **0.2σ** < 0.25σ = nearest wrong bin

## Key files

| file | purpose |
|------|---------|
| `test_linearity.py` | pytest suite (2 tests: 1D/2D near-linear assertion); `story()` narrative; `visualize_paths(dim, n, universe)` demo tool |
| `linearity_probe.py` | CLI tool — evaluate or sweep any config against the linear baseline |
| `test_dataset_properties.py` | automated checks: path spread, boundary hit rate, velocity/displacement divergence |
| `config/{1D,2D}/.../library.yaml` | named dataset profiles with documented baselines |

## Common commands
```bash
# Check if current production config is linear
python project/path_integration/test/linearity_probe.py \
    config/path_integration/2D/grid_search/design_choices/continuous.yaml

# Find the linear↔non-linear transition
python project/path_integration/test/linearity_probe.py \
    project/path_integration/test/config/2D/grid_search/linearity/library.yaml \
    --universe hard_boundary --n-probe 12

# View path samples for a profile
python project/path_integration/test/test_linearity.py --visualize --universe hard_boundary
# → /out/demo/profiles/2D/hard_boundary/

# Run all property tests
pytest project/path_integration/test/test_dataset_properties.py -v
pytest project/path_integration/test/test_linearity.py -v
```
