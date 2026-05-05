# Benchmark: Interactivity Score

This module extracts and evaluates scenarios from binary scenario files based on an **Interactivity Score**.

## Interactivity Score Formula

```
Score = 0.20 × agent_score
      + 0.25 × accel_score
      + 0.15 × steering_score
      + 0.20 × density_score
      + 0.20 × goal_dist_score
```

### Components

| Component | Formula | Range | Description |
|-----------|---------|-------|-------------|
| `agent_score` | `min(num_agents / 15, 1)` | 0-1 | Number of agents with goal ≥ 10m |
| `accel_score` | `min(avg_accel_change / 30, 1)` | 0-1 | Average acceleration changes |
| `steering_score` | `min(avg_steering_change / 2, 1)` | 0-1 | Average steering changes (rad) |
| `density_score` | `min(avg_agents_in_10m / 3, 1)` | 0-1 | Average agents within 10m radius |
| `goal_dist_score` | `min((ego_goal - 10) / 40, 1)` | 0-1 | Ego goal distance (10m→0, 50m→1) |

### Filter Criteria

A scenario is **skipped** if:
- Ego goal distance < 10m
- Less than 5 agents visible at t=0 (valid + goal ≥ 10m)

## Usage

### Calculate and display scores

```bash
python -m pufferlib.ocean.benchmark.extract_benchmark \
    --split /path/to/binaries/training \
    --limit 1000
```

### Extract top N scenarios

```bash
python -m pufferlib.ocean.benchmark.extract_benchmark \
    --split /path/to/binaries/training \
    --extract 100 \
    --extract-dir /path/to/output/
```

### Generate plots

```bash
python -m pufferlib.ocean.benchmark.extract_benchmark \
    --split /path/to/binaries/training \
    --plot \
    --plot-top 20 \
    --plot-dir scenario_plots/
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--split` | Path to directory with binary files | (required) |
| `--output` | CSV output file for scores | `interactivity_scores_<split>.csv` |
| `--limit` | Maximum number of files to process | all |
| `--min-goal-dist` | Minimum goal distance for ego and filtering | 10.0 |
| `--extract` | Number of top scenarios to extract | - |
| `--extract-dir` | Target directory for extracted scenarios | - |
| `--plot` | Generate plots for scenarios | False |
| `--plot-dir` | Directory for plots | `scenario_plots/` |
| `--plot-top` | Only plot top N scenarios | all |
| `--sort-by` | Sort results by column | `interactivity_score` |

## Extraction Process

When using `--extract N --extract-dir /path/`, the script:

1. **Processes all binary files** in the source directory
2. **Filters scenarios** based on criteria (ego goal ≥ 10m, ≥5 visible agents at t=0)
3. **Computes interactivity scores** for all valid scenarios
4. **Sorts by score** (highest first)
5. **Copies top N binary files** to the target directory
6. **Creates a manifest.csv** in the target directory containing scores and metrics for all extracted scenarios

The extracted files are exact copies of the original binary files, preserving all trajectory and map data. The manifest allows you to quickly see which scenarios were selected and their scores.

## Output

### CSV File

Contains for each scenario:
- `filename`: Name of the binary file
- `interactivity_score`: Overall score (0-1)
- `num_agents`: Filtered agents (goal ≥ 10m)
- `agents_valid_t0`: Agents visible at t=0
- `ego_goal_distance`: Goal distance of ego in meters
- `avg_agents_in_radius`: Average agents within 10m radius
- `avg_accel_change`: Average acceleration changes
- `avg_steering_change`: Average steering changes
- `goal_dist_score`: Normalized goal distance score

### Plots

Each plot shows:
- Road network (gray)
- Agents (blue)
- **Ego** (red star + red rectangle)
- **Ego Goal** (green star + dashed line)
- Title with score and metrics

## Example

```bash
# Extract top 50 most interactive scenarios from training
python -m pufferlib.ocean.benchmark.extract_benchmark \
    --split /path/to/binaries/training \
    --extract 50 \
    --extract-dir ./interactive_scenarios/ \
    --plot \
    --plot-top 10
```

This will:
1. Process all scenarios in the training directory
2. Extract the 50 most interactive scenarios to `./interactive_scenarios/`
3. Create `./interactive_scenarios/manifest.csv` with all scores
4. Generate plots for the top 10 scenarios in `scenario_plots/`
