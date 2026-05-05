# SMART Prediction Model

PufferDrive includes an autoregressive trajectory prediction model based on SMART (arxiv 2405.15677). This model predicts future trajectories for all agents in a driving scenario by tokenizing motion and map data and applying transformer-based decoding.

---

## Architecture Overview

### Motion Tokens

A codebook of **2048 pre-computed trajectory templates** represents short motion segments. Each ground-truth trajectory is matched to the nearest token in this codebook, converting continuous trajectories into a discrete token sequence. This enables autoregressive prediction over a finite vocabulary.

### Map Tokens

A codebook of **1024 road polyline templates** represents road geometry. Road segments from the scenario are matched to map tokens, providing a compact representation of the road network for the model.

### Token Sequence Structure

Each scenario spans 91 timesteps. These are compressed into **18 token positions** using a shift of 5 (each token covers 5 timesteps). The sequence is split as:

- **First 2 token positions** - History context (observed past trajectory)
- **Remaining 16 token positions** - Predicted future trajectory

During inference, the model autoregressively generates the 16 future tokens conditioned on the 2 history tokens and the map context.

---

## Model Components

All model code lives in `pufferlib/prediction/smart/`:

| File | Description |
|---|---|
| `prediction_model.py` | Main wrapper that ties together the full prediction pipeline |
| `smart_decoder.py` | Orchestrates the map encoder and agent decoder into a single forward pass |
| `agent_decoder.py` | Temporal self-attention over agent token sequences + cross-attention to map tokens |
| `map_decoder.py` | Encodes road tokens via self-attention to produce map context embeddings |
| `attention_layer.py` | Multi-head graph attention implementation shared across encoder and decoder |
| `fourier_embedding.py` | Sinusoidal (sin/cos) positional embeddings for time and space |

### Data Pipeline

Data loading and tokenization code lives in `pufferlib/prediction/`:

| File | Description |
|---|---|
| `dataset.py` | `WaymoBinaryDataset` - loads `.bin` scenario files and converts them to `HeteroData` graph objects |
| `binary_reader.py` | Low-level binary parsing of PufferDrive's `.bin` scenario format |
| `trajectory_tokenizer.py` | Matches continuous trajectories to the motion codebook (2048 tokens) |
| `map_tokenizer.py` | Matches road polylines to the map codebook (1024 tokens) |

---

## Quick Start

### Step 1: Build the Prediction Cache

Before training, pre-process the binary scenario files into cached tokenized format. This step is required and significantly speeds up training by avoiding repeated tokenization.

```bash
python scripts/build_prediction_cache.py \
  --config pufferlib/config/prediction/smart.ini \
  --splits training validation \
  --num-workers 32
```

This reads all `.bin` files from the configured data directories, tokenizes trajectories and maps, and writes cached `.pt` files. Adjust `--num-workers` based on available CPU cores.

For a quick test on a small subset:

```bash
python scripts/build_prediction_cache.py \
  --config pufferlib/config/prediction/smart_overfit.ini \
  --splits training \
  --num-workers 4
```

### Step 2: Train the Model

**Single-GPU training:**

```bash
python -m pufferlib.prediction.puffer_prediction pretrain \
  --config pufferlib/config/prediction/smart.ini
```

**Multi-GPU training with torchrun (distributed data parallel):**

```bash
torchrun --nproc_per_node=8 -m pufferlib.prediction.puffer_prediction pretrain \
  --config pufferlib/config/prediction/smart.ini
```

This distributes training across 8 GPUs on a single node. Each GPU processes a shard of the data.

**Train the small model variant (faster, good for prototyping):**

```bash
python -m pufferlib.prediction.puffer_prediction pretrain \
  --config pufferlib/config/prediction/smart_1m.ini
```

**Debug/overfit on a tiny dataset:**

```bash
python -m pufferlib.prediction.puffer_prediction pretrain \
  --config pufferlib/config/prediction/smart_overfit.ini
```

This uses only 4 scenario files, which is useful for verifying the training loop works end-to-end before committing to a full run.

### Step 3: Evaluate (WOSAC Realism Metrics)

After training, evaluate trajectory realism using the Waymo Open Sim Agents Challenge protocol:

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
  --planner.type smart \
  --planner.smart.weights-path path/to/weights.pt \
  --map-ids 0-228
```

This runs the trained SMART model as a planner on validation scenarios and computes WOSAC realism scores. The `--map-ids` argument specifies which scenarios to evaluate (0-228 covers the full validation set).

To evaluate on a subset:

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
  --planner.type smart \
  --planner.smart.weights-path path/to/weights.pt \
  --map-ids 0-49
```

---

## Configuration

All prediction configs live in `pufferlib/config/prediction/`. Each is an INI file with three main sections.

### Config Variants

| Config File | Description |
|---|---|
| `smart.ini` | Full model - hidden_dim=128, 6 agent decoder layers, 8 attention heads |
| `smart_1m.ini` | Small model - hidden_dim=64, 4 agent decoder layers, 4 attention heads (roughly 1M parameters) |
| `smart_overfit.ini` | Debug config - only loads 4 scenario files, useful for testing the pipeline |

### `[data]` Section

Controls data loading paths, splits, caching, and preprocessing parameters:

- Path to binary scenario directories
- Which splits to use (training, validation)
- Cache directory location
- Number of data loading workers

### `[model]` Section

Controls model architecture:

- `hidden_dim` - Embedding and hidden layer dimensionality
- Number of encoder/decoder layers
- Number of attention heads
- Dropout rates
- Codebook sizes (motion tokens, map tokens)

### `[train]` Section

Controls training hyperparameters:

- Learning rate and schedule
- Batch size
- Number of epochs
- Gradient clipping
- Checkpoint frequency
- Distributed training settings

---

## End-to-End Example

Here is a complete workflow from raw data to evaluated predictions:

```bash
# 1. Build cache (do this once)
python scripts/build_prediction_cache.py \
  --config pufferlib/config/prediction/smart.ini \
  --splits training validation \
  --num-workers 32

# 2. Train on 8 GPUs
torchrun --nproc_per_node=8 -m pufferlib.prediction.puffer_prediction pretrain \
  --config pufferlib/config/prediction/smart.ini

# 3. Evaluate realism on validation scenarios
python pufferlib/ocean/benchmark/eval_realism.py \
  --planner.type smart \
  --planner.smart.weights-path experiments/smart_20240101/best_model.pt \
  --map-ids 0-228
```
