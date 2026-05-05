# SMART Prediction Model

Trajectory prediction model based on [SMART](https://github.com/rainmaker22/SMART) (arXiv:2405.15677). Predicts future motion tokens for all agent types (vehicles, pedestrians, cyclists) using a spatial-temporal heterogeneous graph network.

## Architecture

- **Input**: Per-scenario graph with agents (vehicles + VRUs) and tokenized road polylines
- **Agent types**: Vehicles (type 0), Pedestrians (type 1), Cyclists (type 2) — each with separate motion token codebooks (2048 tokens per type) and token embedding MLPs
- **Graph edges**: Temporal (same agent across time), Agent-to-Agent (radius 60m), Map-to-Agent (radius 30m)
- **Output**: Next motion token logits (2048 classes) per agent per time step
- **Prediction targets**: Max 32 agents per scene, closest to ego (matching original SMART)

## Quick Start

### 1. Build Cache (recommended)

Pre-processes all `.bin` scenarios into cached `.pt` files using parallel workers. This makes epoch 1 as fast as subsequent epochs.

```bash
python -m pufferlib.prediction.puffer_prediction warmup-cache \
    --config pufferlib/config/prediction/smart_1m.ini \
    --workers 80
```

### 2. Train

```bash
# Single GPU
python -m pufferlib.prediction.puffer_prediction pretrain \
    --config pufferlib/config/prediction/smart_1m.ini

# Multi-GPU
torchrun --nproc_per_node=8 -m pufferlib.prediction.puffer_prediction pretrain \
    --config pufferlib/config/prediction/smart_1m.ini

# Resume from checkpoint
python -m pufferlib.prediction.puffer_prediction pretrain \
    --config pufferlib/config/prediction/smart_1m.ini \
    --resume experiments/prediction/smart_1M/epoch_050.pt
```

### 3. Fine-tune on RL rollouts

```bash
python -m pufferlib.prediction.puffer_prediction finetune \
    --config pufferlib/config/prediction/smart_1m.ini
```

## Configuration

See `pufferlib/config/prediction/smart.ini` (default, 7M params) and `smart_1m.ini` (~1M params).

Key settings in `[data]`:

| Parameter | Description |
|-----------|-------------|
| `data_dir` | Path to Waymo binary dataset (e.g. `/data/ag_nr/share/binaries_full`) |
| `cache_dir` | Path to store cached `.pt` files. Empty = no caching |
| `max_agents` | Max agents per scene (-1 = no limit). All agents stay in the graph, but max 32 get prediction loss |
| `batch_size` | Scenarios per batch |
| `num_workers` | DataLoader workers |

Key settings in `[model]`:

| Parameter | Default (7M) | 1M variant |
|-----------|-------------|------------|
| `hidden_dim` | 128 | 64 |
| `num_agent_layers` | 6 | 2 |
| `num_map_layers` | 3 | 2 |
| `num_heads` | 8 | 4 |
| `head_dim` | 16 | 16 |
| `num_freq_bands` | 64 | 32 |

## VRU Support

The model handles all three agent types with per-type components:
- **Token codebooks**: `cluster_frame_5_2048.pkl` contains `veh`, `ped`, `cyc` entries (2048 motion tokens each)
- **Token embeddings**: Separate MLPs (`token_emb_veh`, `token_emb_ped`, `token_emb_cyc`)
- **Type embedding**: `nn.Embedding(4, hidden_dim)` differentiates agent types in the graph
- **Shared prediction head**: One MLP outputs 2048 logits for all types; the correct codebook is selected per agent type during token matching and trajectory decoding
