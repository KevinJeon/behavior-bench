# PufferDrive Benchmark Splits

This document describes how to reproduce the **Interactive1k** (`pufferinter`) and **Random1k** (`pufferrandom`) benchmark splits from the Waymo Open Motion Dataset (WOMD).

The two splits are defined by CSV manifests checked into the repo at:

- [pufferlib/resources/splits/interactive1k.csv](pufferlib/resources/splits/interactive1k.csv)
- [pufferlib/resources/splits/random1k.csv](pufferlib/resources/splits/random1k.csv)

Each manifest references 1000 scenarios from the WOMD validation set. The pipeline below produces a directory of binary scenario files matching one of these manifests, ready to be consumed by the evaluation harness.

---

## Step 1 — Waymo JSONs → Validation Binaries

The full WOMD validation set is converted into PufferDrive binaries with lane connectivity. The pipeline takes two inputs side by side:

- **Raw Waymo TFRecords** — needed only for lane connectivity extraction. Available from the [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/) (registration required) under `uncompressed/scenario/{training,validation,testing}/`.
- **GPUDrive-style JSON scenarios** — one JSON per scenario, used to build the binary trajectories and road graph.

### 1a. Obtaining the GPUDrive JSONs

To reproduce the benchmark you need a GPUDrive-style JSON for **every** scenario in the WOMD validation set. Generate them from the raw Waymo TFRecords using [ScenarioMax](https://github.com/valeoai/ScenarioMax), which converts WOMD (and nuPlan, Argoverse, …) into the GPUDrive JSON format:

```bash
git clone https://github.com/valeoai/ScenarioMax.git
cd ScenarioMax
pip install -e .

# Convert WOMD TFRecords → GPUDrive JSONs (see ScenarioMax docs for full options)
python -m scenariomax.convert \
  --source waymo \
  --input /path/to/waymo/uncompressed/scenario \
  --output /path/to/gpudrive \
  --format gpudrive
```

> The pre-built [`EMERGE-lab/GPUDrive`](https://huggingface.co/datasets/EMERGE-lab/GPUDrive) HuggingFace dataset (downloadable via [data_utils/womd/download_womd_data.py](data_utils/womd/download_womd_data.py)) is only a subset of WOMD and **does not cover the full validation set** — it cannot be used to reproduce Interactive1k / Random1k. Use ScenarioMax for the benchmark.

### 1b. JSONs → Binaries

Run [data_utils/womd/create_training_binaries.py](data_utils/womd/create_training_binaries.py) to extract lane connectivity from the TFRecords, enrich the JSONs, and convert to `.bin` files:

```bash
python data_utils/womd/create_training_binaries.py \
  --tfrecord-root /path/to/waymo \
  --json-root /path/to/gpudrive \
  --enriched-root /path/to/gpudrive_with_connectivity \
  --output-root /path/to/binaries \
  --workers 80
```

Expected input layout:

```
/path/to/waymo/{training,validation,testing}/      # raw Waymo TFRecords
/path/to/gpudrive/{training,validation,testing}/   # GPUDrive JSONs (Option A or B)
```

Output:

```
/path/to/binaries/validation/
  map_000000.bin
  map_000001.bin
  ...
```

The filenames here (`map_NNNNNN.bin`) are the `original_filename` values referenced by the split manifests.

> If you already have GPUDrive JSONs with lane connectivity baked in and only need the JSON → binary step, use `process_all_maps()` from [pufferlib/ocean/drive/drive.py](pufferlib/ocean/drive/drive.py) directly.

---

## Step 2 — Validation Binaries → Benchmark Split

Use [scripts/remap_files.py](scripts/remap_files.py) together with one of the manifest CSVs to create a split.

```bash
python scripts/remap_files.py \
  --source-path /path/to/binaries/validation \
  --csv-path    pufferlib/resources/splits/interactive1k.csv \
  --target-path /path/to/test_eval
```

The script:

1. Reads `interactive1k.csv` (columns: `new_filename, original_filename, ego_agent_idx, interactivity_score, map_id`).
2. For each row, copies `<source-path>/<original_filename>` to `<target-path>/<csv_stem>/<new_filename>`.
3. Copies the CSV itself to the output directory as `manifest.csv`.

Resulting layout:

```
/path/to/test_eval/interactive1k/
  map_000000.bin       # was map_032035.bin in the full validation set
  map_000001.bin       # was map_005373.bin
  ...
  map_000999.bin
  manifest.csv
```

For the random split, swap in `random1k.csv` — the output directory will be named after the CSV stem (`random1k/`).

---

## Using the Split in Evaluation

Once the split directory exists, point `DRIVE_BINARIES_DATA_ROOT` at its parent and run:

```bash
export DRIVE_BINARIES_DATA_ROOT=/path/to/test_eval

python pufferlib/ocean/benchmark/eval.py \
  --eval.split interactive1k --map-ids all
```

The split name passed to `--eval.split` must match the directory name produced in Step 2 (i.e. the CSV stem).
