# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Create binaries with lane connectivity for all splits (training, validation, testing).

Efficient pipeline:
1. For each split, process all TFRecords in parallel → extract connectivity per scenario_id
2. Enrich all JSONs with connectivity in parallel (workers share connectivity via fork COW)
3. Convert enriched JSONs to .bin files in parallel

Directory structure:
  Input JSONs:     /path/to/gpudrive/{training,validation,testing}/
  Input TFRecords: /path/to/waymo/{training,validation,testing}/
  Output JSONs:    <enriched-root>/{training,validation,testing}/
  Output Binaries: <output-root>/{training,validation,testing}/

Usage:
    python data_utils/womd/create_training_binaries.py \
        --tfrecord-root /path/to/waymo \
        --json-root /path/to/gpudrive \
        --enriched-root /path/to/gpudrive_with_connectivity \
        --output-root /path/to/binaries \
        --workers 80
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---- Global for fork COW sharing ----
_global_connectivity_map = None


def _enrich_init(connectivity_map):
    global _global_connectivity_map
    _global_connectivity_map = connectivity_map


def _enrich_worker(args):
    json_path, output_path = args
    try:
        with open(json_path) as f:
            data = json.load(f)
        sid = data.get("scenario_id", "")
        connectivity = _global_connectivity_map.get(sid, {})
        if connectivity:
            for road in data.get("roads", []):
                rid = road.get("id")
                if rid in connectivity:
                    road["entry_lanes"] = connectivity[rid]["entry_lanes"]
                    road["exit_lanes"] = connectivity[rid]["exit_lanes"]
        with open(output_path, "w") as f:
            json.dump(data, f)
        return (True, bool(connectivity))
    except Exception as e:
        return (False, str(e))


def extract_connectivity_parallel(tfrecord_dir, num_workers=80):
    """Process all TFRecords in parallel via subprocesses."""
    tfrecord_dir = Path(tfrecord_dir)
    tfrecord_files = sorted(f for f in tfrecord_dir.iterdir() if 'tfrecord' in f.name)
    if not tfrecord_files:
        print(f"  No TFRecord files found in {tfrecord_dir}")
        return {}

    print(f"  Processing {len(tfrecord_files)} TFRecord files with {num_workers} workers...")

    helper_script = Path(tempfile.mktemp(suffix='.py'))
    helper_script.write_text('''
import json, pickle, sys
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

tf_paths = json.loads(sys.argv[1])
result = {}
for tf_path in tf_paths:
    try:
        ds = tf.data.TFRecordDataset(tf_path)
        for raw in ds:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw.numpy())
            sid = scenario.scenario_id
            conn = {}
            for mf in scenario.map_features:
                if mf.HasField("lane"):
                    conn[mf.id] = {
                        "entry_lanes": list(mf.lane.entry_lanes),
                        "exit_lanes": list(mf.lane.exit_lanes),
                    }
            if conn:
                result[sid] = conn
    except Exception as e:
        print(f"WARN: {tf_path}: {e}", file=sys.stderr)

with open(sys.argv[2], "wb") as f:
    pickle.dump(result, f)
print(f"Done: {len(tf_paths)} tfrecords -> {len(result)} scenarios")
''')

    batch_size = max(1, len(tfrecord_files) // num_workers)
    batches = []
    for i in range(0, len(tfrecord_files), batch_size):
        batches.append([str(f) for f in tfrecord_files[i:i + batch_size]])

    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    processes = []
    output_files = []
    for i, batch in enumerate(batches):
        out_file = tempfile.mktemp(suffix=f'_b{i}.pkl')
        output_files.append(out_file)
        p = subprocess.Popen(
            [sys.executable, str(helper_script), json.dumps(batch), out_file],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append(p)

    print(f"  Launched {len(batches)} workers (~{batch_size} tfrecords each)")

    connectivity_map = {}
    total_scenarios = 0
    failed_workers = 0
    pbar = tqdm(total=len(processes), desc="  TFRecord workers", unit="worker")
    # Poll workers for completion instead of blocking sequentially
    remaining = list(range(len(processes)))
    done = set()
    while remaining:
        still_running = []
        for i in remaining:
            p = processes[i]
            ret = p.poll()
            if ret is None:
                still_running.append(i)
                continue
            # Worker finished
            out_file = output_files[i]
            stdout, stderr = p.communicate()
            if ret != 0:
                err = stderr.decode().strip().split('\n')[-3:] if stderr else []
                tqdm.write(f"    Worker {i} FAILED: {'; '.join(err)}")
                failed_workers += 1
            else:
                try:
                    with open(out_file, 'rb') as f:
                        data = pickle.load(f)
                    connectivity_map.update(data)
                    total_scenarios += len(data)
                    os.unlink(out_file)
                except Exception as e:
                    tqdm.write(f"    Worker {i}: load error: {e}")
            done.add(i)
            pbar.update(1)
            pbar.set_postfix(scenarios=total_scenarios, failed=failed_workers)
        remaining = still_running
        if remaining:
            time.sleep(0.5)
    pbar.close()

    helper_script.unlink(missing_ok=True)
    print(f"  Total: {len(connectivity_map)} scenarios with connectivity")
    return connectivity_map


def process_split(split, tfrecord_root, json_root, enriched_root, output_root, workers):
    """Process one split end-to-end."""
    tfrecord_dir = Path(tfrecord_root) / split
    json_dir = Path(json_root) / split
    enriched_dir = Path(enriched_root) / split
    output_dir = Path(output_root) / split

    enriched_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    print(f"\n{'='*60}")
    print(f"SPLIT: {split}")
    print(f"  JSONs: {len(json_files)} in {json_dir}")
    print(f"  TFRecords: {tfrecord_dir}")
    print(f"  Enriched output: {enriched_dir}")
    print(f"  Binary output: {output_dir}")
    print(f"{'='*60}")

    if not json_files:
        print(f"  SKIP: No JSON files found")
        return

    # Step 1: Extract connectivity
    print(f"\n[{split}] Step 1: Extract lane connectivity from TFRecords")
    connectivity_map = extract_connectivity_parallel(tfrecord_dir, num_workers=workers)

    # Step 2: Enrich JSONs
    print(f"\n[{split}] Step 2: Enrich {len(json_files)} JSONs with connectivity")
    tasks = [(jf, enriched_dir / jf.name) for jf in json_files]

    with Pool(workers, initializer=_enrich_init, initargs=(connectivity_map,)) as pool:
        results = list(tqdm(pool.imap(_enrich_worker, tasks, chunksize=50),
                            total=len(tasks), desc=f"Enriching {split}"))

    ok = sum(1 for s, _ in results if s)
    fail = sum(1 for s, _ in results if not s)
    matched = sum(1 for s, enriched in results if s and enriched)
    print(f"  Success: {ok}, Failed: {fail}, With connectivity: {matched}")

    # Free connectivity map before binary conversion
    del connectivity_map

    # Step 3: Convert to binaries
    print(f"\n[{split}] Step 3: Convert enriched JSONs to binaries")
    from pufferlib.ocean.drive.drive import process_all_maps
    process_all_maps(
        json_dir=enriched_dir,
        binary_dir=output_dir,
        max_maps=10_000_000,
        num_workers=workers,
    )

    bin_count = len(list(output_dir.glob("*.bin")))
    print(f"  [{split}] Done: {bin_count} binary files in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create binaries with lane connectivity for all splits")
    parser.add_argument("--tfrecord-root", required=True, help="Root dir with TFRecords (has training/validation/testing)")
    parser.add_argument("--json-root", required=True, help="Root dir with gpudrive JSONs (has training/validation/testing)")
    parser.add_argument("--enriched-root", required=True, help="Output root for enriched JSONs")
    parser.add_argument("--output-root", required=True, help="Output root for binary .bin files")
    parser.add_argument("--workers", type=int, default=80, help="Number of parallel workers")
    parser.add_argument("--splits", type=str, default="training,validation,testing",
                        help="Comma-separated splits to process")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]

    print(f"Processing splits: {splits}")
    print(f"Workers: {args.workers}")
    print(f"TFRecord root: {args.tfrecord_root}")
    print(f"JSON root: {args.json_root}")
    print(f"Enriched root: {args.enriched_root}")
    print(f"Binary root: {args.output_root}")

    for split in splits:
        process_split(split, args.tfrecord_root, args.json_root,
                       args.enriched_root, args.output_root, args.workers)

    # Final summary
    print(f"\n{'='*60}")
    print("ALL DONE!")
    for split in splits:
        output_dir = Path(args.output_root) / split
        bin_count = len(list(output_dir.glob("*.bin"))) if output_dir.exists() else 0
        enriched_dir = Path(args.enriched_root) / split
        json_count = len(list(enriched_dir.glob("*.json"))) if enriched_dir.exists() else 0
        print(f"  {split}: {json_count} enriched JSONs, {bin_count} binaries")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
