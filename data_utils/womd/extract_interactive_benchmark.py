# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Extract validation_interactive binaries with lane connectivity.

Pipeline:
1. Extract lane connectivity from Waymo TFRecords
2. Load interactive JSONs, filter for vehicle-only tracks_to_predict
3. Enrich filtered JSONs with connectivity
4. Convert enriched JSONs to .bin files
5. Duplicate bins (one per tracks_to_predict agent) and write manifest.csv

Usage:
    CUDA_VISIBLE_DEVICES="" python data_utils/womd/extract_interactive_benchmark.py \
        --tfrecord-dir /path/to/waymo/validation \
        --interactive-dir /path/to/interactive_json \
        --enriched-dir /path/to/gpudrive_with_connectivity/validation_interactive \
        --output-dir /path/to/binaries/validation_interactive \
        --workers 80
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_utils.womd.create_training_binaries import (
    extract_connectivity_parallel,
    _enrich_init,
    _enrich_worker,
)


def filter_interactive_jsons(interactive_dir):
    """Load interactive JSONs and filter for scenarios where all tracks_to_predict are vehicles.

    Returns list of (json_path, scenario_id, [(track_index, object_id), ...]) tuples.
    """
    interactive_dir = Path(interactive_dir)
    json_files = sorted(interactive_dir.glob("*.json"))
    print(f"Found {len(json_files)} interactive JSONs in {interactive_dir}")

    kept = []
    skipped_type = 0
    for jf in tqdm(json_files, desc="Filtering interactive scenarios"):
        with open(jf) as f:
            data = json.load(f)

        scenario_id = data.get("scenario_id", "")
        metadata = data.get("metadata", {})
        tracks = metadata.get("tracks_to_predict", [])
        objects = data.get("objects", [])

        all_vehicles = True
        agent_info = []
        for tp in tracks:
            idx = tp["track_index"]
            if idx >= len(objects):
                all_vehicles = False
                break
            obj = objects[idx]
            if obj.get("type") != "vehicle":
                all_vehicles = False
                break
            agent_info.append((idx, obj.get("id", -1)))

        if all_vehicles and len(agent_info) > 0:
            kept.append((jf, scenario_id, agent_info))
        else:
            skipped_type += 1

    print(f"  Kept: {len(kept)} (both tracks are vehicles)")
    print(f"  Skipped: {skipped_type} (non-vehicle tracks)")
    return kept


def enrich_jsons(filtered, connectivity_map, enriched_dir, workers):
    """Enrich filtered JSONs with lane connectivity."""
    enriched_dir = Path(enriched_dir)
    enriched_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(str(jf), str(enriched_dir / jf.name)) for jf, _, _ in filtered]

    with Pool(workers, initializer=_enrich_init, initargs=(connectivity_map,)) as pool:
        results = list(tqdm(
            pool.imap(_enrich_worker, tasks, chunksize=50),
            total=len(tasks), desc="Enriching JSONs"
        ))

    ok = sum(1 for s, _ in results if s)
    fail = sum(1 for s, _ in results if not s)
    matched = sum(1 for s, enriched in results if s and enriched)
    print(f"  Success: {ok}, Failed: {fail}, With connectivity: {matched}")

    if fail > 0:
        for (jf, _, _), (success, info) in zip(filtered, results):
            if not success:
                print(f"    FAILED: {jf.name}: {info}")

    return results


def create_manifest_with_duplicated_bins(filtered, enriched_dir, binary_dir, output_dir, results):
    """Create duplicated bins (one per agent) and write manifest.csv.

    process_all_maps names bins as map_NNNNNN.bin based on sorted enriched JSON order.
    We need to map those back and create per-agent copies.
    """
    enriched_dir = Path(enriched_dir)
    binary_dir = Path(binary_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mapping: enriched filename -> (scenario_id, agent_info)
    # Only include successfully enriched files
    enriched_to_info = {}
    for (jf, scenario_id, agent_info), (success, _) in zip(filtered, results):
        if success:
            enriched_to_info[jf.name] = (scenario_id, agent_info)

    # process_all_maps sorts JSONs and assigns sequential map IDs
    enriched_jsons = sorted(enriched_dir.glob("*.json"))
    src_bins = []
    for i, ejf in enumerate(enriched_jsons):
        if ejf.name in enriched_to_info:
            bin_path = binary_dir / f"map_{i:06d}.bin"
            if bin_path.exists():
                src_bins.append((bin_path, ejf.name, enriched_to_info[ejf.name]))

    print(f"Found {len(src_bins)} binary files to process")

    # Create per-agent copies and manifest rows
    manifest_rows = []
    map_counter = 0
    for bin_path, orig_name, (scenario_id, agent_info) in src_bins:
        for track_index, object_id in agent_info:
            new_filename = f"map_{map_counter:06d}.bin"
            dst = output_dir / new_filename
            shutil.copy2(bin_path, dst)
            manifest_rows.append({
                "new_filename": new_filename,
                "original_filename": orig_name,
                "ego_agent_idx": track_index,
                "scenario_id": scenario_id,
                "map_id": map_counter,
            })
            map_counter += 1

    # Write manifest
    manifest_path = output_dir / "manifest.csv"
    fieldnames = ["new_filename", "original_filename", "ego_agent_idx", "scenario_id", "map_id"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"  Created {map_counter} bin files in {output_dir}")
    print(f"  Manifest written to {manifest_path} ({len(manifest_rows)} entries)")
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract validation_interactive binaries with lane connectivity"
    )
    parser.add_argument("--tfrecord-dir", required=True,
                        help="Directory with Waymo validation TFRecords")
    parser.add_argument("--interactive-dir", required=True,
                        help="Directory with interactive WOMD JSONs (e.g. waymo_0)")
    parser.add_argument("--enriched-dir", required=True,
                        help="Output directory for enriched JSONs")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for final bins + manifest.csv")
    parser.add_argument("--workers", type=int, default=80,
                        help="Number of parallel workers")
    args = parser.parse_args()

    enriched_dir = Path(args.enriched_dir)
    # Intermediate binary dir (before per-agent duplication)
    tmp_binary_dir = enriched_dir.parent / (enriched_dir.name + "_bins_tmp")

    print("=" * 60)
    print("Extract Interactive Benchmark")
    print(f"  TFRecords:     {args.tfrecord_dir}")
    print(f"  Interactive:   {args.interactive_dir}")
    print(f"  Enriched:      {args.enriched_dir}")
    print(f"  Output:        {args.output_dir}")
    print(f"  Workers:       {args.workers}")
    print("=" * 60)

    # Step 1: Extract connectivity from TFRecords
    print("\n[Step 1] Extract lane connectivity from TFRecords")
    connectivity_map = extract_connectivity_parallel(args.tfrecord_dir, num_workers=args.workers)

    # Step 2: Filter interactive scenarios (vehicle-only tracks_to_predict)
    print("\n[Step 2] Filter interactive scenarios")
    filtered = filter_interactive_jsons(args.interactive_dir)

    if not filtered:
        print("No scenarios passed filtering. Exiting.")
        return

    # Step 3: Enrich filtered JSONs with connectivity
    print("\n[Step 3] Enrich filtered JSONs with lane connectivity")
    results = enrich_jsons(filtered, connectivity_map, enriched_dir, args.workers)

    # Free connectivity map
    del connectivity_map

    # Step 4: Convert enriched JSONs to binaries
    print("\n[Step 4] Convert enriched JSONs to binaries")
    from pufferlib.ocean.drive.drive import process_all_maps
    tmp_binary_dir.mkdir(parents=True, exist_ok=True)
    process_all_maps(
        json_dir=enriched_dir,
        binary_dir=tmp_binary_dir,
        max_maps=10_000_000,
        num_workers=args.workers,
    )

    # Step 5: Create per-agent bins + manifest
    print("\n[Step 5] Create per-agent bins and manifest")
    manifest = create_manifest_with_duplicated_bins(
        filtered, enriched_dir, tmp_binary_dir, Path(args.output_dir), results
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("DONE!")
    print(f"  Enriched JSONs: {len(list(enriched_dir.glob('*.json')))} in {enriched_dir}")
    print(f"  Final bins:     {len(list(Path(args.output_dir).glob('*.bin')))} in {args.output_dir}")
    print(f"  Manifest:       {len(manifest)} entries")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
