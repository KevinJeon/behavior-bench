# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Batch-convert nuPlan scenarios to PufferDrive .bin format (151 timesteps).

Reads the gpudrive JSON metadata to find matching nuPlan scenarios, then
converts each via nuplan_to_binary with TRAJECTORY_LENGTH=151.

Usage:
    NUPLAN_DATA_ROOT=/path/to/nuplan/dataset \
    NUPLAN_MAPS_ROOT=/path/to/nuplan/maps \
    python -m pufferlib.nuplan_integration.batch_convert \
        --json-dir /path/to/gpudrive_json/training \
        --output-dir /path/to/binaries/training \
        --max-maps 10000
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


def _convert_single(args):
    """Worker: convert a single scenario. Returns (idx, success, error_msg)."""
    idx, json_path, output_path, data_root, map_root = args
    try:
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
            NuPlanScenarioBuilder,
        )
        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
        from nuplan.planning.utils.multithreading.worker_sequential import Sequential
        from pufferlib.nuplan_integration.nuplan_to_binary import nuplan_to_binary

        with open(json_path) as f:
            data = json.load(f)
        metadata = data["metadata"]

        db_path = os.path.join(data_root, "nuplan-v1.1", "trainval", f"{metadata['log_name']}.db")
        if not os.path.exists(db_path):
            return (idx, False, f"DB not found: {db_path}")

        builder = NuPlanScenarioBuilder(
            data_root=data_root, map_root=map_root,
            sensor_root=data_root, db_files=[db_path], map_version="nuplan-maps-v1.0",
        )

        target_ts = metadata["initial_lidar_timestamp"]
        scenarios = list(builder.get_scenarios(
            ScenarioFilter(
                scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
                num_scenarios_per_type=None, limit_total_scenarios=None,
                timestamp_threshold_s=None, ego_displacement_minimum_m=None,
                expand_scenarios=False, remove_invalid_goals=False, shuffle=False,
            ), Sequential(),
        ))

        scenario = None
        for s in scenarios:
            try:
                if abs(s.initial_ego_state.time_point.time_us - target_ts) < 100_000:
                    scenario = s
                    break
            except Exception:
                continue

        if scenario is None:
            return (idx, False, "No matching scenario found")

        map_api = scenario.map_api
        route_ids = list(scenario.get_route_roadblock_ids())
        nuplan_to_binary(scenario, map_api, route_ids, output_path)
        return (idx, True, None)

    except Exception as e:
        return (idx, False, f"{type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", required=True, help="Directory with gpudrive JSONs")
    parser.add_argument("--output-dir", required=True, help="Output directory for .bin files")
    parser.add_argument("--max-maps", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=min(8, cpu_count()))
    args = parser.parse_args()

    data_root = os.environ.get("NUPLAN_DATA_ROOT", "")
    map_root = os.environ.get("NUPLAN_MAPS_ROOT", "")

    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))[:args.max_maps]
    print(f"Found {len(json_files)} JSON files, converting up to {args.max_maps}")

    tasks = []
    for i, jf in enumerate(json_files):
        out_path = str(output_dir / f"map_{i:06d}.bin")
        tasks.append((i, str(jf), out_path, data_root, map_root))

    success = 0
    failed = 0
    errors = []

    with Pool(args.num_workers) as pool:
        for idx, ok, err in tqdm(pool.imap_unordered(_convert_single, tasks), total=len(tasks)):
            if ok:
                success += 1
            else:
                failed += 1
                errors.append((idx, err))

    print(f"\nDone: {success} success, {failed} failed out of {len(tasks)}")
    if errors[:10]:
        print("First 10 errors:")
        for idx, err in errors[:10]:
            print(f"  map_{idx:06d}: {err}")


if __name__ == "__main__":
    main()
