# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Extract lane connectivity (entry_lanes, exit_lanes) from Waymo TFRecord files
and add it to the existing JSON map files.

Usage:
    # Process a single TFRecord, match with JSON files, add connectivity
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python extract_lane_connectivity.py \
        --tfrecord /path/to/testing.tfrecord-00000-of-00150 \
        --json-dir /path/to/gpudrive/testing \
        --output-dir /path/to/gpudrive_with_connectivity/testing

    # Dry-run: just show matches without writing
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python extract_lane_connectivity.py \
        --tfrecord /path/to/testing.tfrecord-00000-of-00150 \
        --json-dir /path/to/gpudrive/testing \
        --dry-run
"""
import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2


def extract_connectivity_from_scenario(scenario):
    """Extract lane connectivity from a parsed Waymo Scenario proto.

    Returns dict: lane_id -> {entry_lanes: [...], exit_lanes: [...]}
    """
    connectivity = {}
    for mf in scenario.map_features:
        if mf.HasField('lane'):
            connectivity[mf.id] = {
                'entry_lanes': list(mf.lane.entry_lanes),
                'exit_lanes': list(mf.lane.exit_lanes),
            }
    return connectivity


def build_scenario_index(tfrecord_path):
    """Read all scenarios from a TFRecord file and build a lookup by scenario_id."""
    ds = tf.data.TFRecordDataset(str(tfrecord_path))
    index = {}
    for i, raw in enumerate(ds):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw.numpy())
        index[scenario.scenario_id] = scenario
    return index


def enrich_json_with_connectivity(json_data, connectivity):
    """Add entry_lanes and exit_lanes to road entries in the JSON data."""
    for road in json_data.get("roads", []):
        rid = road.get("id")
        if rid in connectivity:
            road["entry_lanes"] = connectivity[rid]["entry_lanes"]
            road["exit_lanes"] = connectivity[rid]["exit_lanes"]
    return json_data


def process_tfrecord(tfrecord_path, json_dir, output_dir=None, dry_run=False):
    """Process one TFRecord file: match scenarios with JSON files and add connectivity."""
    tfrecord_path = Path(tfrecord_path)
    json_dir = Path(json_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading TFRecord: {tfrecord_path}")
    scenario_index = build_scenario_index(tfrecord_path)
    print(f"  Found {len(scenario_index)} scenarios")

    # Build lookup: scenario_id -> JSON file path
    json_files = sorted(json_dir.glob("*.json"))
    print(f"Scanning {len(json_files)} JSON files for matches...")

    matched = 0
    total_lanes_enriched = 0

    for jf in json_files:
        with open(jf) as f:
            jdata = json.load(f)

        sid = jdata.get("scenario_id", "")
        if sid not in scenario_index:
            continue

        scenario = scenario_index[sid]
        connectivity = extract_connectivity_from_scenario(scenario)

        # Count lanes that will get connectivity
        json_lane_ids = {r["id"] for r in jdata.get("roads", []) if r.get("type") == "lane"}
        enriched = len(json_lane_ids & set(connectivity.keys()))
        total_lanes_enriched += enriched
        matched += 1

        if dry_run:
            print(f"  [MATCH] {jf.name} -> scenario_id={sid}, {enriched}/{len(json_lane_ids)} lanes enriched")
        else:
            enriched_data = enrich_json_with_connectivity(jdata, connectivity)
            out_path = (output_dir / jf.name) if output_dir else jf
            with open(out_path, "w") as f:
                json.dump(enriched_data, f)
            print(f"  [WROTE] {out_path.name} ({enriched} lanes enriched)")

    print(f"\nDone: {matched} scenarios matched, {total_lanes_enriched} lanes enriched total")
    return matched


def main():
    parser = argparse.ArgumentParser(description="Extract lane connectivity from Waymo TFRecords")
    parser.add_argument("--tfrecord", required=True, help="Path to TFRecord file")
    parser.add_argument("--json-dir", required=True, help="Directory with JSON map files")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for enriched JSONs (default: overwrite in-place)")
    parser.add_argument("--dry-run", action="store_true", help="Just show matches, don't write")
    args = parser.parse_args()

    process_tfrecord(args.tfrecord, args.json_dir, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()
