# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Pre-build prediction cache using multiple CPU workers.

Usage:
    python scripts/build_prediction_cache.py \
        --config pufferlib/config/prediction/smart.ini \
        --splits training validation \
        --num-workers 32

This is much faster than building the cache on-the-fly during training,
since tokenization (motion codebook matching + map tokenization) is
CPU-bound and parallelizes well.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import torch

from pufferlib.prediction.puffer_prediction import load_config
from pufferlib.prediction.binary_reader import read_binary_scenario, list_binary_files
from pufferlib.prediction.trajectory_tokenizer import load_motion_codebook
from pufferlib.prediction.map_tokenizer import load_map_codebook


# Global worker state (initialized once per worker process)
_worker_dataset = None


def _init_worker(data_cfg):
    """Initialize a WaymoBinaryDataset in each worker process."""
    global _worker_dataset
    from pufferlib.prediction.dataset import WaymoBinaryDataset
    _worker_dataset = WaymoBinaryDataset(
        data_dir=data_cfg['data_dir'],
        split='training',  # split doesn't matter for processing
        num_historical_steps=int(data_cfg.get('num_historical_steps', 11)),
        num_future_steps=int(data_cfg.get('num_future_steps', 80)),
        shift=int(data_cfg.get('shift', 5)),
        max_agents=int(data_cfg.get('max_agents', -1)),
        num_actions=int(data_cfg.get('num_actions', -1)),
        cache_dir=None,  # we handle caching ourselves
        max_files=0,  # don't load any file list
    )


def _process_one(args):
    """Process a single binary file and save to cache."""
    bin_path, cache_path = args
    try:
        scenario = read_binary_scenario(bin_path)
        data = _worker_dataset._process_scenario(scenario)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(data, cache_path)
        return True
    except Exception as e:
        print(f"  ERROR {bin_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Pre-build prediction cache')
    parser.add_argument('--config', required=True, help='Path to .ini config')
    parser.add_argument('--splits', nargs='+', default=['training', 'validation'])
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count())
    parser.add_argument('--max-files', type=int, default=-1,
                        help='Max files per split (-1 = all)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data_dir from config')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Override cache_dir from config')
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get('data', config)
    data_dir = args.data_dir or data_cfg['data_dir']
    cache_dir = args.cache_dir or data_cfg.get('cache_dir')
    if not cache_dir or cache_dir in ('None', ''):
        print("ERROR: cache_dir not set in config")
        sys.exit(1)

    data_cfg['data_dir'] = data_dir  # ensure override propagates to workers

    print(f"Data dir:    {data_dir}")
    print(f"Cache dir:   {cache_dir}")
    print(f"Workers:     {args.num_workers}")
    print(f"Splits:      {args.splits}")
    print()

    for split in args.splits:
        file_list = list_binary_files(data_dir, split)
        if args.max_files > 0:
            file_list = file_list[:args.max_files]

        # Find files that need processing
        tasks = []
        skipped = 0
        for filepath in file_list:
            cache_path = os.path.join(cache_dir, split,
                                      Path(filepath).stem + '.pt')
            if os.path.exists(cache_path):
                skipped += 1
            else:
                tasks.append((filepath, cache_path))

        print(f"[{split}] {len(file_list)} total, "
              f"{skipped} cached, {len(tasks)} to process")

        if not tasks:
            continue

        t0 = time.perf_counter()
        done = 0
        errors = 0

        with mp.Pool(args.num_workers,
                      initializer=_init_worker,
                      initargs=(data_cfg,)) as pool:
            for ok in pool.imap_unordered(_process_one, tasks, chunksize=8):
                if ok:
                    done += 1
                else:
                    errors += 1

                total = done + errors
                if total % 500 == 0 or total == len(tasks):
                    elapsed = time.perf_counter() - t0
                    rate = total / elapsed
                    remaining = (len(tasks) - total) / max(rate, 0.01)
                    pct = 100 * total / len(tasks)
                    print(f"\r  [{split}] {total}/{len(tasks)} ({pct:.0f}%) "
                          f"{rate:.1f} files/s | "
                          f"ETA: {remaining/60:.0f}min | "
                          f"errors: {errors}",
                          end="", flush=True)

        elapsed = time.perf_counter() - t0
        print(f"\n  [{split}] Done: {done} cached, {errors} errors "
              f"in {elapsed/60:.1f}min ({done/elapsed:.1f} files/s)\n")


if __name__ == '__main__':
    main()
