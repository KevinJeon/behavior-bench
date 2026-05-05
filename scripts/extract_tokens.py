# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Extract K=200 motion token vocabulary from training data (Trajeglish K-disk).

Implements the K-disk algorithm from Philion et al. (arxiv 2312.04535):
1. Extract local-frame polygon displacements at shift=2 boundaries
2. K-disk sampling: sample random transition, remove all within epsilon
   (mean 4-corner L2 distance), repeat until K tokens selected
3. Each selected token IS a real data sample (not an average)

Memory-efficient: streams segments to disk in chunks, then subsamples for K-disk.

Usage:
    python scripts/extract_tokens.py \
        --data-root /path/to/binaries \
        --split training \
        --output codebook_200_shift2.pkl \
        --K 200 --shift 2 --num-workers 80
"""

import os
import argparse
import pickle
import time
import tempfile
import multiprocessing as mp
from pathlib import Path

import numpy as np

from pufferlib.prediction.binary_reader import read_binary_scenario
from pufferlib.prediction.trajectory_tokenizer import cal_polygon_contour


VEHICLE_TYPE = 1
DEFAULT_WIDTH = 2.0
DEFAULT_LENGTH = 4.8
DT = 0.1


def _extract_segments_from_map(args_tuple):
    """Worker: extract local-frame polygon displacements from one .bin file.

    Returns list of tuples: (polygon (4,2), traj (shift+1,3), polygon_all (shift+1,4,2))
    as flat bytes for memory efficiency.
    """
    bin_path, shift, width, length = args_tuple
    try:
        scenario = read_binary_scenario(bin_path)
    except Exception:
        return []

    segments = []
    for obj in scenario['objects']:
        if obj['type'] != VEHICLE_TYPE:
            continue

        traj_x = obj['traj_x']
        traj_y = obj['traj_y']
        traj_heading = obj['traj_heading']
        traj_valid = obj['traj_valid'].astype(bool)
        obj_width = obj.get('width', width)
        obj_length = obj.get('length', length)
        T = len(traj_x)

        for t0 in range(0, T - shift, shift):
            t1 = t0 + shift
            if not traj_valid[t0:t1 + 1].all():
                continue

            x0, y0, h0 = traj_x[t0], traj_y[t0], traj_heading[t0]
            cos_h = np.cos(-h0)
            sin_h = np.sin(-h0)

            local_traj = np.zeros((shift + 1, 3), dtype=np.float32)
            for k in range(shift + 1):
                tk = t0 + k
                dx_w = traj_x[tk] - x0
                dy_w = traj_y[tk] - y0
                local_traj[k, 0] = dx_w * cos_h - dy_w * sin_h
                local_traj[k, 1] = dx_w * sin_h + dy_w * cos_h
                local_traj[k, 2] = traj_heading[tk] - h0

            polygon_all = np.zeros((shift + 1, 4, 2), dtype=np.float32)
            for k in range(shift + 1):
                contour = cal_polygon_contour(
                    np.array([local_traj[k, 0]]),
                    np.array([local_traj[k, 1]]),
                    np.array([local_traj[k, 2]]),
                    obj_width, obj_length,
                )
                polygon_all[k] = contour[0]

            final_polygon = polygon_all[-1].astype(np.float64)

            segments.append((final_polygon, local_traj, polygon_all))

    return segments


def polygon_corner_distance(poly_a, poly_b):
    """Mean L2 distance between ordered bounding box corners."""
    corner_dists = np.sqrt(np.sum((poly_a - poly_b) ** 2, axis=-1))
    return np.mean(corner_dists, axis=-1)


def k_disk_sampling(polygons, K, epsilon, rng=None):
    """K-disk vocabulary selection (Trajeglish Algorithm 1).

    procedure SampleKDisks(X, N, epsilon):
        S <- {}
        while len(S) < N:
            x0 ~ X
            X <- {x in X | d(x0, x) > epsilon}
            S <- S + {x0}
        return S
    """
    if rng is None:
        rng = np.random.RandomState(42)

    M = len(polygons)
    alive = np.ones(M, dtype=bool)
    selected = []

    while len(selected) < K:
        candidates = np.where(alive)[0]
        if len(candidates) == 0:
            print(f"  K-disk: ran out of samples at {len(selected)}/{K} tokens")
            break

        pick = rng.randint(len(candidates))
        x0_idx = candidates[pick]
        selected.append(x0_idx)

        x0_poly = polygons[x0_idx]
        dists = polygon_corner_distance(polygons[candidates], x0_poly[None])
        too_close = dists <= epsilon
        alive[candidates[too_close]] = False

        if len(selected) % 50 == 0:
            remaining = alive.sum()
            print(f"  K-disk: selected {len(selected)}/{K}, "
                  f"{remaining} candidates remaining")

    return selected


def find_epsilon_for_k(polygons, K, rng=None):
    """Binary search for epsilon that yields exactly K tokens."""
    if rng is None:
        rng = np.random.RandomState(42)

    sample_size = min(10000, len(polygons))
    sample_idx = rng.choice(len(polygons), sample_size, replace=False)
    ref_idx = rng.choice(len(polygons), 100, replace=False)
    all_dists = []
    for ri in ref_idx:
        d = polygon_corner_distance(polygons[sample_idx], polygons[ri][None])
        all_dists.extend(d.tolist())
    all_dists = np.array(all_dists)
    print(f"  Distance stats: min={all_dists.min():.4f}, "
          f"median={np.median(all_dists):.4f}, max={all_dists.max():.4f}")

    eps_lo = 0.001
    eps_hi = np.percentile(all_dists, 50)
    best_eps = None
    best_selected = None
    best_diff = float('inf')

    for iteration in range(20):
        eps_mid = (eps_lo + eps_hi) / 2
        trial_rng = np.random.RandomState(rng.randint(1 << 31))
        selected = k_disk_sampling(polygons, K, eps_mid, rng=trial_rng)
        n_selected = len(selected)
        diff = abs(n_selected - K)

        print(f"  eps={eps_mid:.6f}: got {n_selected} tokens (target {K})")

        if diff < best_diff or (diff == best_diff and n_selected >= K):
            best_diff = diff
            best_eps = eps_mid
            best_selected = selected

        if n_selected == K:
            break
        elif n_selected < K:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid

    return best_eps, best_selected


def main():
    parser = argparse.ArgumentParser(description="Extract motion token vocabulary (K-disk)")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--output", type=str, default="codebook_200_shift2.pkl")
    parser.add_argument("--K", type=int, default=200)
    parser.add_argument("--shift", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=80)
    parser.add_argument("--max-maps", type=int, default=0)
    parser.add_argument("--subsample", type=int, default=5_000_000,
                        help="Max segments to subsample for K-disk")
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("DRIVE_BINARIES_DATA_ROOT", "")
    if not data_root:
        raise ValueError("Set --data-root or DRIVE_BINARIES_DATA_ROOT")

    split_dir = Path(data_root) / args.split
    bin_files = sorted(split_dir.glob("*.bin"))
    if args.max_maps > 0:
        bin_files = bin_files[:args.max_maps]
    print(f"Processing {len(bin_files)} maps from {split_dir}")

    shift = args.shift
    subsample_target = args.subsample

    # Phase 1: Extract segments with reservoir sampling to stay in memory.
    # Reservoir uses Python lists during collection (no pre-allocated numpy
    # arrays that would be copied to each forked worker process).
    t_start = time.time()
    tasks = [(str(f), shift, DEFAULT_WIDTH, DEFAULT_LENGTH) for f in bin_files]

    rng = np.random.RandomState(42)
    reservoir = []  # list of (polygon, traj, polygon_all) tuples
    total_segments = 0
    processed = 0

    with mp.Pool(args.num_workers) as pool:
        for segments in pool.imap_unordered(_extract_segments_from_map, tasks, chunksize=64):
            for seg in segments:
                total_segments += 1
                if len(reservoir) < subsample_target:
                    reservoir.append(seg)
                else:
                    j = rng.randint(0, total_segments)
                    if j < subsample_target:
                        reservoir[j] = seg

            processed += 1
            if processed % 10000 == 0:
                elapsed = time.time() - t_start
                rate = processed / elapsed
                remaining = (len(tasks) - processed) / max(rate, 1e-6)
                print(f"  {processed}/{len(tasks)} maps ({rate:.0f}/s, "
                      f"~{remaining/60:.0f}min remaining, "
                      f"{total_segments} segments total, "
                      f"{len(reservoir)} in reservoir)")

    n = len(reservoir)
    elapsed = time.time() - t_start
    print(f"Extracted {total_segments} segments total, "
          f"reservoir sampled {n} in {elapsed:.0f}s")

    # Convert reservoir to numpy arrays (after pool is closed)
    print("  Converting reservoir to numpy arrays...")
    polygons = np.array([s[0] for s in reservoir], dtype=np.float64)
    trajectories = np.array([s[1] for s in reservoir], dtype=np.float32)
    polygon_alls = np.array([s[2] for s in reservoir], dtype=np.float32)
    del reservoir  # free list memory

    # Phase 2: K-disk
    print(f"Building K-disk codebook: {n} segments -> {args.K} tokens")
    rng = np.random.RandomState(42)
    print("  Searching for optimal epsilon...")
    best_eps, selected = find_epsilon_for_k(polygons, args.K, rng=rng)
    print(f"  Best epsilon: {best_eps:.6f}m, got {len(selected)} tokens")

    # Pad if needed
    K = args.K
    if len(selected) < K:
        print(f"  Padding {K - len(selected)} tokens with farthest-point sampling...")
        selected_set = set(selected)
        remaining = [i for i in range(n) if i not in selected_set]
        if remaining:
            selected_polys = polygons[np.array(selected)]
            for _ in range(K - len(selected)):
                rem_polys = polygons[remaining]
                min_dists = np.full(len(remaining), np.inf)
                for sp in selected_polys:
                    d = polygon_corner_distance(rem_polys, sp[None])
                    min_dists = np.minimum(min_dists, d)
                best_rem = np.argmax(min_dists)
                selected.append(remaining[best_rem])
                selected_polys = np.vstack([selected_polys, rem_polys[best_rem:best_rem+1]])
                remaining.pop(best_rem)

    selected = selected[:K]

    # Build codebook from selected samples
    sel = np.array(selected)
    traj_codebook = trajectories[sel]
    token_polygon = polygons[sel]
    token_all_polygon = polygon_alls[sel]

    # Discretization error
    eval_size = min(100000, n)
    eval_idx = np.random.choice(n, eval_size, replace=False)
    eval_polys = polygons[eval_idx]
    min_errors = np.full(eval_size, np.inf)
    for k in range(K):
        d = polygon_corner_distance(eval_polys, token_polygon[k][None])
        min_errors = np.minimum(min_errors, d)
    print(f"  Discretization error: mean={min_errors.mean()*100:.2f}cm, "
          f"median={np.median(min_errors)*100:.2f}cm, "
          f"p99={np.percentile(min_errors, 99)*100:.2f}cm")

    codebook = {
        'token': {'veh': token_polygon},
        'traj': {'veh': traj_codebook},
        'token_all': {'veh': token_all_polygon},
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(codebook, f)
    print(f"Saved codebook to {output_path}")

    # Stats
    dx = traj_codebook[:, -1, 0]
    dy = traj_codebook[:, -1, 1]
    dh = traj_codebook[:, -1, 2]
    speeds = np.sqrt(dx**2 + dy**2) / (shift * DT)
    print(f"Token stats:")
    print(f"  dx=[{dx.min():.3f}, {dx.max():.3f}]m")
    print(f"  dy=[{dy.min():.3f}, {dy.max():.3f}]m")
    print(f"  dh=[{dh.min():.3f}, {dh.max():.3f}]rad")
    print(f"  speed=[{speeds.min():.1f}, {speeds.max():.1f}]m/s")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        for i in range(K):
            x = traj_codebook[i, :, 0]
            y = traj_codebook[i, :, 1]
            ax.plot(x, y, '-o', markersize=2, alpha=0.5, linewidth=0.8)
        ax.set_xlabel('dx (m)')
        ax.set_ylabel('dy (m)')
        ax.set_title(f'All {K} tokens (local frame)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        sc = ax.scatter(dx, dy, c=dh, cmap='coolwarm', s=20, alpha=0.7)
        plt.colorbar(sc, ax=ax, label='dheading (rad)')
        ax.set_xlabel('dx final (m)')
        ax.set_ylabel('dy final (m)')
        ax.set_title('Token endpoints colored by heading change')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.hist(speeds, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Approx speed (m/s)')
        ax.set_ylabel('Count')
        ax.set_title('Token speed distribution')
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'Motion Token Vocabulary (K={K}, shift={shift}, '
                     f'{10//shift}Hz, K-disk)', fontsize=14)
        fig.tight_layout()
        plot_path = output_path.with_suffix('.png')
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved vocabulary plot to {plot_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
