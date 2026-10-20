"""Model-free K-means placement and the existing minimum-cost auction bids.

K-means minimizes UE-to-center squared XY distance, not the radio reward.
The runner selects a shared height/beam angle on validation maps only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ver2 import allocation_cost_key, gym_env


KMEANS_SEED = 2026
KMEANS_RESTARTS = 10
KMEANS_MAX_ITERATIONS = 100


def kmeans_centers(
    points: np.ndarray,
    num_clusters: int,
    *,
    ue_seed: int,
) -> np.ndarray:
    """Lloyd iterations with D² initialization; local RNG, stable UE ordering."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("K-means requires finite XY points with shape (N, 2)")
    if num_clusters < 1 or len(np.unique(points, axis=0)) < num_clusters:
        raise ValueError("K-means requires at least K distinct points and K > 0")
    if ue_seed < 0:
        raise ValueError("ue_seed must be non-negative")
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    rng = np.random.default_rng(np.random.SeedSequence([KMEANS_SEED, ue_seed]))
    best_inertia = float("inf")
    best_centers = None
    for _ in range(KMEANS_RESTARTS):
        centers = [points[int(rng.integers(len(points)))].copy()]
        while len(centers) < num_clusters:
            distances = np.square(points[:, None, :] - np.asarray(centers)).sum(2)
            weights = distances.min(axis=1)
            centers.append(points[int(rng.choice(len(points), p=weights / weights.sum()))].copy())
        centers = np.asarray(centers)
        for _ in range(KMEANS_MAX_ITERATIONS):
            distances = np.square(points[:, None, :] - centers).sum(2)
            labels = distances.argmin(axis=1)
            updated = centers.copy()
            occupied = []
            for cluster in range(num_clusters):
                members = points[labels == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
                    occupied.append(cluster)
            # Relocate empty clusters to uncovered points; never emit NaN means.
            for cluster in range(num_clusters):
                if cluster not in occupied:
                    nearest = np.square(points[:, None, :] - updated[occupied]).sum(2).min(1)
                    updated[cluster] = points[int(nearest.argmax())]
                    occupied.append(cluster)
            converged = np.allclose(updated, centers, rtol=0.0, atol=1e-6)
            centers = updated
            if converged:
                break
        inertia = float(np.square(points[:, None, :] - centers).sum(2).min(1).sum())
        if inertia < best_inertia:
            best_inertia, best_centers = inertia, centers.copy()
    assert best_centers is not None
    return best_centers[np.lexsort((best_centers[:, 1], best_centers[:, 0]))]


def assign_distinct_grid(centers: np.ndarray, grid_xy: np.ndarray) -> np.ndarray:
    """Exactly minimize total squared snapping distance with unique grid slots."""

    centers = np.asarray(centers, dtype=np.float64)
    grid_xy = np.asarray(grid_xy, dtype=np.float64)
    if any(a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all() for a in (centers, grid_xy)):
        raise ValueError("Centers and grid must contain finite XY points")
    count = len(centers)
    if not 1 <= count <= 10 or len(np.unique(grid_xy, axis=0)) < count:
        raise ValueError("Grid assignment requires 1..10 centers and at least K distinct grid points")
    unique_grid, original_indices = np.unique(grid_xy, axis=0, return_index=True)
    costs = np.square(centers[:, None, :] - unique_grid).sum(2)
    # ponytail: O(grid * K * 2**K), small for five UAVs; use a rectangular
    # linear-assignment solver if the fleet grows beyond this guarded ceiling.
    states = {0: (0.0, (-1,) * count)}
    for grid_index in range(len(unique_grid)):
        previous = tuple(states.items())
        for mask, (cost, assignment) in previous:
            for center_index in range(count):
                bit = 1 << center_index
                if mask & bit:
                    continue
                next_mask = mask | bit
                next_cost = cost + float(costs[center_index, grid_index])
                if next_cost < states.get(next_mask, (float("inf"), ()))[0]:
                    next_assignment = list(assignment)
                    next_assignment[center_index] = int(original_indices[grid_index])
                    states[next_mask] = (next_cost, tuple(next_assignment))
    return np.asarray(states[(1 << count) - 1][1], dtype=np.int64)


def kmeans_placement(
    environment: gym_env,
    centers: np.ndarray,
    *,
    height: float | np.ndarray,
    theta: float | np.ndarray,
    assignment_order: Sequence[int],
) -> np.ndarray:
    """Snap all centers jointly; cyclic order only maps center slots to UAV IDs."""

    if np.shape(centers) != (environment.num_uavs, 2):
        raise ValueError("One XY center is required per UAV")
    order = tuple(int(index) for index in assignment_order)
    if sorted(order) != list(range(environment.num_uavs)):
        raise ValueError("assignment_order must be a permutation of UAV IDs")
    candidates = environment.candidate_positions
    # Scalars retain fixed geometry; vectors specify geometry per center slot.
    heights = np.broadcast_to(np.asarray(height, dtype=float), (environment.num_uavs,))
    thetas = np.broadcast_to(np.asarray(theta, dtype=float), (environment.num_uavs,))
    geometry_indices = [np.flatnonzero((candidates[:, 2] == h) & (candidates[:, 3] == t))
                        for h, t in zip(heights, thetas)]
    if any(not len(indices) for indices in geometry_indices):
        raise ValueError("K-means height/theta must belong to candidate levels")
    grid_xy = candidates[geometry_indices[0], :2]
    if any(not np.array_equal(candidates[indices, :2], grid_xy) for indices in geometry_indices):
        raise ValueError("K-means geometry levels must share the same XY grid")
    grid_slots = assign_distinct_grid(centers, grid_xy)
    placement = np.empty(environment.num_uavs, dtype=np.int64)
    placement[list(order)] = [indices[slot] for indices, slot in zip(geometry_indices, grid_slots)]
    if not environment.placement_action_space.contains(placement):
        raise ValueError("K-means produced an invalid placement")
    return placement


def greedy_allocation_action(environment: gym_env) -> np.ndarray:
    """Choose the cheapest bid without assuming policy masks guarantee feasibility."""

    pair_masks = environment.get_allocation_action_masks()["pair"]
    action = np.zeros((environment.num_uavs, 3), dtype=np.int64)
    for uav_index, pair_mask in enumerate(pair_masks):
        if environment.link_features[uav_index, environment.current_ue, 1] < 0.5:
            continue
        pairs = sorted(
            (tuple(map(int, pair)) for pair in np.argwhere(pair_mask)),
            key=lambda pair: (
                allocation_cost_key(
                    environment.prb_weight * environment.prb_levels[pair[0]] / environment.total_max_prbs
                    + environment.power_weight * environment.power_levels_w[pair[1]] / environment.power_penalty_denominator_w
                ),
                pair[0], pair[1],
            ),
        )
        best_rank = None
        for prb_index, power_index in pairs:
            num_prbs = int(environment.prb_levels[prb_index])
            power_w = float(environment.power_levels_w[power_index])
            if num_prbs == 0 or power_w == 0.0:
                continue
            proposal = environment._evaluate_allocation_proposal(
                uav_index, power_index, num_prbs, power_w
            )
            rank = environment._proposal_rank(proposal)
            if best_rank is not None and rank[0] > best_rank[0]:
                break
            if proposal["accepted"] and (best_rank is None or rank < best_rank):
                best_rank = rank
                action[uav_index] = (1, prb_index, power_index)
    return action


def evaluate_greedy_placement(environment: gym_env, placement: np.ndarray) -> dict:
    """Evaluate a placement after reset, using the real masks, rewards and auction."""

    if environment.phase != environment.PLACEMENT_PHASE:
        raise ValueError("Reset the environment before evaluating placement")
    _, placement_reward, terminated, truncated, info = environment.step(placement)
    allocation_return = 0.0
    placement_return = float(placement_reward)
    transitions = 1
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = environment.step(greedy_allocation_action(environment))
        allocation_return += float(reward)
        placement_return += float(reward)
        transitions += 1
    xyz = environment.candidate_positions[np.asarray(placement), :3]
    return {
        "evaluation_transitions": transitions,
        "sum_reward": placement_return,
        "allocation_return": allocation_return,
        "placement_return": placement_return,
        "served_count": int(info["served_count"]),
        "served_ratio": float(info["served_ratio"]),
        "total_used_prbs": int(info["total_used_prbs"]),
        "total_used_power_w": float(info["total_used_power_w"]),
        "placement_duplicates": len(xyz) - len(np.unique(xyz, axis=0)),
    }
