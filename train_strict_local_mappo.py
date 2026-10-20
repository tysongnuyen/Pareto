"""Train MAPPO placement plus five local bidders on disjoint maps.

Default schedule
----------------
* One global episode uses two UE maps. Each map is sampled once under each of
  the five cyclic UAV selection orders: ten joint placements per global.
* Placement PPO accumulates ten globals (100 placement samples) per update.
  The first ten globals warm up allocation only; the final 350 globals update
  placement only with allocation frozen. The default total is 710 globals.
* After every active global, allocation performs at most 50 PPO updates on
  that global's ten fixed map/placement pairs (500 real transitions/update),
  stopping after 10 consecutive evaluations without a new best mean reward.
* Keep the final allocation actors, critic and optimizer states. Inner best
  deterministic scores control patience and never roll training back.
* Placement targets are initial deterministic returns before allocation adapts
  to that global. Validation, diagnostic and test seeds stay disjoint.
* Allocation entropy is scheduled by actual allocation training episodes.

Outputs
-------
``training_metrics.csv``
    Ten sample rows per global, identified by placement_sample_id and placement_index_in_global.
``allocation_metrics.csv``
    One row for every allocation inner episode, including return, resource use,
    served UEs, entropy coefficient, and PPO update diagnostics.
``uav_placement_metrics.csv``
    One diagnostic row per UAV and episode, including placement geometry,
    coverage, served UEs, attributed reward, and resource use.
``episode_sum_reward_by_seed.png``
    Frozen-batch return curves: post-adaptation while allocation trains and
    frozen-allocator zero-shot in the final placement-only phase.
``placement_10_episode_mean_reward_by_seed.png``
    Non-overlapping 10-sample return block means, using post-adaptation returns
    while allocation trains and frozen-allocator returns in placement-only.
``validation_episode_sum_reward_by_seed.png``
    Mean deterministic validation return across the fixed validation ensemble.
``validation_metrics.csv`` and ``best_validation_summary.json``
    One validation measurement per layout and the checkpoint selected only by
    mean scalar team return.
``test_metrics.csv`` and ``test_summary.json``
    One final deterministic evaluation of the best checkpoint on test layouts.
``diagnostic_metrics.csv`` and ``diagnostic_summary.json``
    The same selected checkpoint on the former test split; never used for
    checkpoint selection.
``checkpoints/*.pt``
    Periodic, final, and best-validation checkpoints. Exact training resume is
    supported only at a completed frozen-batch boundary; inner-loop checkpoints
    remain model-selection/evaluation artifacts and are rejected for resume.
``greedy_baseline_metrics.csv`` and ``greedy_baseline_summary.json``
    Policy placement followed by deterministic minimum-cost proposals through
    the environment's physical feasibility checks and fixed auction.
``kmeans_baseline_*.csv`` and ``kmeans_baseline_summary.json``
    Model-free K-means/grid placement with the same greedy allocator. Select a
    common height/angle on validation only, then evaluate once per test map.
``inference_benchmark.json``
    Optional measured deterministic episode latency and process/device memory;
    no unmeasured speedup or scaling claim is inferred from it.

The environment is imported directly from ``ver2.py`` beside this script.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
import time
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator

import env as radio_env
from kmeans_baseline import (
    KMEANS_MAX_ITERATIONS,
    KMEANS_RESTARTS,
    KMEANS_SEED,
    evaluate_greedy_placement,
    kmeans_centers,
    kmeans_placement,
)
from strict_local_independent_mappo import (
    StrictLocalIndependentMAPPO,
    evaluate_one_episode,
    train_frozen_placement_batch,
)
from ver2 import gym_env, resolve_reward_weights


DEFAULT_TRAIN_SEEDS = (*range(14, 214), *range(274, 434), *range(484, 1744))
DEFAULT_VALIDATION_SEEDS = tuple(range(214, 224))
DEFAULT_DIAGNOSTIC_SEEDS = tuple(range(224, 274))
DEFAULT_TEST_SEEDS = tuple(range(434, 484))
ALLOCATION_WARMUP_BATCHES = 1
DEFAULT_PLACEMENT_ONLY_EPISODES = 350
MAPS_PER_GLOBAL = 2
PLACEMENTS_PER_MAP = 5
PLACEMENTS_PER_GLOBAL = MAPS_PER_GLOBAL * PLACEMENTS_PER_MAP
CSV_FIELDS = (
    "global_episode",
    "placement_sample_id",
    "placement_index_in_global",
    "frozen_batch_id",
    "problem_index_in_batch",
    "training_phase",
    "placement_order",
    "allocation_training_active",
    "allocation_state_unchanged",
    "frozen_batch_summary_row",
    "placement_update_id",
    "placement_update_summary_row",
    "training_budget_used",
    "allocation_episodes_used",
    "placement_samples_collected",
    "real_allocation_training_transitions",
    "deterministic_selection_evaluation_transitions",
    "counterfactual_preview_transitions",
    "cumulative_placement_actor_optimizer_steps",
    "cumulative_placement_critic_optimizer_steps",
    "cumulative_allocation_actor_optimizer_steps",
    "cumulative_allocation_critic_optimizer_steps",
    "cumulative_total_optimizer_steps",
    "frozen_batch_wall_time_seconds",
    "wall_time_seconds",
    "ue_seed",
    "episode_in_seed",
    "sum_reward",
    "allocation_return",
    "placement_return",
    "initial_zero_shot_return",
    "served_count",
    "served_ratio",
    "total_used_prbs",
    "total_used_power_w",
    "placement_duplicates",
    "placement_updated",
    "placement_target_accepted",
    "placement_episodes_collected",
    "placement_actor_loss",
    "placement_critic_loss",
    "placement_entropy",
    "placement_clip_fraction",
    "placement_advantage_mean",
    "placement_value_mean",
    "placement_popart_mean",
    "placement_popart_std",
    "placement_gradient_steps",
    "placement_actor_optimizer_steps",
    "placement_policy_sample_uses",
    "allocation_actor_loss",
    "allocation_critic_loss",
    "allocation_entropy",
    "allocation_participation_entropy",
    "allocation_prb_entropy",
    "allocation_power_entropy",
    "allocation_reject_fraction",
    "allocation_counterfactual_advantage_mean",
    "allocation_counterfactual_nonzero_fraction",
    "allocation_clip_fraction",
    "allocation_policy_kl",
    "allocation_entropy_coef",
    "allocation_value_mean",
    "allocation_value_max_abs",
    "allocation_q_mean",
    "allocation_q_max_abs",
    "allocation_return_mean",
    "allocation_rollout_size",
    "allocation_gradient_steps",
    "allocation_critic_gradient_steps",
    "allocation_policy_sample_uses",
    "allocation_inner_episodes",
    "allocation_converged",
    "allocation_convergence_change",
    "allocation_stop_reason",
    "allocation_inner_updates",
)

ALLOCATION_CSV_FIELDS = (
    "global_episode",
    "placement_sample_id",
    "placement_index_in_global",
    "frozen_batch_id",
    "problem_index_in_batch",
    "allocation_update_id",
    "allocation_update_summary_row",
    "training_budget_used",
    "allocation_episode",
    "inner_episode",
    "real_allocation_training_transitions",
    "counterfactual_preview_transitions",
    "allocation_policy_minibatch_steps",
    "allocation_critic_optimizer_steps",
    "ue_seed",
    "episode_in_seed",
    "sum_reward",
    "allocation_return",
    "placement_return",
    "served_count",
    "served_ratio",
    "total_used_prbs",
    "total_used_power_w",
    "allocation_entropy_coef",
    "allocation_actor_loss",
    "allocation_critic_loss",
    "allocation_entropy",
    "allocation_participation_entropy",
    "allocation_prb_entropy",
    "allocation_power_entropy",
    "allocation_reject_fraction",
    "allocation_counterfactual_advantage_mean",
    "allocation_counterfactual_nonzero_fraction",
    "accepted_count",
    "rejected_count",
    "qos_failure_count",
    "hard_invalid_count",
    "zero_resource_count",
    "out_of_coverage_count",
    "prb_budget_exceeded_count",
    "power_budget_exceeded_count",
    "proposal_count",
    "reject_proposal_count",
    "valid_proposal_count",
    "valid_loser_count",
    "shared_prbs_remaining",
    *(
        field
        for uav_index in range(5)
        for field in (
            f"uav_{uav_index}_selected_count",
            f"uav_{uav_index}_participation_count",
            f"uav_{uav_index}_win_count",
        )
    ),
    *(
        field
        for uav_index in range(5)
        for field in (
            f"allocation_uav_{uav_index}_selection_fraction",
            f"allocation_uav_{uav_index}_participation_fraction",
            f"allocation_uav_{uav_index}_win_fraction",
            f"allocation_uav_{uav_index}_counterfactual_advantage_mean",
            f"allocation_uav_{uav_index}_counterfactual_abs_advantage_mean",
            f"allocation_uav_{uav_index}_counterfactual_nonzero_fraction",
        )
    ),
    "allocation_clip_fraction",
    "allocation_policy_kl",
    "allocation_value_mean",
    "allocation_value_max_abs",
    "allocation_q_mean",
    "allocation_q_max_abs",
    "allocation_return_mean",
    "allocation_rollout_size",
    "allocation_gradient_steps",
    "allocation_critic_gradient_steps",
    "allocation_policy_sample_uses",
)

UAV_PLACEMENT_CSV_FIELDS = (
    "global_episode",
    "placement_sample_id",
    "placement_index_in_global",
    "ue_seed",
    "episode_in_seed",
    "uav_id",
    "selection_rank",
    "placement_index",
    "x_m",
    "y_m",
    "height_m",
    "theta_deg",
    "coverage_radius_m",
    "covered_ues",
    "unique_covered_ues",
    "overlap_covered_ues",
    "served_ues",
    "attributed_placement_reward",
    "used_prbs",
    "used_power_w",
)

EVALUATION_CSV_FIELDS = (
    "global_episode",
    "evaluation_kind",
    "evaluation_transitions",
    "placement_order",
    "ue_seed",
    "sum_reward",
    "allocation_return",
    "placement_return",
    "served_count",
    "served_ratio",
    "total_used_prbs",
    "total_used_power_w",
    "placement_duplicates",
)


def parse_seed_list(raw_value: str) -> tuple[int, ...]:
    """Parse a comma-separated CLI seed list while preserving its order."""

    values = tuple(
        int(part.strip())
        for part in raw_value.split(",")
        if part.strip()
    )
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("The seed list must not contain duplicates")
    return values


def cyclic_placement_order(
    problem_index: int,
    num_uavs: int,
) -> tuple[int, ...]:
    """Right-rotate UAV priority so every five maps cover every rank once."""

    if problem_index < 0:
        raise ValueError("problem_index cannot be negative")
    if num_uavs <= 0:
        raise ValueError("num_uavs must be positive")
    first = (-int(problem_index)) % int(num_uavs)
    return tuple((first + offset) % num_uavs for offset in range(num_uavs))


def format_placement_order(order: Sequence[int]) -> str:
    """Format zero-based actor indices as the human-facing one-based order."""

    return "-".join(str(int(index) + 1) for index in order)


def training_phase_for_episode(
    global_episode: int,
    *,
    warmup_samples: int,
    joint_training_end: int,
    total_episodes: int,
) -> str:
    """Return the declared 10/warm-up, joint, or placement-only phase."""

    if not 1 <= global_episode <= total_episodes:
        raise ValueError("global_episode is outside the training schedule")
    if global_episode <= warmup_samples:
        return "allocation_warmup"
    if global_episode <= joint_training_end:
        return "joint_placement_allocation"
    return "placement_only_allocation_frozen"


def allocation_model_sha256(
    learner: StrictLocalIndependentMAPPO,
) -> str:
    """Hash allocation actor/critic tensors for the frozen-phase invariant."""

    digest = hashlib.sha256()
    for scope, module in (
        ("actors", learner.allocation_actors),
        ("critic", learner.allocation_critic),
    ):
        for name, value in module.state_dict().items():
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{scope}.{name}\0{tensor.dtype}\0".encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            )
    return digest.hexdigest()


def validate_seed_splits(
    train_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    test_seeds: Sequence[int],
    diagnostic_seeds: Sequence[int] = (),
) -> None:
    """Require internally unique, pairwise-disjoint UE seed sets."""

    named_seed_sets = {
        "train": tuple(int(seed) for seed in train_seeds),
        "validation": tuple(int(seed) for seed in validation_seeds),
        "diagnostic": tuple(int(seed) for seed in diagnostic_seeds),
        "test": tuple(int(seed) for seed in test_seeds),
    }
    invalid = [
        name
        for name, values in named_seed_sets.items()
        if (name != "diagnostic" and not values)
        or len(set(values)) != len(values)
    ]
    if invalid:
        raise ValueError(f"Seed splits must be non-empty and unique: {invalid}")
    overlaps = {
        f"{left}/{right}": sorted(
            set(named_seed_sets[left]) & set(named_seed_sets[right])
        )
        for left, right in (
            ("train", "validation"),
            ("train", "diagnostic"),
            ("train", "test"),
            ("validation", "diagnostic"),
            ("validation", "test"),
            ("diagnostic", "test"),
        )
        if set(named_seed_sets[left]) & set(named_seed_sets[right])
    }
    if overlaps:
        raise ValueError(
            "Train/validation/diagnostic/test UE seeds must be disjoint: "
            f"{overlaps}"
        )


def build_interleaved_seed_schedule(
    seeds: Sequence[int],
    episodes_per_seed: int,
    training_seed: int,
) -> tuple[int, ...]:
    """Repeat one shuffled cycle so every frozen batch has distinct maps."""

    schedule_rng = random.Random(training_seed)
    cycle = [int(seed) for seed in seeds]
    schedule_rng.shuffle(cycle)
    return tuple(cycle * episodes_per_seed)


def configure_deterministic_training() -> None:
    """Enable deterministic CPU/CUDA kernels required by exact resume."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_training_randomness(seed: int) -> None:
    """Seed policy sampling, minibatch permutations, NumPy, and Python RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable")
    return device


def runtime_provenance(device: torch.device) -> dict[str, Any]:
    """Capture the runtime and source identity needed to audit a resumed run."""

    def installed_version(distribution: str) -> str | None:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return None

    source_hashes: dict[str, str | None] = {}
    source_directory = Path(__file__).resolve().parent
    for name in (
        "env.py",
        "ver2.py",
        "strict_local_independent_mappo.py",
        "learning_credit.py",
        "kmeans_baseline.py",
        Path(__file__).name,
    ):
        path = source_directory / name
        source_hashes[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cudnn_runtime": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
        "gymnasium": installed_version("gymnasium"),
        "matplotlib": matplotlib.__version__,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None
        ),
        "source_sha256": source_hashes,
        "command": list(sys.argv),
    }


def resume_configuration_fingerprint(args: argparse.Namespace) -> str:
    """Hash effective training configuration, not omitted-vs-explicit defaults."""

    values = vars(args).copy()
    values["power_weight"], values["prb_weight"] = resolve_reward_weights(
        args.coordination_mode, args.power_weight, args.prb_weight
    )
    for key in (
        "resume",
        "output_dir",
        "progress_every",
        "checkpoint_every",
        "plot_tick_interval",
        "benchmark_inference_episodes",
        "no_greedy_baseline",
        "no_kmeans_baseline",
        "kmeans_baseline_only",
    ):
        values.pop(key, None)
    payload = json.dumps(
        values, sort_keys=True, default=list, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def optimizer_step_counts(
    learner: StrictLocalIndependentMAPPO,
) -> dict[str, int]:
    """Read exact lifetime ``optimizer.step()`` calls from the learner."""

    return learner.optimizer_step_counts()


def mean_update_metric(updates: Sequence[dict[str, float]], key: str) -> float:
    """Average one allocation metric across rollout updates in an episode."""

    values = [float(update[key]) for update in updates if key in update]
    return float(np.mean(values)) if values else float("nan")


def summarize_allocation_actions(
    info: dict[str, Any], num_uavs: int = 5
) -> dict[str, int | None]:
    """Count local proposals and fixed-auction outcomes for one episode."""

    allocations = [item for item in info["allocations"] if item is not None]
    proposal_evaluations = [
        proposal
        for item in allocations
        for proposal in item.get("proposal_evaluations", [])
    ]
    counts = {
        "accepted_count": sum(bool(item.get("accepted")) for item in allocations),
        "rejected_count": sum(
            not bool(item.get("accepted")) for item in allocations
        ),
        "qos_failure_count": sum(
            proposal.get("reason") == "qos_not_met"
            for proposal in proposal_evaluations
        ),
        "hard_invalid_count": sum(
            proposal.get("reason")
            in {
                "out_of_coverage",
                "prb_budget_exceeded",
                "power_budget_exceeded",
                "zero_resource",
            }
            for proposal in proposal_evaluations
        ),
    }
    for reason in (
        "zero_resource", "out_of_coverage", "prb_budget_exceeded",
        "power_budget_exceeded",
    ):
        counts[f"{reason}_count"] = sum(
            proposal.get("reason") == reason for proposal in proposal_evaluations
        )
    counts["proposal_count"] = len(proposal_evaluations)
    counts["reject_proposal_count"] = len(allocations) * num_uavs - len(proposal_evaluations)
    counts["valid_proposal_count"] = sum(
        bool(proposal.get("accepted")) for proposal in proposal_evaluations
    )
    counts["valid_loser_count"] = counts["valid_proposal_count"] - counts["accepted_count"]
    for uav_index in range(num_uavs):
        participation_count = sum(
            int(proposal.get("uav_id", -1)) == uav_index
            for proposal in proposal_evaluations
        )
        win_count = sum(
            item.get("winning_uav_id") == uav_index for item in allocations
        )
        # Retain selected_count as a legacy alias; it has always counted wins.
        counts[f"uav_{uav_index}_selected_count"] = win_count
        counts[f"uav_{uav_index}_participation_count"] = participation_count
        counts[f"uav_{uav_index}_win_count"] = win_count
    return {
        **{key: int(value) for key, value in counts.items()},
        "shared_prbs_remaining": info.get("shared_prbs_remaining"),
    }


def build_allocation_log_rows(
    metrics: dict[str, Any],
    *,
    global_episode: int,
    allocation_episodes_before_cycle: int,
    ue_seed: int,
    episode_in_seed: int,
) -> list[dict[str, float | int]]:
    """Build one CSV row for every allocation episode in a frozen cycle."""

    results = metrics["allocation_episode_results"]
    entropy_coefficients = metrics["allocation_entropy_coefficients"]
    if len(results) != len(entropy_coefficients):
        raise RuntimeError("Allocation results and entropy history are misaligned")

    rows: list[dict[str, float | int]] = []
    for inner_episode, (result, entropy_coefficient) in enumerate(
        zip(results, entropy_coefficients),
        start=1,
    ):
        info = result["final_info"]
        if int(info["episode_ue_seed"]) != int(ue_seed):
            raise RuntimeError("Allocation episode used the wrong UE layout seed")
        allocation_episode = allocation_episodes_before_cycle + inner_episode
        updates = result["allocation_updates"]
        action_counts = summarize_allocation_actions(info)
        rows.append(
            {
                "global_episode": int(global_episode),
                "training_budget_used": int(global_episode + allocation_episode),
                "allocation_episode": int(allocation_episode),
                "inner_episode": int(inner_episode),
                "ue_seed": int(ue_seed),
                "episode_in_seed": int(episode_in_seed),
                "sum_reward": float(result["placement_return"]),
                "allocation_return": float(result["allocation_return"]),
                "placement_return": float(result["placement_return"]),
                "served_count": int(result["served_count"]),
                "served_ratio": float(result["served_ratio"]),
                "total_used_prbs": int(info["total_used_prbs"]),
                "total_used_power_w": float(info["total_used_power_w"]),
                "allocation_entropy_coef": float(entropy_coefficient),
                "allocation_actor_loss": mean_update_metric(
                    updates, "allocation_actor_loss"
                ),
                "allocation_critic_loss": mean_update_metric(
                    updates, "allocation_critic_loss"
                ),
                "allocation_entropy": mean_update_metric(
                    updates, "allocation_entropy"
                ),
                "allocation_participation_entropy": mean_update_metric(
                    updates, "allocation_participation_entropy"
                ),
                "allocation_prb_entropy": mean_update_metric(
                    updates, "allocation_prb_entropy"
                ),
                "allocation_power_entropy": mean_update_metric(
                    updates, "allocation_power_entropy"
                ),
                "allocation_reject_fraction": mean_update_metric(
                    updates, "allocation_reject_fraction"
                ),
                "allocation_counterfactual_advantage_mean": mean_update_metric(
                    updates, "allocation_counterfactual_advantage_mean"
                ),
                "allocation_counterfactual_nonzero_fraction": mean_update_metric(
                    updates, "allocation_counterfactual_nonzero_fraction"
                ),
                **action_counts,
                **{
                    key: mean_update_metric(updates, key)
                    for uav_index in range(5)
                    for key in (
                        f"allocation_uav_{uav_index}_selection_fraction",
                        f"allocation_uav_{uav_index}_participation_fraction",
                        f"allocation_uav_{uav_index}_win_fraction",
                        f"allocation_uav_{uav_index}_counterfactual_advantage_mean",
                        f"allocation_uav_{uav_index}_counterfactual_abs_advantage_mean",
                        f"allocation_uav_{uav_index}_counterfactual_nonzero_fraction",
                    )
                },
                "allocation_clip_fraction": mean_update_metric(
                    updates, "allocation_clip_fraction"
                ),
                "allocation_policy_kl": mean_update_metric(
                    updates, "allocation_policy_kl"
                ),
                "allocation_value_mean": mean_update_metric(
                    updates, "allocation_value_mean"
                ),
                "allocation_value_max_abs": mean_update_metric(
                    updates, "allocation_value_max_abs"
                ),
                "allocation_q_mean": mean_update_metric(
                    updates, "allocation_q_mean"
                ),
                "allocation_q_max_abs": mean_update_metric(
                    updates, "allocation_q_max_abs"
                ),
                "allocation_return_mean": mean_update_metric(
                    updates, "allocation_return_mean"
                ),
                "allocation_rollout_size": mean_update_metric(
                    updates, "allocation_rollout_size"
                ),
                "allocation_gradient_steps": mean_update_metric(
                    updates, "allocation_gradient_steps"
                ),
                "allocation_critic_gradient_steps": mean_update_metric(
                    updates, "allocation_critic_gradient_steps"
                ),
                "allocation_policy_sample_uses": mean_update_metric(
                    updates, "allocation_policy_sample_uses"
                ),
            }
        )
    return rows


def build_allocation_batch_log_rows(
    metrics: dict[str, Any],
    *,
    first_global_episode: int,
    allocation_episodes_before_batch: int,
    allocation_updates_before_batch: int,
    frozen_batch_id: int,
    num_ues: int,
    episode_in_seed: dict[int, int],
    placements_per_global: int = 1,
) -> list[dict[str, float | int]]:
    """Log the true update-major order of a frozen placement batch."""

    results_by_problem = metrics["allocation_episode_results_by_problem"]
    entropy_by_problem = metrics[
        "allocation_entropy_coefficients_by_problem"
    ]
    seeds = [int(seed) for seed in metrics["ue_seeds"]]
    if not (
        len(results_by_problem) == len(entropy_by_problem) == len(seeds)
    ):
        raise RuntimeError("Frozen-batch allocation histories are misaligned")
    num_updates = int(metrics["allocation_inner_updates"])
    update_metrics = metrics["allocation_updates"]
    if len(update_metrics) != num_updates:
        raise RuntimeError("Frozen-batch update metrics are misaligned")
    rows: list[dict[str, float | int]] = []
    allocation_episode = int(allocation_episodes_before_batch)
    for update_index in range(num_updates):
        for problem_index, ue_seed in enumerate(seeds):
            allocation_episode += 1
            one_result_metrics = {
                "allocation_episode_results": [
                    results_by_problem[problem_index][update_index]
                ],
                "allocation_entropy_coefficients": [
                    entropy_by_problem[problem_index][update_index]
                ],
            }
            row = build_allocation_log_rows(
                one_result_metrics,
                global_episode=first_global_episode + problem_index // placements_per_global,
                allocation_episodes_before_cycle=allocation_episode - 1,
                ue_seed=ue_seed,
                episode_in_seed=episode_in_seed[ue_seed] + seeds[:problem_index].count(ue_seed),
            )[0]
            row["inner_episode"] = update_index + 1
            summary_row = problem_index == 0
            one_update = update_metrics[update_index]
            row.update(
                {
                    "placement_sample_id": (first_global_episode - 1) * placements_per_global + problem_index + 1,
                    "placement_index_in_global": problem_index % placements_per_global + 1,
                    "frozen_batch_id": int(frozen_batch_id),
                    "problem_index_in_batch": int(problem_index + 1),
                    "allocation_update_id": int(
                        allocation_updates_before_batch + update_index + 1
                    ),
                    "allocation_update_summary_row": int(summary_row),
                    "real_allocation_training_transitions": int(num_ues),
                    "counterfactual_preview_transitions": (
                        int(one_update.get("allocation_unique_outcomes", 0))
                        if summary_row
                        else 0
                    ),
                    "allocation_policy_minibatch_steps": (
                        int(one_update.get("allocation_gradient_steps", 0))
                        if summary_row
                        else 0
                    ),
                    "allocation_critic_optimizer_steps": (
                        int(
                            one_update.get(
                                "allocation_critic_gradient_steps", 0
                            )
                        )
                        if summary_row
                        else 0
                    ),
                }
            )
            rows.append(row)
    return rows


def allocation_entropy_coefficient(
    episode: int,
    *,
    initial: float,
    final: float,
    anneal_start_episode: int,
    anneal_end_episode: int,
) -> float:
    """Hold, linearly anneal, then hold the allocation entropy coefficient."""

    if episode <= anneal_start_episode:
        return float(initial)
    if episode >= anneal_end_episode:
        return float(final)
    progress = (episode - anneal_start_episode) / (
        anneal_end_episode - anneal_start_episode
    )
    return float(initial + progress * (final - initial))


def evaluate_seed_set(
    environment: gym_env,
    learner: StrictLocalIndependentMAPPO,
    seeds: Sequence[int],
    *,
    global_episode: int,
    evaluation_kind: str,
) -> list[dict[str, float | int]]:
    """Evaluate every UE layout once using its fixed UE sequence."""

    rows: list[dict[str, float | int]] = []
    for problem_index, ue_seed in enumerate(seeds):
        placement_order = cyclic_placement_order(
            problem_index, learner.num_uavs
        )
        metrics = evaluate_one_episode(
            environment,
            learner,
            ue_seed=int(ue_seed),
            placement_order=placement_order,
        )
        info = metrics["final_info"]
        if int(info["episode_ue_seed"]) != int(ue_seed):
            raise RuntimeError("Evaluation used the wrong UE layout seed")
        placement = [int(value) for value in metrics["placement_actions"]]
        rows.append(
            {
                "global_episode": int(global_episode),
                "evaluation_kind": evaluation_kind,
                "evaluation_transitions": int(environment.num_ues + 1),
                "placement_order": format_placement_order(placement_order),
                "ue_seed": int(ue_seed),
                "sum_reward": float(metrics["placement_return"]),
                "allocation_return": float(metrics["allocation_return"]),
                "placement_return": float(metrics["placement_return"]),
                "served_count": int(metrics["served_count"]),
                "served_ratio": float(metrics["served_ratio"]),
                "total_used_prbs": int(info["total_used_prbs"]),
                "total_used_power_w": float(info["total_used_power_w"]),
                "placement_duplicates": len(placement) - len(set(placement)),
            }
        )
    return rows


def evaluate_greedy_allocation_baseline(
    environment: gym_env,
    learner: StrictLocalIndependentMAPPO,
    seeds: Sequence[int],
    *,
    global_episode: int,
) -> list[dict[str, float | int]]:
    """Use policy placement, then cheapest physically feasible bids and auction."""

    rows: list[dict[str, float | int]] = []
    for problem_index, ue_seed in enumerate(seeds):
        placement_order = cyclic_placement_order(
            problem_index, learner.num_uavs
        )
        observation, _ = environment.reset(options={"ue_seed": int(ue_seed)})
        placement = learner.act_placement(
            observation,
            environment.get_actor_observations(),
            deterministic=True,
            store_for_update=False,
            selection_order=placement_order,
        )
        metrics = evaluate_greedy_placement(environment, placement)
        rows.append(
            {
                "global_episode": int(global_episode),
                "evaluation_kind": "policy_placement_greedy_allocation",
                "placement_order": format_placement_order(placement_order),
                "ue_seed": int(ue_seed),
                **metrics,
            }
        )
    return rows


def evaluate_kmeans_seed_set(
    environment: gym_env,
    seeds: Sequence[int],
    *,
    height: float,
    theta: float,
    evaluation_kind: str,
    centers_by_seed: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A fixed K-means protocol; no learned policy and no per-test-map tuning."""

    rows, placements = [], []
    for problem_index, ue_seed in enumerate(seeds):
        ue_seed = int(ue_seed)
        environment.reset(options={"ue_seed": ue_seed})
        if ue_seed not in centers_by_seed:
            centers_by_seed[ue_seed] = kmeans_centers(
                np.asarray([(ue.x_m, ue.y_m) for ue in environment.ues]),
                environment.num_uavs,
                ue_seed=ue_seed,
            )
        centers = centers_by_seed[ue_seed]
        order = cyclic_placement_order(problem_index, environment.num_uavs)
        placement = kmeans_placement(
            environment, centers, height=height, theta=theta, assignment_order=order
        )
        metrics = evaluate_greedy_placement(environment, placement)
        if environment.episode_ue_seed != ue_seed:
            raise RuntimeError("K-means evaluation used the wrong UE layout seed")
        rows.append({
            "global_episode": 0,  # Model-free baseline has no training checkpoint.
            "evaluation_kind": evaluation_kind,
            "placement_order": format_placement_order(order),
            "ue_seed": ue_seed,
            "height_m": float(height),
            "theta_deg": float(theta),
            **metrics,
        })
        for center_index, uav_index in enumerate(order):
            candidate_index = int(placement[uav_index])
            x, y, h, angle, radius = environment.candidate_positions[candidate_index]
            placements.append({
                "ue_seed": ue_seed,
                "uav_id": uav_index,
                "center_x_m": float(centers[center_index, 0]),
                "center_y_m": float(centers[center_index, 1]),
                "candidate_index": candidate_index,
                "x_m": float(x), "y_m": float(y), "height_m": float(h),
                "theta_deg": float(angle), "radius_m": float(radius),
            })
    return rows, placements


def run_kmeans_baseline(
    validation_environment: gym_env,
    test_environment: gym_env,
    validation_seeds: Sequence[int],
    test_seeds: Sequence[int],
    output_directory: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select common h/theta on validation, lock it, then evaluate the test split."""

    for seeds in (validation_seeds, test_seeds):
        if not len(seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("K-means validation/test seeds must be non-empty and unique")
    if set(validation_seeds) & set(test_seeds):
        raise ValueError("K-means validation and test seeds must be disjoint")
    started = time.perf_counter()
    geometry_options = np.unique(validation_environment.candidate_positions[:, 2:4], axis=0)
    validation_rows, trials = [], []
    centers_by_seed: dict[int, np.ndarray] = {}
    best_score = None
    selected = None
    for height, theta in geometry_options:
        rows, _ = evaluate_kmeans_seed_set(
            validation_environment, validation_seeds, height=float(height),
            theta=float(theta), evaluation_kind="kmeans_greedy_validation",
            centers_by_seed=centers_by_seed,
        )
        validation_rows.extend(rows)
        trial = {
            "height_m": float(height), "theta_deg": float(theta),
            **summarize_evaluation(rows, num_ues=validation_environment.num_ues),
        }
        trials.append(trial)
        score = validation_selection_score(trial)
        if best_score is None or score > best_score:
            best_score, selected = score, trial
        print(
            f"K-means validation h={height:g} theta={theta:g} "
            f"mean_return={trial['mean_sum_reward']:.6f}", flush=True,
        )
    if selected is None:
        raise ValueError("No valid height/angle candidates for K-means")
    test_rows, placements = evaluate_kmeans_seed_set(
        test_environment, test_seeds, height=selected["height_m"],
        theta=selected["theta_deg"], evaluation_kind="kmeans_placement_greedy_allocation",
        centers_by_seed={},
    )
    summary = {
        "baseline": "K-means XY placement + distinct-grid assignment + unchanged greedy auction allocation",
        "uses_learned_checkpoint": False,
        "coordination_mode": test_environment.coordination_mode,
        "allocation_action_semantics": test_environment.allocation_action_semantics,
        "resource_constraints": {
            "shared_prb_pool": test_environment.shared_prb_pool,
            "total_system_prbs": test_environment.total_max_prbs,
            "max_prbs_per_action": test_environment.max_prbs_per_action,
            "power_levels_w": test_environment.power_levels_w.tolist(),
            "max_power_w_per_uav": test_environment.max_power_w_per_uav.tolist(),
        },
        "reward": {
            "served_reward": test_environment.served_reward,
            "power_weight": test_environment.power_weight,
            "prb_weight": test_environment.prb_weight,
            "power_penalty_denominator_w": test_environment.power_penalty_denominator_w,
            "prb_penalty_denominator": test_environment.total_max_prbs,
        },
        "kmeans": {
            "initialization": "k-means++ D-squared sampling",
            "num_clusters": validation_environment.num_uavs,
            "seed": KMEANS_SEED, "restarts": KMEANS_RESTARTS,
            "max_iterations": KMEANS_MAX_ITERATIONS,
            "restart_selection": "minimum XY squared-distance inertia, never radio reward",
            "rng": "local NumPy SeedSequence([kmeans_seed, ue_seed]); lexicographically sorted UE XY",
            "grid_assignment": "minimum total squared XY snapping distance; unique grid point per UAV",
            "uav_assignment": "sorted centers mapped to cyclic UAV IDs, matching evaluation rank balance",
            "ue_association": "clustering does not restrict which UAV may bid for a UE",
        },
        "geometry_selection": "one common height/theta pair selected by validation mean scalar team return only; ties prefer lower height then angle",
        "selected_geometry": {key: selected[key] for key in ("height_m", "theta_deg")},
        "validation_seeds": list(map(int, validation_seeds)),
        "validation_trials": trials,
        "validation_evaluation_transitions": sum(int(row["evaluation_transitions"]) for row in validation_rows),
        "test_seeds": list(map(int, test_seeds)),
        "evaluation_transitions": sum(int(row["evaluation_transitions"]) for row in test_rows),
        "wall_time_seconds": time.perf_counter() - started,
        "shuffle_ue_order": test_environment.shuffle_ue_order,
        "runtime_provenance": runtime_provenance(torch.device("cpu")),
        **summarize_evaluation(test_rows, num_ues=test_environment.num_ues),
    }
    for filename, records in (
        ("kmeans_baseline_validation.csv", validation_rows),
        ("kmeans_baseline_metrics.csv", test_rows),
        ("kmeans_baseline_placements.csv", placements),
    ):
        with (output_directory / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (output_directory / "kmeans_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"K-means+greedy test mean_return={summary['mean_sum_reward']:.6f} "
        f"mean_served={summary['mean_served_count']:.2f}/{test_environment.num_ues}",
        flush=True,
    )
    return summary, test_rows


def benchmark_deterministic_inference(
    environment: gym_env,
    learner: StrictLocalIndependentMAPPO,
    seeds: Sequence[int],
    *,
    episodes: int,
    device: torch.device,
) -> dict[str, Any]:
    """Measure complete deterministic episodes without training-only storage."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not seeds:
        raise ValueError("At least one benchmark seed is required")
    rollout_sizes = (
        learner.placement_rollout_size,
        learner.allocation_rollout_size,
    )
    evaluate_one_episode(environment, learner, ue_seed=int(seeds[0]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for index in range(episodes):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        evaluate_one_episode(
            environment,
            learner,
            ue_seed=int(seeds[index % len(seeds)]),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started_at)
    if rollout_sizes != (
        learner.placement_rollout_size,
        learner.allocation_rollout_size,
    ):
        raise RuntimeError("Inference benchmark populated training rollouts")
    cpu_peak_rss_bytes: int | None = None
    try:
        import resource

        cpu_peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            cpu_peak_rss_bytes *= 1024
    except (ImportError, OSError):
        pass
    values = np.asarray(durations, dtype=np.float64)
    return {
        "episodes_measured": int(episodes),
        "warmup_episodes": 1,
        "episode_steps": int(environment.num_ues + 1),
        "num_uavs": int(environment.num_uavs),
        "num_ues": int(environment.num_ues),
        "num_placement_candidates": int(environment.num_solutions),
        "median_episode_latency_seconds": float(np.median(values)),
        "p95_episode_latency_seconds": float(np.percentile(values, 95)),
        "cpu_process_peak_rss_bytes": cpu_peak_rss_bytes,
        "cuda_peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "cuda_peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        ),
        "device": str(device),
        "dtype": "float32",
        "store_for_update": False,
        "counterfactual_previews": 0,
        "memory_note": "CPU RSS is process-lifetime peak, not benchmark-only delta.",
    }


def summarize_evaluation(
    rows: Sequence[dict[str, float | int]],
    *,
    num_ues: int,
) -> dict[str, float | int]:
    """Aggregate one deterministic evaluation per layout."""

    if not rows:
        raise ValueError("At least one evaluation row is required")
    rewards = np.asarray([row["sum_reward"] for row in rows], dtype=np.float64)
    allocation_rewards = np.asarray(
        [row["allocation_return"] for row in rows], dtype=np.float64
    )
    placement_rewards = np.asarray(
        [row["placement_return"] for row in rows], dtype=np.float64
    )
    served = np.asarray([row["served_count"] for row in rows], dtype=np.int64)
    powers = np.asarray(
        [row["total_used_power_w"] for row in rows],
        dtype=np.float64,
    )
    prbs = np.asarray([row["total_used_prbs"] for row in rows], dtype=np.float64)
    perfect_count = int(np.count_nonzero(served == num_ues))
    layout_seeds = {int(row["ue_seed"]) for row in rows}
    return {
        "num_layouts": int(len(layout_seeds)),
        "num_evaluations": int(len(rows)),
        "mean_sum_reward": float(rewards.mean()),
        "std_sum_reward": float(rewards.std(ddof=0)),
        "mean_allocation_return": float(allocation_rewards.mean()),
        "mean_placement_return": float(placement_rewards.mean()),
        "mean_served_count": float(served.mean()),
        "min_served_count": int(served.min()),
        "max_served_count": int(served.max()),
        "perfect_count": perfect_count,
        "perfect_rate": float(perfect_count / len(rows)),
        "mean_total_used_power_w": float(powers.mean()),
        "mean_total_used_prbs": float(prbs.mean()),
    }


def validation_selection_score(
    summary: dict[str, float | int],
) -> tuple[float]:
    """Use the same mean scalar team return optimized during training."""

    score = float(summary["mean_sum_reward"])
    if not np.isfinite(score):
        raise ValueError("Validation mean team return must be finite")
    return (round(score, 12),)


def summarize_uav_placement_metrics(
    environment: gym_env,
    final_info: dict[str, Any] | Sequence[dict[str, Any]],
    placement_return: float,
) -> list[dict[str, float | int]]:
    """Attribute observable coverage, service, reward, and use to each UAV."""

    infos = (
        [final_info]
        if isinstance(final_info, dict)
        else list(final_info)
    )
    if not infos:
        raise ValueError("At least one placement scoring episode is required")
    num_uavs = int(environment.num_uavs)
    coverage = np.asarray(environment.link_features[:, :, 1]) >= 0.5
    if coverage.shape[0] != num_uavs:
        raise RuntimeError("Coverage rows do not match the UAV count")
    coverage_multiplicity = coverage.sum(axis=0)
    served_ues = np.zeros(num_uavs, dtype=np.float64)
    attributed_rewards = np.zeros(num_uavs, dtype=np.float64)

    for info in infos:
        for allocation in info["allocations"]:
            if not allocation or not bool(allocation.get("accepted", False)):
                continue
            uav_id = int(allocation["uav_id"])
            if not 0 <= uav_id < num_uavs:
                raise RuntimeError(f"Invalid winning UAV id: {uav_id}")
            served_ues[uav_id] += 1
            attributed_rewards[uav_id] += (
                float(environment.served_reward)
                - float(allocation["resource_cost"])
            )
    served_ues /= len(infos)
    attributed_rewards /= len(infos)

    if not np.isclose(
        attributed_rewards.sum(), placement_return, rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError("Per-UAV attributed rewards do not sum to episode return")

    placement_indices = [int(value) for value in infos[-1]["placement_indices"]]
    if any(
        [int(value) for value in info["placement_indices"]] != placement_indices
        for info in infos
    ):
        raise RuntimeError("Placement changed inside a frozen-placement cycle")
    used_prbs = np.mean(
        [info["used_prbs_per_uav"] for info in infos], axis=0
    )
    used_power_w = np.mean(
        [info["used_power_w_per_uav"] for info in infos], axis=0
    )
    positions = environment.selected_positions
    if not all(
        len(values) == num_uavs
        for values in (placement_indices, used_prbs, used_power_w, positions)
    ):
        raise RuntimeError("Per-UAV episode diagnostics have inconsistent lengths")

    rows: list[dict[str, float | int]] = []
    for uav_id, position in enumerate(positions):
        own_coverage = coverage[uav_id]
        rows.append(
            {
                "uav_id": uav_id,
                "placement_index": placement_indices[uav_id],
                "x_m": float(position.x_u),
                "y_m": float(position.y_u),
                "height_m": float(position.h_u),
                "theta_deg": float(position.theta_u),
                "coverage_radius_m": float(position.r_u),
                "covered_ues": int(own_coverage.sum()),
                "unique_covered_ues": int(
                    np.logical_and(own_coverage, coverage_multiplicity == 1).sum()
                ),
                "overlap_covered_ues": int(
                    np.logical_and(own_coverage, coverage_multiplicity > 1).sum()
                ),
                "served_ues": float(served_ues[uav_id]),
                "attributed_placement_reward": float(
                    attributed_rewards[uav_id]
                ),
                "used_prbs": float(used_prbs[uav_id]),
                "used_power_w": float(used_power_w[uav_id]),
            }
        )
    return rows


def plot_raw_episode_returns(
    rewards: Sequence[float],
    tick_interval: int,
    output_path: Path,
) -> None:
    """Plot every frozen-placement outer-cycle return."""

    figure, axis = plt.subplots(figsize=(18, 7))
    episodes = np.arange(1, len(rewards) + 1)
    axis.plot(episodes, rewards, linewidth=1.0, alpha=0.85)

    axis.set_title("Frozen-placement outer-cycle reward")
    axis.set_xlabel(f"Outer episode (tick interval = {tick_interval})")
    axis.set_ylabel("Sum reward (episode_return)")
    axis.set_xlim(0.5, max(1.5, len(rewards) + 0.5))
    axis.xaxis.set_major_locator(MultipleLocator(tick_interval))
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_block_mean_returns(
    rewards: Sequence[float],
    block_size: int,
    tick_interval: int,
    output_path: Path,
) -> None:
    """Plot non-overlapping outer-cycle reward means."""

    figure, axis = plt.subplots(figsize=(18, 7))
    values = np.asarray(rewards, dtype=np.float64)
    complete_length = (len(values) // block_size) * block_size
    block_means = (
        values[:complete_length].reshape(-1, block_size).mean(axis=1)
        if complete_length
        else np.asarray([], dtype=np.float64)
    )
    block_end_episodes = np.arange(block_size, complete_length + 1, block_size)
    axis.plot(
        block_end_episodes,
        block_means,
        linewidth=1.6,
        marker="o",
        markersize=2.8,
    )

    axis.set_title(
        f"Mean sum reward per non-overlapping {block_size}-episode block"
    )
    axis.set_xlabel(
        f"End episode of each block (tick interval = {tick_interval})"
    )
    axis.set_ylabel(f"Mean sum reward over {block_size} episodes")
    axis.set_xlim(0.5, max(1.5, len(values) + 0.5))
    axis.xaxis.set_major_locator(MultipleLocator(tick_interval))
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_plots(
    rewards: Sequence[float],
    args: argparse.Namespace,
    output_directory: Path,
) -> None:
    """Write both requested reward figures from the current training history."""

    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1, PLACEMENTS_PER_GLOBAL).mean(axis=1)
    plot_raw_episode_returns(
        rewards,
        args.plot_tick_interval,
        output_directory / "episode_sum_reward_by_seed.png",
    )
    plot_block_mean_returns(
        rewards,
        args.reward_average_window,
        args.plot_tick_interval,
        output_directory
        / f"placement_{args.reward_average_window}_episode_mean_reward_by_seed.png",
    )


def validation_mean_series(
    rewards_by_seed: dict[int, list[tuple[int, float]]],
    seed_order: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return one ensemble-mean validation value per evaluation episode."""

    rewards_by_episode: dict[int, list[float]] = defaultdict(list)
    for ue_seed in seed_order:
        for global_episode, reward in rewards_by_seed.get(int(ue_seed), []):
            rewards_by_episode[int(global_episode)].append(float(reward))
    incomplete = {
        episode: len(rewards)
        for episode, rewards in rewards_by_episode.items()
        if len(rewards) != len(seed_order)
    }
    if incomplete:
        raise RuntimeError(
            "Validation mean requires one result per seed at every episode: "
            f"{incomplete}"
        )
    global_episodes = np.asarray(sorted(rewards_by_episode), dtype=np.int64)
    mean_rewards = np.asarray(
        [np.mean(rewards_by_episode[episode]) for episode in global_episodes],
        dtype=np.float64,
    )
    return global_episodes, mean_rewards


def plot_validation_returns(
    rewards_by_seed: dict[int, list[tuple[int, float]]],
    seed_order: Sequence[int],
    tick_interval: int,
    output_path: Path,
) -> None:
    """Plot one mean return over the fixed validation seed ensemble."""

    figure, axis = plt.subplots(figsize=(18, 7))
    global_episodes, mean_rewards = validation_mean_series(
        rewards_by_seed,
        seed_order,
    )
    axis.plot(
        global_episodes,
        mean_rewards,
        linewidth=1.8,
        marker="o",
        markersize=4.0,
    )

    axis.set_title("Mean deterministic validation sum reward")
    axis.set_xlabel("Global training episode")
    axis.set_ylabel(f"Mean sum reward over {len(seed_order)} validation seeds")
    axis.xaxis.set_major_locator(MultipleLocator(max(1, tick_interval)))
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train MAPPO and select/test it on disjoint UE layout seeds."
        )
    )
    parser.add_argument(
        "--train-seeds",
        type=parse_seed_list,
        default=DEFAULT_TRAIN_SEEDS,
        help="Comma-separated training UE layout seeds.",
    )
    parser.add_argument(
        "--validation-seeds",
        type=parse_seed_list,
        default=DEFAULT_VALIDATION_SEEDS,
        help="Comma-separated validation UE layout seeds.",
    )
    parser.add_argument(
        "--diagnostic-seeds",
        type=parse_seed_list,
        default=DEFAULT_DIAGNOSTIC_SEEDS,
        help=(
            "Comma-separated former-test seeds used only for diagnostic "
            "comparison after model selection."
        ),
    )
    parser.add_argument(
        "--test-seeds",
        type=parse_seed_list,
        default=DEFAULT_TEST_SEEDS,
        help="Comma-separated held-out test UE layout seeds.",
    )
    parser.add_argument("--episodes-per-seed", type=int, default=1,
                        help="Visits per training map; each visit generates five placement samples.")
    parser.add_argument(
        "--placement-episodes",
        type=int,
        default=710,
        help="Global episodes; each uses 2 maps and all 5 cyclic UAV orders per map.",
    )
    parser.add_argument(
        "--placement-only-episodes",
        type=int,
        default=DEFAULT_PLACEMENT_ONLY_EPISODES,
        help=(
            "Final global episodes trained against one completely frozen "
            "allocator; must be divisible by placement-batch-episodes."
        ),
    )
    parser.add_argument("--training-seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--coordination-mode",
        choices=("full", "basic", "none"),
        default="none",
        help=(
            "none: shared 135 PRB, learned PRB/power, own remaining power and "
            "shared PRB balance only; full/basic retain legacy Pareto allocation."
        ),
    )
    parser.add_argument("--served-reward", type=float, default=1.0)
    parser.add_argument(
        "--power-weight", type=float, default=None,
        help=("Coefficient multiplying winner_power_w / power_budget "
              "(2 W for none, team budget for full/basic; default 4/27)."),
    )
    parser.add_argument(
        "--prb-weight", type=float, default=None,
        help=("Coefficient multiplying winner_prbs / total_system_prbs "
              "(default 1 for none, 4/27 for full/basic)."),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "Resume an exact run from a completed frozen-batch checkpoint. "
            "Mid-inner-loop/model-selection checkpoints are rejected."
        ),
    )
    parser.add_argument(
        "--no-greedy-baseline",
        action="store_true",
        help="Skip the test-map policy-placement/greedy-allocation baseline.",
    )
    parser.add_argument(
        "--no-kmeans-baseline",
        action="store_true",
        help="Skip the independent K-means-placement/greedy-allocation baseline.",
    )
    parser.add_argument(
        "--kmeans-baseline-only",
        action="store_true",
        help="Run validation selection and test of K-means+greedy without training or a checkpoint.",
    )
    parser.add_argument(
        "--benchmark-inference-episodes",
        type=int,
        default=0,
        help=(
            "Measured deterministic episodes after testing; 0 disables the "
            "latency/memory benchmark."
        ),
    )
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=0.02,
                        help="Stop each allocation actor separately at this KL; placement stops when any actor exceeds it.")
    parser.add_argument(
        "--output-dir",
        default="training_outputs_none_2maps_5orders_no_rollback",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--plot-tick-interval", type=int, default=20)
    parser.add_argument("--reward-average-window", type=int, default=10)
    parser.add_argument("--evaluation-every", type=int, default=20)
    parser.add_argument("--placement-batch-episodes", type=int, default=10,
                        help="Global episodes accumulated per placement PPO update (10 placements per global).")
    parser.add_argument("--placement-minibatch-size", type=int, default=10)
    parser.add_argument("--placement-ppo-epochs", type=int, default=4)
    parser.add_argument(
        "--allocation-batch-episodes",
        type=int,
        default=10,
        help=(
            "Frozen map/placement episodes collected under one policy before "
            "each PPO update; must be 10 (2 maps times 5 cyclic orders)."
        ),
    )
    parser.add_argument("--allocation-ppo-epochs", type=int, default=4)
    parser.add_argument("--allocation-minibatch-size", type=int, default=500)
    parser.add_argument(
        "--allocation-inner-updates", "--allocation-max-inner-updates",
        dest="allocation_max_inner_updates",
        type=int,
        default=50,
        help="Maximum allocation PPO updates per global episode; final training state is retained.",
    )
    parser.add_argument(
        "--allocation-inner-patience", type=int, default=10,
        help="Stop inner allocation after this many consecutive evaluations without a new best mean reward; no rollback.",
    )
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--actor-lr", type=float, default=2e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--placement-entropy-coef", type=float, default=5e-4)
    parser.add_argument(
        "--placement-popart-beta",
        type=float,
        default=1e-4,
        help="Per-target EMA step size for placement Pop-Art statistics.",
    )
    parser.add_argument("--allocation-entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--allocation-entropy-final-coef",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--allocation-entropy-anneal-start-episode",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--allocation-entropy-anneal-end-episode",
        type=int,
        default=3000,
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--no-shuffle-ue-order",
        action="store_true",
        help="Use make_ue generation order instead of each layout's fixed shuffle.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject invalid schedules before creating a model or output directory."""

    if args.kmeans_baseline_only and (args.resume or args.no_kmeans_baseline):
        raise ValueError("kmeans-baseline-only cannot resume training or skip K-means")
    for name in ("served_reward", "power_weight", "prb_weight"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
    positive_values = {
        "torch_num_threads": args.torch_num_threads,
        "episodes_per_seed": args.episodes_per_seed,
        "placement_episodes": args.placement_episodes,
        "progress_every": args.progress_every,
        "checkpoint_every": args.checkpoint_every,
        "evaluation_every": args.evaluation_every,
        "plot_tick_interval": args.plot_tick_interval,
        "reward_average_window": args.reward_average_window,
        "placement_batch_episodes": args.placement_batch_episodes,
        "placement_minibatch_size": args.placement_minibatch_size,
        "placement_ppo_epochs": args.placement_ppo_epochs,
        "allocation_batch_episodes": args.allocation_batch_episodes,
        "allocation_ppo_epochs": args.allocation_ppo_epochs,
        "allocation_minibatch_size": args.allocation_minibatch_size,
        "allocation_max_inner_updates": args.allocation_max_inner_updates,
        "allocation_inner_patience": args.allocation_inner_patience,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if not np.isfinite(args.target_kl) or args.target_kl <= 0:
        raise ValueError("target_kl must be finite and positive")
    if any(
        not np.isfinite(value) or value < 0.0
        for value in (
            args.placement_entropy_coef,
            args.allocation_entropy_coef,
            args.allocation_entropy_final_coef,
        )
    ):
        raise ValueError("Entropy coefficients must be non-negative")
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (args.actor_lr, args.critic_lr)
    ):
        raise ValueError(
            "Actor and critic learning rates must be finite and positive"
        )
    if args.allocation_entropy_final_coef > args.allocation_entropy_coef:
        raise ValueError("Final allocation entropy cannot exceed its initial value")
    if not np.isfinite(args.placement_popart_beta) or not (
        0.0 < args.placement_popart_beta <= 1.0
    ):
        raise ValueError("placement_popart_beta must be in (0, 1]")
    if not (
        0
        <= args.allocation_entropy_anneal_start_episode
        < args.allocation_entropy_anneal_end_episode
    ):
        raise ValueError("Allocation entropy annealing episodes are invalid")
    if args.placement_minibatch_size > args.placement_batch_episodes * PLACEMENTS_PER_GLOBAL:
        raise ValueError(
            "placement_minibatch_size cannot exceed the placement sample batch"
        )
    unbalanced_evaluation_splits = [
        name
        for name, seeds in (
            ("validation_seeds", args.validation_seeds),
            ("diagnostic_seeds", args.diagnostic_seeds),
            ("test_seeds", args.test_seeds),
        )
        if len(seeds) % 5
    ]
    if unbalanced_evaluation_splits:
        raise ValueError(
            "Validation, diagnostic, and test seed counts must be divisible "
            "by five so cyclic placement evaluation is rank-balanced: "
            f"{unbalanced_evaluation_splits}"
        )
    if args.embed_dim % args.num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if args.placement_episodes % args.placement_batch_episodes:
        raise ValueError(
            "placement_episodes must be divisible by placement_batch_episodes"
        )
    if args.placement_only_episodes < 0:
        raise ValueError("placement_only_episodes cannot be negative")
    if args.placement_only_episodes % args.placement_batch_episodes:
        raise ValueError(
            "placement_only_episodes must be divisible by "
            "placement_batch_episodes"
        )
    warmup_samples = ALLOCATION_WARMUP_BATCHES * args.placement_batch_episodes
    joint_training_samples = (
        args.placement_episodes
        - warmup_samples
        - args.placement_only_episodes
    )
    if joint_training_samples <= 0:
        raise ValueError(
            "The schedule must contain at least one joint placement+allocation "
            "batch after warm-up"
        )
    if MAPS_PER_GLOBAL > len(args.train_seeds):
        raise ValueError(
            "The number of training seeds must cover two maps per global"
        )
    available_episodes = len(args.train_seeds) * args.episodes_per_seed
    if args.placement_episodes * MAPS_PER_GLOBAL > available_episodes:
        raise ValueError(
            "The training seed schedule is shorter than placement_episodes: "
            f"available maps={available_episodes}, requested maps={args.placement_episodes * MAPS_PER_GLOBAL}"
        )
    if args.allocation_batch_episodes != PLACEMENTS_PER_GLOBAL:
        raise ValueError(
            "allocation_batch_episodes must equal 10 (2 maps times 5 orders)"
        )
    if args.benchmark_inference_episodes < 0:
        raise ValueError("benchmark_inference_episodes cannot be negative")
    if float(args.gamma) != 1.0:
        raise ValueError(
            "gamma must be exactly 1.0 so training, snapshot selection, and "
            "evaluation optimize the same undiscounted team return"
        )


def _read_csv_at_resume_boundary(
    path: Path,
    fieldnames: Sequence[str],
    completed_episode: int,
    *,
    committed_row_count: int | None = None,
) -> list[dict[str, str]]:
    """Read the committed CSV prefix without modifying the source file."""

    if not path.is_file():
        raise FileNotFoundError(f"Resume log not found: {path}")
    if committed_row_count is not None and (
        isinstance(committed_row_count, bool)
        or not isinstance(committed_row_count, int)
        or committed_row_count < 0
    ):
        raise ValueError("Checkpoint CSV row count must be a non-negative integer")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fieldnames):
            raise ValueError(f"Resume log schema does not match: {path}")
        rows: list[dict[str, str]] = []
        committed_reader = reader if committed_row_count is None else islice(reader, committed_row_count)
        for row_number, row in enumerate(committed_reader, start=2):
            if None in row:
                raise ValueError(
                    f"Resume log row {row_number} has extra fields: {path}"
                )
            try:
                global_episode = int(row["global_episode"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Resume log row {row_number} has an invalid "
                    f"global_episode: {path}"
                ) from exc
            if global_episode <= 0:
                raise ValueError(
                    f"Resume log row {row_number} has a non-positive "
                    f"global_episode: {path}"
                )
            if global_episode <= completed_episode:
                rows.append(dict(row))
        if committed_row_count is not None and len(rows) != committed_row_count:
            raise ValueError(f"Resume log is missing checkpointed rows: {path}")
    return rows


def _stage_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> Path:
    """Write one replacement beside its destination and return its path."""

    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            writer = csv.DictWriter(temporary, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            temporary.flush()
            os.fsync(temporary.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _truncate_csv_at_resume_boundary(
    path: Path,
    fieldnames: Sequence[str],
    completed_episode: int,
) -> int:
    """Atomically discard CSV rows after one completed outer boundary."""

    rows = _read_csv_at_resume_boundary(path, fieldnames, completed_episode)
    staged = _stage_csv(path, fieldnames, rows)
    staged.replace(path)
    return len(rows)


def _read_diagnostics_at_resume_boundary(
    path: Path,
    diagnostic_counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Read and validate the committed diagnostics prefix without writes."""

    if not path.is_file():
        raise FileNotFoundError(f"Resume diagnostics not found: {path}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in diagnostic_counts.values()
    ):
        raise ValueError("Checkpoint diagnostic counts must be non-negative integers")
    retained: list[dict[str, Any]] = []
    observed = {phase: [] for phase in diagnostic_counts}
    expected_rows = sum(diagnostic_counts.values())
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        # A crash may leave arbitrary partial JSON after the checkpointed prefix.
        if len(retained) == expected_rows:
            break
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Resume diagnostics line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Resume diagnostics line {line_number} is not an object"
            )
        phase = payload.get("phase")
        if phase not in diagnostic_counts:
            raise ValueError(
                f"Resume diagnostics line {line_number} has unknown phase {phase!r}"
            )
        update = payload.get("update")
        if isinstance(update, bool) or not isinstance(update, int) or update <= 0:
            raise ValueError(
                f"Resume diagnostics line {line_number} has an invalid update"
            )
        if update <= diagnostic_counts[phase]:
            json.dumps(payload, allow_nan=False)
            retained.append(payload)
            observed[phase].append(update)
    expected = {
        phase: list(range(1, count + 1))
        for phase, count in diagnostic_counts.items()
    }
    if observed != expected:
        raise ValueError(
            "Resume diagnostics do not contain every checkpointed update: "
            f"expected={expected}, observed={observed}"
        )
    return retained


def _stage_diagnostics(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Stage normalized JSONL beside its destination."""

    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            for row in rows:
                temporary.write(json.dumps(row, allow_nan=False) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _truncate_diagnostics_at_resume_boundary(
    path: Path,
    diagnostic_counts: dict[str, int],
) -> None:
    """Atomically discard diagnostics after one completed outer boundary."""

    rows = _read_diagnostics_at_resume_boundary(path, diagnostic_counts)
    staged = _stage_diagnostics(path, rows)
    staged.replace(path)


def _resume_row_int(row: dict[str, str], field: str, log_name: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Resume {log_name} row has an invalid integer {field!r}"
        ) from exc


def _resume_row_float(row: dict[str, str], field: str, log_name: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Resume {log_name} row has an invalid float {field!r}"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(
            f"Resume {log_name} row has a non-finite {field!r}"
        )
    return value


def _validate_resume_log_rows(
    logs: dict[str, list[dict[str, str]]],
    *,
    completed_episode: int,
    episode_seed_schedule: Sequence[int],
    training_rewards: Sequence[float],
    rewards_by_seed: dict[int, list[float]],
    validation_rewards_by_seed: dict[int, list[tuple[int, float]]],
    validation_seed_order: Sequence[int],
    allocation_episodes_used: int,
    num_uavs: int,
    placement_batch_episodes: int,
    warmup_samples: int,
    joint_training_end: int,
    total_episodes: int,
    placements_per_global: int = 1,
) -> None:
    """Prove that every retained log row matches checkpointed trainer state."""

    required_logs = {"training", "allocation", "validation", "uav_placement"}
    if set(logs) != required_logs:
        raise ValueError("Resume log collection has an invalid schema")
    if len(training_rewards) != completed_episode * placements_per_global:
        raise ValueError("Checkpoint training history length is inconsistent")
    if any(not np.isfinite(value) for value in training_rewards):
        raise ValueError("Checkpoint training rewards contain non-finite values")

    training_rows = logs["training"]
    if len(training_rows) != completed_episode * placements_per_global:
        raise ValueError(
            "Resume training log row count does not match the checkpoint"
        )
    observed_by_seed: dict[int, list[float]] = defaultdict(list)
    episode_count_by_seed: dict[int, int] = defaultdict(int)
    for expected_sample, (row, reward) in enumerate(
        zip(training_rows, training_rewards), start=1
    ):
        expected_episode = (expected_sample - 1) // placements_per_global + 1
        if placements_per_global > 1 and (
            _resume_row_int(row, "placement_sample_id", "training") != expected_sample
            or _resume_row_int(row, "placement_index_in_global", "training") != (expected_sample - 1) % placements_per_global + 1
        ):
            raise ValueError("Resume placement sample index is inconsistent")
        if _resume_row_int(row, "global_episode", "training") != expected_episode:
            raise ValueError("Resume training log has duplicate or missing episodes")
        expected_seed = int(episode_seed_schedule[expected_sample - 1])
        seed = _resume_row_int(row, "ue_seed", "training")
        if seed != expected_seed:
            raise ValueError("Resume training log seed order does not match")
        episode_count_by_seed[seed] += 1
        if _resume_row_int(row, "episode_in_seed", "training") != (
            episode_count_by_seed[seed]
        ):
            raise ValueError("Resume training log episode_in_seed is inconsistent")
        logged_reward = _resume_row_float(row, "sum_reward", "training")
        if logged_reward != float(reward):
            raise ValueError("Resume training log reward history does not match")
        expected_phase = training_phase_for_episode(
            expected_episode,
            warmup_samples=warmup_samples,
            joint_training_end=joint_training_end,
            total_episodes=total_episodes,
        )
        expected_order = cyclic_placement_order(
            (expected_sample - 1) % (placements_per_global if placements_per_global > 1 else placement_batch_episodes),
            num_uavs,
        )
        if row.get("training_phase") != expected_phase:
            raise ValueError("Resume training phase does not match its episode")
        if row.get("placement_order") != format_placement_order(expected_order):
            raise ValueError("Resume placement order does not match its episode")
        expected_allocation_active = int(
            expected_phase != "placement_only_allocation_frozen"
        )
        if _resume_row_int(
            row, "allocation_training_active", "training"
        ) != expected_allocation_active:
            raise ValueError("Resume allocation phase flag is inconsistent")
        if _resume_row_int(
            row, "allocation_state_unchanged", "training"
        ) != (1 - expected_allocation_active):
            raise ValueError("Resume allocation freeze flag is inconsistent")
        observed_by_seed[seed].append(logged_reward)
    normalized_rewards_by_seed = {
        int(seed): [float(value) for value in values]
        for seed, values in rewards_by_seed.items()
        if values
    }
    if dict(observed_by_seed) != normalized_rewards_by_seed:
        raise ValueError("Checkpoint per-seed training rewards are inconsistent")

    allocation_rows = logs["allocation"]
    if len(allocation_rows) != allocation_episodes_used:
        raise ValueError(
            "Resume allocation log row count does not match the checkpoint"
        )
    for expected_allocation_episode, row in enumerate(allocation_rows, start=1):
        if _resume_row_int(row, "allocation_episode", "allocation") != (
            expected_allocation_episode
        ):
            raise ValueError(
                "Resume allocation log has duplicate or missing allocation episodes"
            )
        global_episode = _resume_row_int(row, "global_episode", "allocation")
        if not 1 <= global_episode <= completed_episode:
            raise ValueError("Resume allocation log references an invalid episode")
        sample_id = (_resume_row_int(row, "placement_sample_id", "allocation")
                     if placements_per_global > 1 else global_episode)
        if (sample_id - 1) // placements_per_global + 1 != global_episode:
            raise ValueError("Resume allocation sample/global indices disagree")
        if placements_per_global > 1 and _resume_row_int(row, "placement_index_in_global", "allocation") != (sample_id - 1) % placements_per_global + 1:
            raise ValueError("Resume allocation placement index is inconsistent")
        if _resume_row_int(row, "ue_seed", "allocation") != int(
            episode_seed_schedule[sample_id - 1]
        ):
            raise ValueError("Resume allocation log seed does not match its episode")
        if global_episode > joint_training_end:
            raise ValueError(
                "Resume allocation log contains placement-only training"
            )

    expected_validation: list[tuple[int, int, float]] = []
    validation_order = {
        int(seed): index for index, seed in enumerate(validation_seed_order)
    }
    if len(validation_order) != len(validation_seed_order):
        raise ValueError("Validation seed order contains duplicates")
    seen_validation_keys: set[tuple[int, int]] = set()
    for seed, points in validation_rewards_by_seed.items():
        seed = int(seed)
        if seed not in validation_order:
            raise ValueError("Checkpoint contains an unknown validation seed")
        for episode, reward in points:
            key = (int(episode), seed)
            if key in seen_validation_keys or not 1 <= key[0] <= completed_episode:
                raise ValueError("Checkpoint validation history is inconsistent")
            if not np.isfinite(reward):
                raise ValueError("Checkpoint validation history is non-finite")
            seen_validation_keys.add(key)
            expected_validation.append((key[0], seed, float(reward)))
    expected_validation.sort(
        key=lambda item: (item[0], validation_order[item[1]])
    )
    validation_rows = logs["validation"]
    if len(validation_rows) != len(expected_validation):
        raise ValueError(
            "Resume validation log row count does not match the checkpoint"
        )
    for row, (episode, seed, reward) in zip(
        validation_rows, expected_validation
    ):
        actual = (
            _resume_row_int(row, "global_episode", "validation"),
            _resume_row_int(row, "ue_seed", "validation"),
            _resume_row_float(row, "sum_reward", "validation"),
        )
        if actual != (episode, seed, reward):
            raise ValueError("Resume validation log does not match checkpoint history")
        if row.get("evaluation_kind") != "validation":
            raise ValueError("Resume validation log has the wrong evaluation kind")
        expected_order = cyclic_placement_order(
            validation_order[seed], num_uavs
        )
        if row.get("placement_order") != format_placement_order(expected_order):
            raise ValueError("Resume validation placement order is inconsistent")

    placement_rows = logs["uav_placement"]
    expected_placements = completed_episode * placements_per_global * num_uavs
    if len(placement_rows) != expected_placements:
        raise ValueError(
            "Resume UAV placement log row count does not match the checkpoint"
        )
    placement_keys: set[tuple[int, int]] = set()
    for row in placement_rows:
        global_episode = _resume_row_int(row, "global_episode", "UAV placement")
        uav_id = _resume_row_int(row, "uav_id", "UAV placement")
        if not 1 <= global_episode <= completed_episode or not 0 <= uav_id < num_uavs:
            raise ValueError("Resume UAV placement log references an invalid index")
        sample_id = (_resume_row_int(row, "placement_sample_id", "UAV placement")
                     if placements_per_global > 1 else global_episode)
        if (sample_id - 1) // placements_per_global + 1 != global_episode:
            raise ValueError("Resume UAV placement sample/global indices disagree")
        if placements_per_global > 1 and _resume_row_int(row, "placement_index_in_global", "UAV placement") != (sample_id - 1) % placements_per_global + 1:
            raise ValueError("Resume UAV placement index is inconsistent")
        key = (sample_id, uav_id)
        if key in placement_keys:
            raise ValueError("Resume UAV placement log contains duplicate rows")
        placement_keys.add(key)
        if _resume_row_int(row, "ue_seed", "UAV placement") != int(
            episode_seed_schedule[sample_id - 1]
        ):
            raise ValueError("Resume UAV placement seed does not match its episode")
        expected_order = cyclic_placement_order(
            (sample_id - 1) % (placements_per_global if placements_per_global > 1 else placement_batch_episodes),
            num_uavs,
        )
        expected_rank = expected_order.index(uav_id) + 1
        if _resume_row_int(
            row, "selection_rank", "UAV placement"
        ) != expected_rank:
            raise ValueError("Resume UAV selection rank is inconsistent")
    expected_placement_keys = {
        (episode, uav_id)
        for episode in range(1, completed_episode * placements_per_global + 1)
        for uav_id in range(num_uavs)
    }
    if placement_keys != expected_placement_keys:
        raise ValueError("Resume UAV placement log has missing rows")


def _commit_resume_logs(
    csv_logs: Sequence[
        tuple[Path, Sequence[str], Sequence[dict[str, str]]]
    ],
    diagnostics_path: Path,
    diagnostics: Sequence[dict[str, Any]],
) -> None:
    """Stage every validated replacement before changing any run artifact."""

    staged: list[tuple[Path, Path]] = []
    try:
        for path, fieldnames, rows in csv_logs:
            staged.append((_stage_csv(path, fieldnames, rows), path))
        staged.append(
            (_stage_diagnostics(diagnostics_path, diagnostics), diagnostics_path)
        )
        for temporary_path, destination in staged:
            temporary_path.replace(destination)
    finally:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)


def _archive_stale_resume_artifacts(
    output_directory: Path,
    checkpoint_directory: Path,
    *,
    completed_episode: int,
    total_episodes: int,
    resume_path: Path,
    joint_training_end: int | None = None,
) -> list[Path]:
    """Move artifacts newer than a rollback boundary into a recoverable archive."""

    if completed_episode >= total_episodes:
        return []
    candidates = {
        output_directory / name
        for name in (
            "best_validation_summary.json",
            "diagnostic_metrics.csv",
            "diagnostic_summary.json",
            "test_metrics.csv",
            "test_summary.json",
            "greedy_baseline_metrics.csv",
            "greedy_baseline_summary.json",
            "kmeans_baseline_validation.csv",
            "kmeans_baseline_metrics.csv",
            "kmeans_baseline_placements.csv",
            "kmeans_baseline_summary.json",
            "inference_benchmark.json",
            "training_budget_summary.json",
        )
    }
    candidates.update(output_directory.glob("*.png"))
    candidates.update(output_directory.glob("best_*_validation_summary.json"))
    if joint_training_end is None or completed_episode < joint_training_end:
        candidates.add(output_directory / "latest_inner_best_summary.json")
        candidates.add(checkpoint_directory / "latest_inner_best_model.pt")
        candidates.add(output_directory / "latest_inner_summary.json")
        candidates.add(checkpoint_directory / "latest_inner_model.pt")
    candidates.add(checkpoint_directory / "final_model.pt")
    candidates.add(checkpoint_directory / ".latest_resume.pt.tmp")
    if (
        joint_training_end is not None
        and completed_episode < joint_training_end
    ):
        candidates.add(checkpoint_directory / "joint_training_end.pt")
    latest_resume = checkpoint_directory / "latest_resume.pt"
    if latest_resume.resolve() != resume_path.resolve():
        candidates.add(latest_resume)
        candidates.add(output_directory / "latest_resume_summary.json")
    for checkpoint in checkpoint_directory.glob("*.pt"):
        marker = "global_episode_"
        if marker not in checkpoint.stem:
            continue
        try:
            episode = int(checkpoint.stem.rsplit(marker, 1)[1])
        except ValueError:
            continue
        if episode > completed_episode:
            candidates.add(checkpoint)

    existing = sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.resolve() != resume_path.resolve()
        ),
        key=lambda path: str(path),
    )
    if not existing:
        return []
    archive_root = (
        output_directory
        / "stale_resume_artifacts"
        / f"after_global_episode_{completed_episode}"
    )
    suffix = 2
    while archive_root.exists():
        archive_root = archive_root.with_name(
            f"after_global_episode_{completed_episode}_{suffix}"
        )
        suffix += 1
    moved: list[Path] = []
    for source in existing:
        destination = archive_root / source.relative_to(output_directory)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        moved.append(destination)
    (archive_root / "manifest.json").write_text(
        json.dumps(
            {
                "resume_checkpoint": str(resume_path),
                "completed_global_episode": completed_episode,
                "archived_paths": [
                    str(path.relative_to(archive_root)) for path in moved
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return moved


def main(*, args=None, learner_class=StrictLocalIndependentMAPPO,
         learner_kwargs=None, configuration_overrides=None, source_paths=(),
         environment_class=gym_env) -> None:
    """Run the shared schedule; optional injection supports the separate MAPPO launcher."""
    if args is None:
        args = build_argument_parser().parse_args()
    validate_arguments(args)
    configure_deterministic_training()
    process_started_at = time.perf_counter()
    resumed_elapsed_wall_time = 0.0
    torch.set_num_threads(args.torch_num_threads)
    train_seeds = tuple(int(value) for value in args.train_seeds)
    validation_seeds = tuple(int(value) for value in args.validation_seeds)
    diagnostic_seeds = tuple(int(value) for value in args.diagnostic_seeds)
    test_seeds = tuple(int(value) for value in args.test_seeds)
    validate_seed_splits(
        train_seeds,
        validation_seeds,
        test_seeds,
        diagnostic_seeds,
    )
    output_directory = Path(args.output_dir).expanduser().resolve()
    if (
        args.resume is None
        and output_directory.exists()
        and any(output_directory.iterdir())
    ):
        raise FileExistsError(
            "A fresh run requires an empty output directory; choose a new "
            f"--output-dir instead of overwriting {output_directory}"
        )
    checkpoint_directory = output_directory / "checkpoints"
    diagnostics_path = output_directory / "learning_diagnostics.jsonl"
    diagnostic_counts = {"allocation": 0, "placement": 0}

    def log_learning_diagnostics(phase, metrics):
        diagnostic_counts[phase] += 1
        payload = {"phase": phase, "update": diagnostic_counts[phase], **metrics}
        with diagnostics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=False) + "\n")
        print(f"[{phase} update {diagnostic_counts[phase]}] "
              f"KL={metrics.get(phase + '_policy_kl', 0):.6f}", flush=True)

    seed_training_randomness(args.training_seed)
    device = choose_device(args.device)
    environment = environment_class(
        seed=train_seeds,
        shuffle_ue_order=not args.no_shuffle_ue_order,
        coordination_mode=args.coordination_mode,
        served_reward=args.served_reward,
        power_weight=args.power_weight,
        prb_weight=args.prb_weight,
    )
    validation_environment = environment_class(
        seed=validation_seeds,
        shuffle_ue_order=not args.no_shuffle_ue_order,
        coordination_mode=args.coordination_mode,
        served_reward=args.served_reward,
        power_weight=args.power_weight,
        prb_weight=args.prb_weight,
    )
    diagnostic_environment = environment_class(
        seed=diagnostic_seeds,
        shuffle_ue_order=not args.no_shuffle_ue_order,
        coordination_mode=args.coordination_mode,
        served_reward=args.served_reward,
        power_weight=args.power_weight,
        prb_weight=args.prb_weight,
    )
    test_environment = environment_class(
        seed=test_seeds,
        shuffle_ue_order=not args.no_shuffle_ue_order,
        coordination_mode=args.coordination_mode,
        served_reward=args.served_reward,
        power_weight=args.power_weight,
        prb_weight=args.prb_weight,
    )

    if args.kmeans_baseline_only:
        output_directory.mkdir(parents=True, exist_ok=True)
        run_kmeans_baseline(
            validation_environment, test_environment, validation_seeds,
            test_seeds, output_directory,
        )
        print(f"K-means baseline complete; no training was run. Outputs: {output_directory}")
        return

    learner = learner_class.from_env(
        environment,
        device=device,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        placement_entropy_coef=args.placement_entropy_coef,
        placement_popart_beta=args.placement_popart_beta,
        allocation_entropy_coef=args.allocation_entropy_coef,
        gamma=args.gamma,
        placement_batch_episodes=args.placement_batch_episodes * PLACEMENTS_PER_GLOBAL,
        placement_minibatch_size=args.placement_minibatch_size,
        placement_ppo_epochs=args.placement_ppo_epochs,
        allocation_ppo_epochs=args.allocation_ppo_epochs,
        allocation_minibatch_size=args.allocation_minibatch_size,
        target_kl=args.target_kl,
        diagnostics_callback=log_learning_diagnostics,
        **(learner_kwargs or {}),
    )
    # Keep rollout sampling comparable when mode-specific networks consume a
    # different number of random draws during parameter initialization.
    seed_training_randomness(args.training_seed)
    available_seed_schedule = build_interleaved_seed_schedule(
        train_seeds,
        args.episodes_per_seed,
        args.training_seed,
    )
    total_episodes = args.placement_episodes
    if len(available_seed_schedule) < total_episodes * MAPS_PER_GLOBAL:
        raise ValueError(
            "The UE seed schedule does not contain enough distinct maps for "
            f"{total_episodes * MAPS_PER_GLOBAL} map episodes"
        )
    episode_seed_schedule = tuple(
        seed
        for seed in available_seed_schedule[:total_episodes * MAPS_PER_GLOBAL]
        for _ in range(PLACEMENTS_PER_MAP)
    )
    warmup_placement_samples = min(
        total_episodes,
        ALLOCATION_WARMUP_BATCHES * args.placement_batch_episodes,
    )
    placement_only_samples = int(args.placement_only_episodes)
    joint_training_end = total_episodes - placement_only_samples
    joint_training_samples = joint_training_end - warmup_placement_samples
    placement_training_samples = total_episodes - warmup_placement_samples
    rewards_by_seed: dict[int, list[float]] = defaultdict(list)
    training_rewards: list[float] = []
    validation_rewards_by_seed: dict[int, list[tuple[int, float]]] = defaultdict(
        list
    )

    provenance = runtime_provenance(device)
    for source_path in source_paths:
        source_path = Path(source_path).resolve()
        provenance["source_sha256"][source_path.name] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    configuration_fingerprint = resume_configuration_fingerprint(args)
    coordination_scope = {
        "full": (
            "local_per_uav_plus_broadcast_8_scalar_coordination_summary_"
            "including_future_exclusive_ue_resource_reserve"
        ),
        "basic": (
            "local_per_uav_plus_broadcast_5_scalar_link_and_team_resource_"
            "summary_with_3_reserved_slots_zeroed"
        ),
        "none": (
            "covered UE set attention plus compact local MLP: shared PRB ratio, "
            "own remaining power ratio, current UE QoS, current link and UAV ID; "
            "local MLP residual bypasses context mixing; own UAV geometry, "
            "current UE coordinates, fixed limits and duplicate broadcast are "
            "excluded from the local MLP; local UE set features stay unchanged"
        ),
    }[args.coordination_mode]
    configuration = vars(args).copy()
    configuration["resume"] = (
        str(args.resume.expanduser().resolve()) if args.resume else None
    )
    configuration.update(
        {
            "train_seeds": list(train_seeds),
            "validation_seeds": list(validation_seeds),
            "diagnostic_seeds": list(diagnostic_seeds),
            "test_seeds": list(test_seeds),
            "environment_file": str(
                Path(__file__).resolve().with_name("ver2.py")
            ),
            "resolved_device": str(device),
            "configuration_fingerprint": configuration_fingerprint,
            "runtime_provenance": provenance,
            "total_global_episodes": total_episodes,
            "total_placement_episodes": total_episodes * PLACEMENTS_PER_GLOBAL,
            "maps_per_global": MAPS_PER_GLOBAL,
            "placements_per_map": PLACEMENTS_PER_MAP,
            "placement_samples_per_global": PLACEMENTS_PER_GLOBAL,
            "placement_ppo_batch_samples": learner.placement_batch_episodes,
            "training_map_episodes": total_episodes * MAPS_PER_GLOBAL,
            "allocation_warmup_batches": warmup_placement_samples,
            "allocation_warmup_global_episodes": warmup_placement_samples,
            "joint_training_global_episodes": joint_training_samples,
            "placement_only_global_episodes": placement_only_samples,
            "allocation_warmup_placement_samples": warmup_placement_samples * PLACEMENTS_PER_GLOBAL,
            "joint_training_placement_samples": joint_training_samples * PLACEMENTS_PER_GLOBAL,
            "placement_only_samples": placement_only_samples * PLACEMENTS_PER_GLOBAL,
            "joint_training_end_episode": joint_training_end,
            "placement_training_samples": placement_training_samples * PLACEMENTS_PER_GLOBAL,
            "planned_placement_ppo_updates": (
                placement_training_samples // args.placement_batch_episodes
            ),
            "planned_joint_placement_ppo_updates": (
                joint_training_samples // args.placement_batch_episodes
            ),
            "planned_placement_only_ppo_updates": (
                placement_only_samples // args.placement_batch_episodes
            ),
            "inner_stop_rule": (
                "reward_patience_or_max_updates_per_global_no_rollback"
            ),
            "allocation_kl_stop_rule": "per_actor_per_update_no_rollback",
            "num_uavs": environment.num_uavs,
            "num_ues": environment.num_ues,
            "seed_schedule": (
                "two maps per global, five cyclic placement orders per map; "
                "allocation collects one complete trajectory per pair per update"
            ),
            "allocation_rollout": (
                f"{args.allocation_batch_episodes} complete allocation "
                "trajectories per PPO update"
            ),
            "placement_score": (
                "initial deterministic zero-shot return from the allocator "
                "available at each global entry; warm-up globals are discarded "
                "for placement and used only to warm up allocation; allocation "
                "is held immutable during the final placement-only phase"
            ),
            "placement_selection_order": (
                "all five right-cyclic orders on each of the two maps per global; "
                "sampled placements need not be geometrically distinct"
            ),
            "checkpoint_resume_scope": (
                "exact only at completed outer frozen-batch boundaries; "
                "latest_inner_model.pt contains the last inner state and is evaluation-only"
            ),
            "framework": (
                "five placement PPO actors + five local bidder PPO actors + "
                "centralized placement V and allocation continuation V critics"
            ),
            "allocation_actor_observation_scope": coordination_scope,
            "allocation_actor_feature_version": learner.allocation_actor_feature_version,
            "allocation_actor_local_feature_dim": learner.allocation_actors[0].local_vector_encoder[0].in_features,
            "allocation_entropy_regularization": learner.allocation_entropy_regularization,
            "allocation_broadcast_mode": args.coordination_mode,
            "allocation_broadcast_assumptions": (
                "shared PRB balance is published before each UE decision; "
                "no other UAV resource, link-rank, future-load or reserve "
                "broadcast; no signaling cost or delay modeled"
                if environment.shared_prb_pool else
                "computed centrally before every UE allocation decision from "
                "the simulator state; delivery is instantaneous and this model "
                "assigns no signaling bandwidth, delay, loss, or energy cost"
            ),
            "coordination_ablation_protocol": (
                "none changes resources, action domain, masks and reward from "
                "legacy full/basic; these are not isolated broadcast ablations. "
                "Compare baselines using the same mode, constraints and reward."
            ),
            "allocation_action_order": (
                "learn participate -> PRB index (0..9 => 1..10 PRBs) -> conditional power "
                "index (0..9 => 0.1..1 W); reject ignores resource indices"
                if environment.shared_prb_pool else
                "learn participate -> PRB; append the sole Pareto-valid "
                "power index deterministically, per UAV"
            ),
            "allocation_action_semantics": environment.allocation_action_semantics,
            "auction": (
                "after simultaneous proposals: fixed physical QoS/resource "
                "feasibility filter, then minimum normalized resource bid "
                "rounded to 12 decimal places; ties use higher rate, then lower UAV ID"
            ),
            "allocation_credit": (
                "expected clipped PPO over the entire policy action catalog "
                "(including invalid proposals in none); COMA-style "
                "baseline; Q_cf = exact shared reward/scale + gamma*V(next_state_cf)"
            ),
            "allocation_mask_scope": (
                "coverage only: outside coverage reject only, inside coverage all "
                "100 PRB/power pairs plus reject; no resource, QoS or Pareto pruning"
                if environment.shared_prb_pool else
                "coverage, remaining resources, QoS-valid joint PRB/power "
                "pairs, and resource Pareto pruning; no critic or auction rank"
            ),
            "allocation_critic_observation_scope": (
                "global simulator state plus critic-only per UAV-UE minimum "
                "QoS-valid power by PRB (power/max_action_power, then feasibility "
                "flags); processed UE cost features are zeroed; budgets remain "
                "separate state features; cost table is not exposed to actors or masks"
                if environment.shared_prb_pool else "global simulator state"
            ),
            "allocation_critic_feature_version": learner.allocation_critic_feature_version,
            "allocation_critic_cost_feature_dim": learner.allocation_critic.allocation_cost_dim,
            "allocation_value_metric_semantics": {
                "allocation_value": "V(s) on actual rollout states after critic fitting",
                "allocation_q": "r/reward_scale + V(next_state) for actual actions; terminal continuation is zero",
                "value_q_target_units": "environment reward divided by reward_scale",
                "reward_scale": learner.reward_scale,
                "allocation_return_mean_units": "unscaled environment reward",
            },
            "placement_actor_observation_scope": (
                "global UE/candidate state, own UAV identity and previously "
                "selected placements; allocation-only "
                "realized links do not exist before placement"
            ),
            "reward_objectives": {
                "allocation": (
                    f"{environment.served_reward:g} - "
                    f"{environment.power_weight:g}*winner_power_w/"
                    f"{environment.power_penalty_denominator_w:g} - "
                    f"{environment.prb_weight:g}*winner_prbs/"
                    f"{environment.total_max_prbs}; 0 when no winner"
                ),
                "served_reward": environment.served_reward,
                "power_weight": environment.power_weight,
                "prb_weight": environment.prb_weight,
                "power_penalty_denominator_w": environment.power_penalty_denominator_w,
                "invalid_proposal_penalty": 0.0,
                "placement": (
                    "episode sum of the same served-minus-resource reward"
                ),
                "sum_reward_metric": (
                    "post-adaptation placement_return while allocation trains, "
                    "then frozen-allocator return during placement-only; "
                    "initial_zero_shot_return is the placement PPO target "
                    "outside allocator warm-up"
                ),
            },
            "budget_metric_semantics": {
                "training_budget_used": (
                    "legacy episode-equivalent alias: placement sample index + "
                    "real allocation training episodes; excludes deterministic "
                    "selection/evaluation and counterfactual previews"
                ),
                "global_episode": "one group of two maps, five placements per map",
                "placement_sample_id": "unique sample index across all globals",
                "placement_samples_collected": "ten samples per completed global",
                "allocation_convergence_change": "legacy alias for the last update KL; not an inner stopping condition",
                "real_allocation_training_transitions": (
                    "real allocation env.step calls used in PPO rollouts"
                ),
                "deterministic_selection_evaluation_transitions": (
                    "placement plus allocation env.step calls used to score and "
                    "diagnose allocation; never restore a best inner snapshot"
                ),
                "counterfactual_preview_transitions": (
                    "preview_allocation_step calls used for credit assignment"
                ),
                "optimizer_steps": (
                    "exact lifetime optimizer.step call counters maintained by "
                    "the learner; inner training state is never rolled back"
                ),
                "frozen_batch_wall_time_seconds": (
                    "one global allocation cycle plus its last-inner artifact; "
                    "nonzero only on frozen_batch_summary_row"
                ),
                "wall_time_seconds": "cumulative active process time; excludes downtime",
                "summary_row_flags": (
                    "use frozen_batch_summary_row for batch work, "
                    "placement_update_summary_row for actual placement updates, "
                    "and allocation_update_summary_row for allocation updates"
                ),
                "selection_fraction": (
                    "legacy alias for participation_fraction in learner metrics; "
                    "win_fraction is logged separately"
                ),
            },
            "radio_parameters": {
                "carrier_frequency_hz": float(radio_env.phys.f_c),
                "bandwidth_per_prb_hz": float(
                    radio_env.phys.Bandwidth_BRP
                ),
                "total_system_prbs": int(radio_env.total_system_prbs),
                "prb_budget_model": (
                    "one shared pool; deduct only winner's PRBs; no UAV quota"
                    if environment.shared_prb_pool else "fixed per-UAV PRB quotas"
                ),
                "max_prbs_per_action": int(environment.max_prbs_per_action),
                "prb_levels": environment.prb_levels.tolist(),
                "power_levels_w": environment.power_levels_w.tolist(),
                "max_power_w_per_action": float(environment.power_levels_w[-1]),
                "max_power_w": float(radio_env.phys.P_max_w),
                "max_power_dbm_for_reporting": float(
                    radio_env.phys.P_max_dbm
                ),
                "max_power_w_per_uav": [
                    float(value) for value in environment.max_power_w_per_uav
                ],
                "power_step_w": float(radio_env.power_step_w),
                "max_prbs_per_uav": None if environment.shared_prb_pool else [
                    int(value) for value in environment.max_prbs_per_uav
                ],
                "prb_step": 1,
                "grid_size": int(radio_env.grid_size),
                "ue_requirements_mbps": [
                    float(value) for value in radio_env.required_mbs
                ],
            },
        }
    )
    configuration.update(configuration_overrides or {})
    training_config_path = output_directory / "training_config.json"
    csv_path = output_directory / "training_metrics.csv"
    allocation_csv_path = output_directory / "allocation_metrics.csv"
    validation_csv_path = output_directory / "validation_metrics.csv"
    diagnostic_csv_path = output_directory / "diagnostic_metrics.csv"
    test_csv_path = output_directory / "test_metrics.csv"
    uav_placement_csv_path = output_directory / "uav_placement_metrics.csv"
    best_validation_score: tuple[float] | None = None
    best_validation_summary: dict[str, Any] | None = None
    best_validation_scores_by_phase: dict[str, tuple[float]] = {}
    best_validation_summaries_by_phase: dict[str, dict[str, Any]] = {}
    frozen_allocation_sha256: str | None = None
    frozen_allocation_optimizer_steps: dict[str, int] | None = None
    frozen_allocation_entropy_coef: float | None = None
    start_global_episode = 1
    allocation_episodes_used = 0
    allocation_updates_used = 0
    transition_counts = {
        "real_allocation_training": 0,
        "deterministic_selection_evaluation": 0,
        "counterfactual_preview": 0,
    }

    if args.resume is None:
        output_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text("", encoding="utf-8")
        training_config_path.write_text(
            json.dumps(configuration, indent=2), encoding="utf-8"
        )
    else:
        resume_path = args.resume.expanduser().resolve()
        if not output_directory.is_dir() or not checkpoint_directory.is_dir():
            raise FileNotFoundError(
                "Boundary resume requires the original output and checkpoint "
                "directories"
            )
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if not training_config_path.is_file():
            raise FileNotFoundError(
                "Boundary resume requires the original training_config.json"
            )
        existing_configuration = json.loads(
            training_config_path.read_text(encoding="utf-8")
        )
        if existing_configuration.get("configuration_fingerprint") != (
            configuration_fingerprint
        ):
            raise ValueError("Resume configuration does not match the original run")
        completed_episode = learner.load_checkpoint(
            resume_path,
            map_location=device,
            restore_training_state=True,
        )
        trainer_state = getattr(learner, "loaded_trainer_state", None)
        if not isinstance(trainer_state, dict):
            raise ValueError("Checkpoint has no exact trainer resume state")
        if trainer_state.get("resume_boundary") != "outer_frozen_batch":
            raise ValueError("Only completed outer frozen-batch resume is supported")
        start_global_episode = int(trainer_state["next_global_episode"])
        if start_global_episode != completed_episode + 1:
            raise ValueError("Checkpoint episode and next outer episode disagree")
        if (start_global_episode - 1) % args.placement_batch_episodes:
            raise ValueError("Resume point is inside a frozen placement batch")
        if Path(trainer_state["output_directory"]).resolve() != output_directory:
            raise ValueError("Resume must append to the original output directory")
        if trainer_state.get("configuration_fingerprint") != (
            configuration_fingerprint
        ):
            raise ValueError("Checkpoint was created with a different configuration")
        if list(trainer_state["episode_seed_schedule"]) != list(
            episode_seed_schedule
        ):
            raise ValueError("Resume seed schedule does not match this run")
        saved_provenance = trainer_state.get("runtime_provenance", {})
        for key in (
            "python",
            "numpy",
            "torch",
            "torch_deterministic_algorithms",
            "torch_cudnn_benchmark",
            "torch_cudnn_deterministic",
            "cublas_workspace_config",
            "torch_cuda_runtime",
            "torch_cudnn_runtime",
            "gymnasium",
            "matplotlib",
            "device",
            "platform",
            "cuda_device_name",
            "cuda_device_capability",
            "source_sha256",
        ):
            if saved_provenance.get(key) != provenance.get(key):
                raise ValueError(f"Resume runtime/source mismatch for {key}")
        rewards_by_seed.update(
            {
                int(seed): [float(value) for value in values]
                for seed, values in trainer_state["rewards_by_seed"].items()
            }
        )
        training_rewards[:] = [
            float(value) for value in trainer_state["training_rewards"]
        ]
        validation_rewards_by_seed.update(
            {
                int(seed): [
                    (int(episode), float(value)) for episode, value in values
                ]
                for seed, values in trainer_state[
                    "validation_rewards_by_seed"
                ].items()
            }
        )
        best_validation_score = (
            tuple(trainer_state["best_validation_score"])
            if trainer_state.get("best_validation_score") is not None
            else None
        )
        best_validation_summary = trainer_state.get("best_validation_summary")
        best_validation_scores_by_phase = {
            str(phase): tuple(score)
            for phase, score in trainer_state.get(
                "best_validation_scores_by_phase", {}
            ).items()
        }
        best_validation_summaries_by_phase = copy.deepcopy(
            trainer_state.get("best_validation_summaries_by_phase", {})
        )
        frozen_allocation_sha256 = trainer_state.get(
            "frozen_allocation_sha256"
        )
        saved_frozen_steps = trainer_state.get(
            "frozen_allocation_optimizer_steps"
        )
        frozen_allocation_optimizer_steps = (
            {str(key): int(value) for key, value in saved_frozen_steps.items()}
            if isinstance(saved_frozen_steps, dict)
            else None
        )
        saved_frozen_entropy = trainer_state.get(
            "frozen_allocation_entropy_coef"
        )
        frozen_allocation_entropy_coef = (
            float(saved_frozen_entropy)
            if saved_frozen_entropy is not None
            else None
        )
        if frozen_allocation_entropy_coef is not None and (
            not np.isfinite(frozen_allocation_entropy_coef)
            or frozen_allocation_entropy_coef < 0.0
        ):
            raise ValueError(
                "Checkpoint frozen allocation entropy is invalid"
            )
        allocation_episodes_used = int(
            trainer_state["allocation_episodes_used"]
        )
        allocation_updates_used = int(trainer_state["allocation_updates_used"])
        transition_counts.update(
            {
                key: int(value)
                for key, value in trainer_state["transition_counts"].items()
            }
        )
        diagnostic_counts.update(
            {
                key: int(value)
                for key, value in trainer_state["diagnostic_counts"].items()
            }
        )
        resumed_elapsed_wall_time = float(
            trainer_state["elapsed_wall_time_seconds"]
        )
        if start_global_episode < 1 or start_global_episode > total_episodes + 1:
            raise ValueError("Checkpoint next outer episode is outside this run")
        if len(training_rewards) != completed_episode * PLACEMENTS_PER_GLOBAL:
            raise ValueError(
                "Checkpoint reward history does not match its completed episode"
            )
        if trainer_state.get("optimizer_step_counts") != optimizer_step_counts(
            learner
        ):
            raise ValueError(
                "Checkpoint trainer and learner optimizer counters disagree"
            )
        if start_global_episode > joint_training_end:
            if (
                frozen_allocation_sha256 is None
                or frozen_allocation_optimizer_steps is None
                or frozen_allocation_entropy_coef is None
            ):
                raise ValueError(
                    "Placement-only resume is missing the frozen allocation "
                    "reference"
                )
            learner.allocation_entropy_coef = frozen_allocation_entropy_coef
            if allocation_model_sha256(learner) != frozen_allocation_sha256:
                raise ValueError(
                    "Resumed allocation model differs from the frozen reference"
                )
            current_steps = optimizer_step_counts(learner)
            for key in ("allocation_actor", "allocation_critic"):
                if current_steps[key] != frozen_allocation_optimizer_steps[key]:
                    raise ValueError(
                        "Resumed allocation optimizer count differs from the "
                        "frozen reference"
                    )
        retained_log_rows = {
            "training": _read_csv_at_resume_boundary(
                csv_path, CSV_FIELDS, completed_episode,
                committed_row_count=completed_episode * PLACEMENTS_PER_GLOBAL,
            ),
            "allocation": _read_csv_at_resume_boundary(
                allocation_csv_path,
                ALLOCATION_CSV_FIELDS,
                completed_episode,
                committed_row_count=allocation_episodes_used,
            ),
            "validation": _read_csv_at_resume_boundary(
                validation_csv_path,
                EVALUATION_CSV_FIELDS,
                completed_episode,
                committed_row_count=sum(map(len, validation_rewards_by_seed.values())),
            ),
            "uav_placement": _read_csv_at_resume_boundary(
                uav_placement_csv_path,
                UAV_PLACEMENT_CSV_FIELDS,
                completed_episode,
                committed_row_count=completed_episode * PLACEMENTS_PER_GLOBAL * int(environment.num_uavs),
            ),
        }
        retained_diagnostics = _read_diagnostics_at_resume_boundary(
            diagnostics_path, diagnostic_counts
        )
        _validate_resume_log_rows(
            retained_log_rows,
            completed_episode=completed_episode,
            episode_seed_schedule=episode_seed_schedule,
            training_rewards=training_rewards,
            rewards_by_seed=dict(rewards_by_seed),
            validation_rewards_by_seed=dict(validation_rewards_by_seed),
            validation_seed_order=validation_seeds,
            allocation_episodes_used=allocation_episodes_used,
            num_uavs=int(environment.num_uavs),
            placement_batch_episodes=args.placement_batch_episodes,
            placements_per_global=PLACEMENTS_PER_GLOBAL,
            warmup_samples=warmup_placement_samples,
            joint_training_end=joint_training_end,
            total_episodes=total_episodes,
        )
        _commit_resume_logs(
            (
                (csv_path, CSV_FIELDS, retained_log_rows["training"]),
                (
                    allocation_csv_path,
                    ALLOCATION_CSV_FIELDS,
                    retained_log_rows["allocation"],
                ),
                (
                    validation_csv_path,
                    EVALUATION_CSV_FIELDS,
                    retained_log_rows["validation"],
                ),
                (
                    uav_placement_csv_path,
                    UAV_PLACEMENT_CSV_FIELDS,
                    retained_log_rows["uav_placement"],
                ),
            ),
            diagnostics_path,
            retained_diagnostics,
        )
        archived = _archive_stale_resume_artifacts(
            output_directory,
            checkpoint_directory,
            completed_episode=completed_episode,
            total_episodes=total_episodes,
            resume_path=resume_path,
            joint_training_end=joint_training_end,
        )
        if best_validation_summary is not None:
            (output_directory / "best_validation_summary.json").write_text(
                json.dumps(best_validation_summary, indent=2),
                encoding="utf-8",
            )
        for phase, summary in best_validation_summaries_by_phase.items():
            (
                output_directory / f"best_{phase}_validation_summary.json"
            ).write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
        (output_directory / "latest_resume_summary.json").write_text(
            json.dumps(
                {
                    "checkpoint_path": str(resume_path),
                    **trainer_state,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        save_plots(training_rewards, args, output_directory)
        if any(validation_rewards_by_seed.values()):
            plot_validation_returns(
                validation_rewards_by_seed,
                validation_seeds,
                args.evaluation_every,
                output_directory / "validation_episode_sum_reward_by_seed.png",
            )
        if archived:
            print(
                f"Archived {len(archived)} stale post-boundary artifacts.",
                flush=True,
            )
    log_mode = "a" if args.resume is not None else "w"
    with (
        csv_path.open(log_mode, newline="", encoding="utf-8") as csv_file,
        allocation_csv_path.open(
            log_mode, newline="", encoding="utf-8"
        ) as allocation_csv_file,
        validation_csv_path.open(
            log_mode, newline="", encoding="utf-8"
        ) as validation_csv_file,
        uav_placement_csv_path.open(
            log_mode, newline="", encoding="utf-8"
        ) as uav_placement_csv_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        allocation_writer = csv.DictWriter(
            allocation_csv_file,
            fieldnames=ALLOCATION_CSV_FIELDS,
        )
        validation_writer = csv.DictWriter(
            validation_csv_file,
            fieldnames=EVALUATION_CSV_FIELDS,
        )
        uav_placement_writer = csv.DictWriter(
            uav_placement_csv_file,
            fieldnames=UAV_PLACEMENT_CSV_FIELDS,
        )
        if args.resume is None:
            writer.writeheader()
            allocation_writer.writeheader()
            validation_writer.writeheader()
            uav_placement_writer.writeheader()

        active_batch_metrics: dict[str, Any] | None = None
        active_batch_rows: list[dict[str, float | int]] = []
        active_batch_phase = ""
        batch_wall_time_seconds = 0.0
        for global_episode in range(start_global_episode, total_episodes + 1):
            batch_started_at = time.perf_counter()
            frozen_batch_id = global_episode
            batch_end = global_episode
            batch_seeds = episode_seed_schedule[
                (global_episode - 1) * PLACEMENTS_PER_GLOBAL : global_episode * PLACEMENTS_PER_GLOBAL
            ]
            active_batch_phase = training_phase_for_episode(
                global_episode,
                warmup_samples=warmup_placement_samples,
                joint_training_end=joint_training_end,
                total_episodes=total_episodes,
            )
            if training_phase_for_episode(
                batch_end,
                warmup_samples=warmup_placement_samples,
                joint_training_end=joint_training_end,
                total_episodes=total_episodes,
            ) != active_batch_phase:
                raise RuntimeError("A training phase split a frozen batch")
            batch_orders = tuple(
                cyclic_placement_order(index, int(environment.num_uavs))
                for index in range(len(batch_seeds))
            )
            allocation_training_active = (
                active_batch_phase
                != "placement_only_allocation_frozen"
            )
            print(
                f"global={global_episode:04d}/{total_episodes} start "
                f"phase={active_batch_phase} maps={list(dict.fromkeys(batch_seeds))} "
                f"placements={len(batch_seeds)} "
                f"allocation_updates_max={args.allocation_max_inner_updates if allocation_training_active else 0} "
                f"patience={args.allocation_inner_patience if allocation_training_active else 0}",
                flush=True,
            )
            if not allocation_training_active:
                if (
                    frozen_allocation_sha256 is None
                    or frozen_allocation_optimizer_steps is None
                    or frozen_allocation_entropy_coef is None
                ):
                    raise RuntimeError(
                        "Placement-only phase has no frozen allocation "
                        "reference"
                    )
                if (
                    float(learner.allocation_entropy_coef)
                    != frozen_allocation_entropy_coef
                ):
                    raise RuntimeError(
                        "Allocation entropy changed during the "
                        "placement-only phase"
                    )
                if (
                    allocation_model_sha256(learner)
                    != frozen_allocation_sha256
                ):
                    raise RuntimeError(
                        "Allocation model changed before a placement-only "
                        "batch"
                    )
                current_steps = optimizer_step_counts(learner)
                for key in ("allocation_actor", "allocation_critic"):
                    if (
                        current_steps[key]
                        != frozen_allocation_optimizer_steps[key]
                    ):
                        raise RuntimeError(
                            "Allocation optimizer advanced during the "
                            "placement-only phase"
                        )
            allocation_episodes_before_batch = allocation_episodes_used
            allocation_updates_before_batch = allocation_updates_used
            budget_index_before_inner = allocation_episodes_used

            def entropy_schedule(inner_episode: int) -> float:
                return allocation_entropy_coefficient(
                    budget_index_before_inner + inner_episode,
                    initial=args.allocation_entropy_coef,
                    final=args.allocation_entropy_final_coef,
                    anneal_start_episode=(
                        args.allocation_entropy_anneal_start_episode
                    ),
                    anneal_end_episode=(
                        args.allocation_entropy_anneal_end_episode
                    ),
                )

            active_batch_metrics = train_frozen_placement_batch(
                environment,
                learner,
                ue_seeds=batch_seeds,
                placement_orders=batch_orders,
                max_inner_updates=args.allocation_max_inner_updates,
                inner_patience=args.allocation_inner_patience,
                allocation_entropy_schedule=(
                    entropy_schedule if allocation_training_active else None
                ),
                deterministic_actions=False,
                update_placement=(active_batch_phase != "allocation_warmup"),
                update_allocation=allocation_training_active,
            )
            active_batch_metrics["training_phase"] = active_batch_phase
            if not allocation_training_active and (
                allocation_model_sha256(learner)
                != frozen_allocation_sha256
            ):
                raise RuntimeError(
                    "Allocation model changed inside a placement-only batch"
                )
            if batch_end == joint_training_end:
                frozen_allocation_sha256 = allocation_model_sha256(learner)
                current_steps = optimizer_step_counts(learner)
                frozen_allocation_optimizer_steps = {
                    key: current_steps[key]
                    for key in ("allocation_actor", "allocation_critic")
                }
                frozen_allocation_entropy_coef = float(
                    learner.allocation_entropy_coef
                )
            allocation_episodes_used += int(
                active_batch_metrics["allocation_inner_episodes"]
            )
            batch_inner_updates = int(
                active_batch_metrics["allocation_inner_updates"]
            )
            allocation_updates_used += batch_inner_updates
            transition_counts["real_allocation_training"] += int(
                active_batch_metrics["allocation_inner_episodes"]
            ) * int(environment.num_ues)
            transition_counts["deterministic_selection_evaluation"] += (
                int(active_batch_metrics["deterministic_evaluation_passes"])
                * len(batch_seeds)
                * (int(environment.num_ues) + 1)
            )
            transition_counts["counterfactual_preview"] += sum(
                int(update.get("allocation_unique_outcomes", 0))
                for update in active_batch_metrics["allocation_updates"]
            )
            episode_numbers = {
                int(seed): len(rewards_by_seed[int(seed)]) + 1
                for seed in batch_seeds
            }
            active_batch_rows = build_allocation_batch_log_rows(
                active_batch_metrics,
                first_global_episode=global_episode,
                allocation_episodes_before_batch=(
                    allocation_episodes_before_batch
                ),
                allocation_updates_before_batch=(
                    allocation_updates_before_batch
                ),
                frozen_batch_id=frozen_batch_id,
                num_ues=int(environment.num_ues),
                episode_in_seed=episode_numbers,
                placements_per_global=PLACEMENTS_PER_GLOBAL,
            )
            if allocation_training_active:
                learner.save_checkpoint(
                    checkpoint_directory / "latest_inner_model.pt",
                    episode=batch_end,
                )
                (
                    output_directory / "latest_inner_summary.json"
                ).write_text(
                    json.dumps(
                        {
                            "batch_first_global_episode": global_episode,
                            "batch_last_global_episode": batch_end,
                            "training_phase": active_batch_phase,
                            "ue_seeds": list(batch_seeds),
                            "placement_target_source": active_batch_metrics[
                                "placement_target_source"
                            ],
                            "initial_zero_shot_score": [
                                active_batch_metrics[
                                    "allocation_evaluation_history"
                                ][0]["mean_team_return"]
                            ],
                            "selection_objective": "mean_team_return (patience reference only; no model selection)",
                            "inner_patience": args.allocation_inner_patience,
                            "checks_without_improvement": active_batch_metrics["allocation_evaluation_history"][-1]["checks_without_improvement"],
                            "training_state": "last_inner_update_no_rollback",
                            "stop_reason": active_batch_metrics["allocation_stop_reason"],
                            "final_score": [active_batch_metrics["allocation_evaluation_history"][-1]["mean_team_return"]],
                            "best_score": list(
                                active_batch_metrics["best_score"]
                            ),
                            "best_inner_update": int(
                                active_batch_metrics["best_inner_update"]
                            ),
                            "inner_updates": int(
                                active_batch_metrics[
                                    "allocation_inner_updates"
                                ]
                            ),
                            "converged": bool(
                                active_batch_metrics[
                                    "allocation_converged"
                                ]
                            ),
                            "stopped_by_safety_cap": bool(
                                active_batch_metrics[
                                    "allocation_stopped_by_safety_cap"
                                ]
                            ),
                            "evaluation_history": active_batch_metrics[
                                "allocation_evaluation_history"
                            ],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            batch_wall_time_seconds = (
                time.perf_counter() - batch_started_at
            )

            for problem_index, requested_ue_seed in enumerate(batch_seeds):
                if active_batch_metrics is None:
                    raise RuntimeError("Frozen placement batch was not initialized")
                if int(active_batch_metrics["ue_seeds"][problem_index]) != int(
                    requested_ue_seed
                ):
                    raise RuntimeError("Frozen placement batch seed order changed")
                final_evaluation = active_batch_metrics["final_evaluations"][
                    problem_index
                ]
                initial_evaluation = active_batch_metrics["initial_evaluations"][
                    problem_index
                ]
                metrics = {
                    **final_evaluation,
                    "placement_metrics": active_batch_metrics[
                        "placement_metrics"
                    ],
                    "allocation_updates": active_batch_metrics[
                        "allocation_updates"
                    ],
                    "allocation_inner_episodes": active_batch_metrics[
                        "allocation_inner_episodes"
                    ],
                    "allocation_converged": active_batch_metrics[
                        "allocation_converged"
                    ],
                    "allocation_convergence_change": active_batch_metrics[
                        "allocation_convergence_change"
                    ],
                    "allocation_training_active": active_batch_metrics[
                        "allocation_training_active"
                    ],
                    "allocation_state_unchanged": active_batch_metrics[
                        "allocation_state_unchanged"
                    ],
                    "placement_score_infos": [
                        final_evaluation["final_info"]
                    ],
                }
                inner_episodes = int(metrics["allocation_inner_episodes"])
                inner_updates = int(active_batch_metrics["allocation_inner_updates"])
                placement_sample_id = (global_episode - 1) * PLACEMENTS_PER_GLOBAL + problem_index + 1
                training_budget_used = placement_sample_id + allocation_episodes_used
                entropy_history = active_batch_metrics[
                    "allocation_entropy_coefficients_by_problem"
                ][problem_index]
                current_allocation_entropy_coef = (
                    float(entropy_history[-1])
                    if entropy_history
                    else float(learner.allocation_entropy_coef)
                )
                placement_order = tuple(
                    int(index)
                    for index in active_batch_metrics["placement_orders"][
                        problem_index
                    ]
                )
                final_info = metrics["final_info"]
                ue_seed = int(final_info["episode_ue_seed"])
                if ue_seed != requested_ue_seed:
                    raise RuntimeError(
                        f"Requested UE seed {requested_ue_seed}, got {ue_seed}"
                    )
                allocation_return = float(metrics["allocation_return"])
                placement_return = float(metrics["placement_return"])
                rewards_by_seed[ue_seed].append(placement_return)
                training_rewards.append(placement_return)
                episode_in_seed = len(rewards_by_seed[ue_seed])
                if episode_in_seed > args.episodes_per_seed * PLACEMENTS_PER_MAP:
                    raise RuntimeError(
                        f"Seed {ue_seed} exceeded its placement sample allowance"
                    )

                placement_metrics = metrics["placement_metrics"]
                allocation_updates = metrics["allocation_updates"]
                if any(
                    update["allocation_rollout_size"] % environment.num_ues
                    for update in allocation_updates
                ):
                    raise RuntimeError(
                        "Allocation PPO must receive only complete episode rollouts"
                    )
                placement_indices = [
                    int(value) for value in final_info["placement_indices"]
                ]
                placement_duplicates = len(placement_indices) - len(
                    set(placement_indices)
                )
                if placement_duplicates:
                    raise RuntimeError("Placement actors selected duplicate candidates")
                cumulative_optimizer_steps = optimizer_step_counts(learner)
                frozen_batch_summary_row = int(problem_index == 0)
                placement_update_summary_row = int(
                    frozen_batch_summary_row
                    and bool(placement_metrics.get("placement_updated", 0.0))
                )
                elapsed_wall_time_seconds = (
                    resumed_elapsed_wall_time
                    + time.perf_counter()
                    - process_started_at
                )
                allocation_rows = active_batch_rows if problem_index == 0 else []
                allocation_writer.writerows(allocation_rows)
                allocation_csv_file.flush()
                for allocation_row in allocation_rows:
                    allocation_episode = int(allocation_row["allocation_episode"])
                    if (
                        allocation_episode == 1
                        or allocation_episode % args.progress_every == 0
                    ):
                        print(
                            "allocation="
                            f"{allocation_episode:05d} "
                            "placement="
                            f"{int(allocation_row['global_episode']):03d}/"
                            f"{total_episodes} "
                            f"inner={int(allocation_row['inner_episode']):02d}/"
                            f"{inner_updates:02d} "
                            f"seed={int(allocation_row['ue_seed'])} "
                            "return="
                            f"{float(allocation_row['allocation_return']):.4f} "
                            f"served={int(allocation_row['served_count'])} "
                            f"ratio={float(allocation_row['served_ratio']):.2%}",
                            flush=True,
                        )
                writer.writerow(
                    {
                        "global_episode": global_episode,
                        "placement_sample_id": placement_sample_id,
                        "placement_index_in_global": problem_index + 1,
                        "frozen_batch_id": frozen_batch_id,
                        "problem_index_in_batch": problem_index + 1,
                        "training_phase": active_batch_phase,
                        "placement_order": format_placement_order(
                            placement_order
                        ),
                        "allocation_training_active": int(
                            metrics["allocation_training_active"]
                        ),
                        "allocation_state_unchanged": int(
                            metrics["allocation_state_unchanged"]
                        ),
                        "frozen_batch_summary_row": frozen_batch_summary_row,
                        "placement_update_id": max(
                            0,
                            (global_episode - warmup_placement_samples) // args.placement_batch_episodes,
                        ),
                        "placement_update_summary_row": (
                            placement_update_summary_row
                        ),
                        "training_budget_used": training_budget_used,
                        "allocation_episodes_used": allocation_episodes_used,
                        "placement_samples_collected": global_episode * PLACEMENTS_PER_GLOBAL,
                        "real_allocation_training_transitions": (
                            transition_counts["real_allocation_training"]
                        ),
                        "deterministic_selection_evaluation_transitions": (
                            transition_counts[
                                "deterministic_selection_evaluation"
                            ]
                        ),
                        "counterfactual_preview_transitions": (
                            transition_counts["counterfactual_preview"]
                        ),
                        "cumulative_placement_actor_optimizer_steps": (
                            cumulative_optimizer_steps["placement_actor"]
                        ),
                        "cumulative_placement_critic_optimizer_steps": (
                            cumulative_optimizer_steps["placement_critic"]
                        ),
                        "cumulative_allocation_actor_optimizer_steps": (
                            cumulative_optimizer_steps["allocation_actor"]
                        ),
                        "cumulative_allocation_critic_optimizer_steps": (
                            cumulative_optimizer_steps["allocation_critic"]
                        ),
                        "cumulative_total_optimizer_steps": (
                            cumulative_optimizer_steps["total"]
                        ),
                        "frozen_batch_wall_time_seconds": (
                            batch_wall_time_seconds
                            if frozen_batch_summary_row
                            else 0.0
                        ),
                        "wall_time_seconds": elapsed_wall_time_seconds,
                        "ue_seed": ue_seed,
                        "episode_in_seed": episode_in_seed,
                        "sum_reward": placement_return,
                        "allocation_return": allocation_return,
                        "placement_return": placement_return,
                        "initial_zero_shot_return": float(
                            initial_evaluation["placement_return"]
                        ),
                        "served_count": int(metrics["served_count"]),
                        "served_ratio": float(metrics["served_ratio"]),
                        "total_used_prbs": int(final_info["total_used_prbs"]),
                        "total_used_power_w": float(
                            final_info["total_used_power_w"]
                        ),
                        "placement_duplicates": placement_duplicates,
                        "placement_updated": placement_metrics.get(
                            "placement_updated", 0.0
                        ),
                        "placement_target_accepted": placement_metrics.get(
                            "placement_target_accepted", 0.0
                        ),
                        "placement_episodes_collected": placement_metrics.get(
                            "placement_episodes_collected", 0.0
                        ),
                        "placement_actor_loss": placement_metrics.get(
                            "placement_actor_loss", float("nan")
                        ),
                        "placement_critic_loss": placement_metrics.get(
                            "placement_critic_loss", float("nan")
                        ),
                        "placement_entropy": placement_metrics.get(
                            "placement_entropy", float("nan")
                        ),
                        "placement_clip_fraction": placement_metrics.get(
                            "placement_clip_fraction", float("nan")
                        ),
                        "placement_advantage_mean": placement_metrics.get(
                            "placement_advantage_mean", float("nan")
                        ),
                        "placement_value_mean": placement_metrics.get(
                            "placement_value_mean", float("nan")
                        ),
                        "placement_popart_mean": placement_metrics.get(
                            "placement_popart_mean", float("nan")
                        ),
                        "placement_popart_std": placement_metrics.get(
                            "placement_popart_std", float("nan")
                        ),
                        "placement_gradient_steps": placement_metrics.get(
                            "placement_gradient_steps", float("nan")
                        ),
                        "placement_actor_optimizer_steps": placement_metrics.get(
                            "placement_actor_optimizer_steps", float("nan")
                        ),
                        "placement_policy_sample_uses": placement_metrics.get(
                            "placement_policy_sample_uses", float("nan")
                        ),
                        "allocation_actor_loss": mean_update_metric(
                            allocation_updates,
                            "allocation_actor_loss",
                        ),
                        "allocation_critic_loss": mean_update_metric(
                            allocation_updates,
                            "allocation_critic_loss",
                        ),
                        "allocation_entropy": mean_update_metric(
                            allocation_updates,
                            "allocation_entropy",
                        ),
                        "allocation_participation_entropy": mean_update_metric(
                            allocation_updates,
                            "allocation_participation_entropy",
                        ),
                        "allocation_prb_entropy": mean_update_metric(
                            allocation_updates,
                            "allocation_prb_entropy",
                        ),
                        "allocation_power_entropy": mean_update_metric(
                            allocation_updates,
                            "allocation_power_entropy",
                        ),
                        "allocation_reject_fraction": mean_update_metric(
                            allocation_updates,
                            "allocation_reject_fraction",
                        ),
                        "allocation_counterfactual_advantage_mean": mean_update_metric(
                            allocation_updates,
                            "allocation_counterfactual_advantage_mean",
                        ),
                        "allocation_counterfactual_nonzero_fraction": mean_update_metric(
                            allocation_updates,
                            "allocation_counterfactual_nonzero_fraction",
                        ),
                        "allocation_clip_fraction": mean_update_metric(
                            allocation_updates,
                            "allocation_clip_fraction",
                        ),
                        "allocation_policy_kl": mean_update_metric(
                            allocation_updates,
                            "allocation_policy_kl",
                        ),
                        "allocation_entropy_coef": (
                            current_allocation_entropy_coef
                        ),
                        "allocation_value_mean": mean_update_metric(
                            allocation_updates,
                            "allocation_value_mean",
                        ),
                        "allocation_value_max_abs": mean_update_metric(
                            allocation_updates,
                            "allocation_value_max_abs",
                        ),
                        "allocation_q_mean": mean_update_metric(
                            allocation_updates,
                            "allocation_q_mean",
                        ),
                        "allocation_q_max_abs": mean_update_metric(
                            allocation_updates,
                            "allocation_q_max_abs",
                        ),
                        "allocation_return_mean": mean_update_metric(
                            allocation_updates,
                            "allocation_return_mean",
                        ),
                        "allocation_rollout_size": mean_update_metric(
                            allocation_updates,
                            "allocation_rollout_size",
                        ),
                        "allocation_gradient_steps": mean_update_metric(
                            allocation_updates,
                            "allocation_gradient_steps",
                        ),
                        "allocation_critic_gradient_steps": mean_update_metric(
                            allocation_updates,
                            "allocation_critic_gradient_steps",
                        ),
                        "allocation_policy_sample_uses": mean_update_metric(
                            allocation_updates,
                            "allocation_policy_sample_uses",
                        ),
                        "allocation_inner_episodes": int(
                            metrics["allocation_inner_episodes"]
                        ),
                        "allocation_converged": int(
                            metrics["allocation_converged"]
                        ),
                        "allocation_stop_reason": active_batch_metrics["allocation_stop_reason"],
                        "allocation_inner_updates": inner_updates,
                        "allocation_convergence_change": float(
                            metrics["allocation_convergence_change"]
                        ),
                    }
                )
                csv_file.flush()

                environment.reset(options={"ue_seed": ue_seed})
                environment.step(
                    np.asarray(metrics["placement_actions"], dtype=np.int64)
                )
                for uav_metrics in summarize_uav_placement_metrics(
                    environment,
                    metrics["placement_score_infos"],
                    placement_return,
                ):
                    uav_id = int(uav_metrics["uav_id"])
                    uav_placement_writer.writerow(
                        {
                            "global_episode": global_episode,
                        "placement_sample_id": placement_sample_id,
                        "placement_index_in_global": problem_index + 1,
                            "ue_seed": ue_seed,
                            "episode_in_seed": episode_in_seed,
                            "selection_rank": placement_order.index(uav_id) + 1,
                            **uav_metrics,
                        }
                    )
                uav_placement_csv_file.flush()

                if problem_index == PLACEMENTS_PER_GLOBAL - 1 and (
                    global_episode == 1
                    or global_episode % args.progress_every == 0
                ):
                    print(
                        f"global={global_episode:04d}/{total_episodes} "
                        f"phase={active_batch_phase} "
                        f"order={format_placement_order(placement_order)} "
                        f"seed={ue_seed} "
                        f"sample_in_map={episode_in_seed:03d}/{args.episodes_per_seed * PLACEMENTS_PER_MAP} "
                        f"sample_allocation_return={allocation_return:.4f} "
                        f"sample_placement_return={placement_return:.4f} "
                        f"mean_served={np.mean([item['served_count'] for item in active_batch_metrics['final_evaluations']]):.2f} "
                        f"sample_ratio={metrics['served_ratio']:.2%} "
                        "inner_episodes="
                        f"{inner_episodes} "
                        f"stop={active_batch_metrics['allocation_stop_reason']} "
                        "allocation_training="
                        f"{bool(metrics['allocation_training_active'])} "
                        f"episodes_used={training_budget_used} "
                        "allocation_entropy_coef="
                        f"{current_allocation_entropy_coef:.6f} "
                        "placement_update="
                        f"{bool(placement_metrics.get('placement_updated', 0.0))} "
                        "placement_target="
                        f"{'accepted' if placement_metrics.get('placement_target_accepted', 0.0) else 'discarded'}",
                        flush=True,
                    )

            should_evaluate = (
                global_episode % args.placement_batch_episodes == 0
                and global_episode > warmup_placement_samples
                and args.evaluation_every > 0
                and (
                    global_episode % args.evaluation_every == 0
                    or global_episode == joint_training_end
                    or global_episode == total_episodes
                )
            )
            if should_evaluate:
                validation_rows = evaluate_seed_set(
                    validation_environment,
                    learner,
                    validation_seeds,
                    global_episode=global_episode,
                    evaluation_kind="validation",
                )
                transition_counts[
                    "deterministic_selection_evaluation"
                ] += sum(
                    int(row["evaluation_transitions"])
                    for row in validation_rows
                )
                for validation_row in validation_rows:
                    validation_writer.writerow(validation_row)
                for validation_seed in validation_seeds:
                    seed_rewards = [
                        float(row["sum_reward"])
                        for row in validation_rows
                        if int(row["ue_seed"]) == int(validation_seed)
                    ]
                    if len(seed_rewards) != 1:
                        raise RuntimeError(
                            "Validation must evaluate each UE layout exactly once"
                        )
                    validation_rewards_by_seed[int(validation_seed)].append(
                        (global_episode, seed_rewards[0])
                    )
                validation_csv_file.flush()
                validation_summary = summarize_evaluation(
                    validation_rows,
                    num_ues=environment.num_ues,
                )
                current_validation_score = validation_selection_score(
                    validation_summary
                )
                validation_phase = active_batch_phase
                previous_phase_score = best_validation_scores_by_phase.get(
                    validation_phase
                )
                is_best_phase_validation = (
                    previous_phase_score is None
                    or current_validation_score > previous_phase_score
                )
                is_best_validation = (
                    best_validation_score is None
                    or current_validation_score > best_validation_score
                )
                if is_best_phase_validation:
                    versioned_best_checkpoint_path = (
                        checkpoint_directory
                        / (
                            f"best_validation_global_episode_{global_episode}.pt"
                            if is_best_validation
                            else f"best_{validation_phase}_validation_"
                            f"global_episode_{global_episode}.pt"
                        )
                    )
                    phase_validation_summary = {
                        "checkpoint_episode": global_episode,
                        "checkpoint_path": str(
                            versioned_best_checkpoint_path.resolve()
                        ),
                        "training_phase": validation_phase,
                        "validation_seeds": list(validation_seeds),
                        "selection_objective": "mean_scalar_team_return",
                        "selection_score": list(current_validation_score),
                        **validation_summary,
                    }
                    learner.save_checkpoint(
                        versioned_best_checkpoint_path,
                        episode=global_episode,
                    )
                    best_validation_scores_by_phase[validation_phase] = (
                        current_validation_score
                    )
                    best_validation_summaries_by_phase[validation_phase] = (
                        phase_validation_summary
                    )
                    (
                        output_directory
                        / f"best_{validation_phase}_validation_summary.json"
                    ).write_text(
                        json.dumps(phase_validation_summary, indent=2),
                        encoding="utf-8",
                    )
                    if is_best_validation:
                        best_validation_score = current_validation_score
                        best_validation_summary = phase_validation_summary
                        (
                            output_directory / "best_validation_summary.json"
                        ).write_text(
                            json.dumps(best_validation_summary, indent=2),
                            encoding="utf-8",
                        )
                plot_validation_returns(
                    validation_rewards_by_seed,
                    validation_seeds,
                    args.evaluation_every,
                    output_directory
                    / "validation_episode_sum_reward_by_seed.png",
                )
                print(
                    f"validation global={global_episode:04d} "
                    "mean_served="
                    f"{validation_summary['mean_served_count']:.2f}/"
                    f"{environment.num_ues} "
                    f"perfect={validation_summary['perfect_count']}/"
                    f"{validation_summary['num_evaluations']} "
                    f"best_phase={is_best_phase_validation} "
                    f"best_global={is_best_validation}",
                    flush=True,
                )

            if global_episode % args.placement_batch_episodes == 0:
                csv_file.flush()
                allocation_csv_file.flush()
                validation_csv_file.flush()
                uav_placement_csv_file.flush()
                elapsed_wall_time_seconds = (
                    resumed_elapsed_wall_time
                    + time.perf_counter()
                    - process_started_at
                )
                trainer_state = {
                    "resume_boundary": "outer_frozen_batch",
                    "completed_global_episode": global_episode,
                    "next_global_episode": global_episode + 1,
                    "output_directory": str(output_directory),
                    "configuration_fingerprint": configuration_fingerprint,
                    "runtime_provenance": provenance,
                    "episode_seed_schedule": list(episode_seed_schedule),
                    "rewards_by_seed": {
                        int(seed): list(values)
                        for seed, values in rewards_by_seed.items()
                    },
                    "training_rewards": list(training_rewards),
                    "validation_rewards_by_seed": {
                        int(seed): list(values)
                        for seed, values in validation_rewards_by_seed.items()
                    },
                    "best_validation_score": (
                        list(best_validation_score)
                        if best_validation_score is not None
                        else None
                    ),
                    "best_validation_summary": best_validation_summary,
                    "best_validation_scores_by_phase": {
                        phase: list(score)
                        for phase, score in best_validation_scores_by_phase.items()
                    },
                    "best_validation_summaries_by_phase": (
                        best_validation_summaries_by_phase
                    ),
                    "frozen_allocation_sha256": frozen_allocation_sha256,
                    "frozen_allocation_optimizer_steps": (
                        frozen_allocation_optimizer_steps
                    ),
                    "frozen_allocation_entropy_coef": (
                        frozen_allocation_entropy_coef
                    ),
                    "allocation_episodes_used": allocation_episodes_used,
                    "allocation_updates_used": allocation_updates_used,
                    "transition_counts": dict(transition_counts),
                    "diagnostic_counts": dict(diagnostic_counts),
                    "optimizer_step_counts": optimizer_step_counts(learner),
                    "elapsed_wall_time_seconds": elapsed_wall_time_seconds,
                }
                latest_resume_path = checkpoint_directory / "latest_resume.pt"
                temporary_resume_path = checkpoint_directory / ".latest_resume.pt.tmp"
                learner.save_checkpoint(
                    temporary_resume_path,
                    episode=global_episode,
                    trainer_state=trainer_state,
                )
                temporary_resume_path.replace(latest_resume_path)
                (output_directory / "latest_resume_summary.json").write_text(
                    json.dumps(
                        {
                            "checkpoint_path": str(latest_resume_path.resolve()),
                            **trainer_state,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                if (
                    global_episode % args.checkpoint_every == 0
                    or global_episode == total_episodes
                ):
                    learner.save_checkpoint(
                        checkpoint_directory
                        / f"global_episode_{global_episode}.pt",
                        episode=global_episode,
                        trainer_state=trainer_state,
                    )
                    save_plots(training_rewards, args, output_directory)
                if global_episode == joint_training_end:
                    learner.save_checkpoint(
                        checkpoint_directory / "joint_training_end.pt",
                        episode=global_episode,
                        trainer_state=trainer_state,
                    )

    if learner.placement_rollout_size or learner.allocation_rollout_size:
        raise RuntimeError(
            "Training ended away from an exact outer frozen-batch boundary"
        )

    trainer_state = dict(trainer_state)
    trainer_state["elapsed_wall_time_seconds"] = (
        resumed_elapsed_wall_time + time.perf_counter() - process_started_at
    )
    trainer_state["optimizer_step_counts"] = optimizer_step_counts(learner)
    learner.save_checkpoint(
        checkpoint_directory / "final_model.pt",
        episode=total_episodes,
        trainer_state=trainer_state,
    )
    save_plots(training_rewards, args, output_directory)
    training_budget_summary = {
        "global_episodes_completed": total_episodes,
        "training_map_episodes": total_episodes * MAPS_PER_GLOBAL,
        "placement_samples_collected": total_episodes * PLACEMENTS_PER_GLOBAL,
        "allocation_warmup_placement_samples": warmup_placement_samples * PLACEMENTS_PER_GLOBAL,
        "joint_training_placement_samples": joint_training_samples * PLACEMENTS_PER_GLOBAL,
        "placement_only_samples": placement_only_samples * PLACEMENTS_PER_GLOBAL,
        "placement_training_samples": placement_training_samples * PLACEMENTS_PER_GLOBAL,
        "placement_ppo_update_batches": (
            placement_training_samples // args.placement_batch_episodes
        ),
        "joint_placement_ppo_update_batches": (
            joint_training_samples // args.placement_batch_episodes
        ),
        "placement_only_ppo_update_batches": (
            placement_only_samples // args.placement_batch_episodes
        ),
        "allocation_training_episodes": allocation_episodes_used,
        "allocation_updates": allocation_updates_used,
        "real_allocation_training_transitions": transition_counts[
            "real_allocation_training"
        ],
        "deterministic_selection_evaluation_transitions": transition_counts[
            "deterministic_selection_evaluation"
        ],
        "counterfactual_preview_transitions": transition_counts[
            "counterfactual_preview"
        ],
        "optimizer_step_counts": optimizer_step_counts(learner),
        "frozen_allocation_sha256": frozen_allocation_sha256,
        "frozen_allocation_optimizer_steps": (
            frozen_allocation_optimizer_steps
        ),
        "frozen_allocation_entropy_coef": frozen_allocation_entropy_coef,
        "best_validation_summaries_by_phase": (
            best_validation_summaries_by_phase
        ),
        "active_wall_time_seconds": trainer_state[
            "elapsed_wall_time_seconds"
        ],
        "wall_time_note": (
            "Cumulative active process time; excludes downtime between an "
            "interrupted run and boundary resume."
        ),
    }
    (output_directory / "training_budget_summary.json").write_text(
        json.dumps(training_budget_summary, indent=2),
        encoding="utf-8",
    )
    if best_validation_summary is None or best_validation_score is None:
        raise RuntimeError("Training finished without a validation checkpoint")
    best_checkpoint_path = Path(
        str(best_validation_summary["checkpoint_path"])
    ).resolve()
    if not best_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Selected validation checkpoint not found: {best_checkpoint_path}"
        )
    best_episode = learner.load_checkpoint(
        best_checkpoint_path,
        map_location=device,
    )
    if best_episode != int(best_validation_summary["checkpoint_episode"]):
        raise RuntimeError("Loaded the wrong best-validation checkpoint")
    diagnostic_rows = evaluate_seed_set(
        diagnostic_environment,
        learner,
        diagnostic_seeds,
        global_episode=best_episode,
        evaluation_kind="diagnostic_former_test_split",
    )
    with diagnostic_csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as diagnostic_csv_file:
        diagnostic_writer = csv.DictWriter(
            diagnostic_csv_file,
            fieldnames=EVALUATION_CSV_FIELDS,
        )
        diagnostic_writer.writeheader()
        diagnostic_writer.writerows(diagnostic_rows)
    diagnostic_summary = {
        "selected_checkpoint_episode": best_episode,
        "selected_checkpoint_training_phase": best_validation_summary[
            "training_phase"
        ],
        "used_for_model_selection": False,
        "evaluation_kind": "diagnostic_former_test_split",
        "diagnostic_seeds": list(diagnostic_seeds),
        "evaluation_transitions": sum(
            int(row["evaluation_transitions"]) for row in diagnostic_rows
        ),
        **summarize_evaluation(
            diagnostic_rows, num_ues=environment.num_ues
        ),
    }
    (output_directory / "diagnostic_summary.json").write_text(
        json.dumps(diagnostic_summary, indent=2),
        encoding="utf-8",
    )
    test_rows = evaluate_seed_set(
        test_environment,
        learner,
        test_seeds,
        global_episode=best_episode,
        evaluation_kind="test",
    )
    with test_csv_path.open("w", newline="", encoding="utf-8") as test_csv_file:
        test_writer = csv.DictWriter(
            test_csv_file,
            fieldnames=EVALUATION_CSV_FIELDS,
        )
        test_writer.writeheader()
        test_writer.writerows(test_rows)
    test_summary = {
        "selected_checkpoint_episode": best_episode,
        "selected_checkpoint_training_phase": best_validation_summary[
            "training_phase"
        ],
        "selected_by": "validation_mean_scalar_team_return_only",
        "test_seeds": list(test_seeds),
        "evaluation_transitions": sum(
            int(row["evaluation_transitions"]) for row in test_rows
        ),
        **summarize_evaluation(test_rows, num_ues=environment.num_ues),
    }
    (output_directory / "test_summary.json").write_text(
        json.dumps(test_summary, indent=2),
        encoding="utf-8",
    )
    if not args.no_greedy_baseline:
        baseline_rows = evaluate_greedy_allocation_baseline(
            test_environment,
            learner,
            test_seeds,
            global_episode=best_episode,
        )
        baseline_csv_path = output_directory / "greedy_baseline_metrics.csv"
        with baseline_csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as baseline_csv_file:
            baseline_writer = csv.DictWriter(
                baseline_csv_file,
                fieldnames=EVALUATION_CSV_FIELDS,
            )
            baseline_writer.writeheader()
            baseline_writer.writerows(baseline_rows)
        learned_by_seed = {
            int(row["ue_seed"]): row for row in test_rows
        }
        baseline_by_seed = {
            int(row["ue_seed"]): row for row in baseline_rows
        }
        if learned_by_seed.keys() != baseline_by_seed.keys():
            raise RuntimeError("Learned and baseline test layouts are misaligned")
        paired_return_differences = [
            float(learned_by_seed[seed]["sum_reward"])
            - float(baseline_by_seed[seed]["sum_reward"])
            for seed in learned_by_seed
        ]
        paired_served_differences = [
            int(learned_by_seed[seed]["served_count"])
            - int(baseline_by_seed[seed]["served_count"])
            for seed in learned_by_seed
        ]
        baseline_summary = {
            "baseline": (
                "learned deterministic placement plus minimum-cost feasible "
                "proposal from every UAV through the same physical checks and auction"
            ),
            "selected_checkpoint_episode": best_episode,
            "test_seeds": list(test_seeds),
            "evaluation_transitions": sum(
                int(row["evaluation_transitions"])
                for row in baseline_rows
            ),
            "learned_minus_greedy_mean_return": float(
                np.mean(paired_return_differences)
            ),
            "learned_minus_greedy_mean_served_count": float(
                np.mean(paired_served_differences)
            ),
            "paired_difference_note": (
                "Differences use the same held-out UE seed for both allocation "
                "policies; positive values favor the learned allocation policy."
            ),
            **summarize_evaluation(
                baseline_rows, num_ues=environment.num_ues
            ),
        }
        (output_directory / "greedy_baseline_summary.json").write_text(
            json.dumps(baseline_summary, indent=2),
            encoding="utf-8",
        )
    if not args.no_kmeans_baseline:
        kmeans_summary, kmeans_rows = run_kmeans_baseline(
            validation_environment, test_environment, validation_seeds,
            test_seeds, output_directory,
        )
        kmeans_by_seed = {int(row["ue_seed"]): row for row in kmeans_rows}
        comparisons = {"learned_placement_learned_allocation": test_rows}
        if not args.no_greedy_baseline:
            comparisons["learned_placement_greedy_allocation"] = baseline_rows
        kmeans_summary["compared_checkpoint_episode"] = best_episode
        kmeans_summary["paired_comparisons"] = {}
        for name, rows in comparisons.items():
            if {int(row["ue_seed"]) for row in rows} != kmeans_by_seed.keys():
                raise RuntimeError("K-means and learned test layouts are misaligned")
            kmeans_summary["paired_comparisons"][name] = {
                f"mean_{metric}_minus_kmeans": float(np.mean([
                    float(row[metric]) - float(kmeans_by_seed[int(row["ue_seed"])][metric])
                    for row in rows
                ]))
                for metric in ("sum_reward", "served_count")
            }
        (output_directory / "kmeans_baseline_summary.json").write_text(
            json.dumps(kmeans_summary, indent=2, allow_nan=False), encoding="utf-8"
        )
    if args.benchmark_inference_episodes:
        benchmark = benchmark_deterministic_inference(
            test_environment,
            learner,
            test_seeds,
            episodes=args.benchmark_inference_episodes,
            device=device,
        )
        (output_directory / "inference_benchmark.json").write_text(
            json.dumps(benchmark, indent=2),
            encoding="utf-8",
        )
    print(
        "test best_episode="
        f"{best_episode} mean_served={test_summary['mean_served_count']:.2f}/"
        f"{environment.num_ues} perfect={test_summary['perfect_count']}/"
        f"{test_summary['num_evaluations']}",
        flush=True,
    )
    print(f"Training complete. Outputs: {output_directory}")


if __name__ == "__main__":
    main()
