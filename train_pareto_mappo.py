"""Launch shared-PRB Pareto MAPPO with the existing Astra train/eval/resume schedule.

Uses env.py/ver2.py, networks and the bundled train_strict_local_mappo.py
in THIS directory. Copy this whole directory to run on another machine.
Default budget is 10 warm-up + 700 joint globals, with no placement-only phase. All common CLI switches
remain available. --resume reloads saved options unless explicitly overridden;
the shared runner then enforces exact configuration/source compatibility.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import local_mappo_baseline as baseline
import pareto_mappo
import env as physics
import ver2
import strict_local_independent_mappo as networks
import learning_credit
import kmeans_baseline
import train_strict_local_mappo as runner


def build_parser():
    parser = runner.build_argument_parser()
    parser.description = "Pareto MAPPO: compact residual actors, global no-cost critic, sampled PPO + GAE."
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--self-check", action="store_true", help="Check the learner only; no training run")
    parser.add_argument("--print-config", action="store_true", help="Show effective options without starting training")
    parser.set_defaults(
        output_dir=str(ROOT / "training_outputs_pareto_mappo_seed2026"),
        device="cpu", torch_num_threads=1,
        placement_episodes=710, placement_only_episodes=0,
        allocation_max_inner_updates=20, allocation_inner_patience=10,
        placement_ppo_epochs=5, allocation_ppo_epochs=5,
        coordination_mode="none", served_reward=1., power_weight=.2, prb_weight=1.,
        allocation_entropy_coef=.003, allocation_entropy_final_coef=.003,
        no_greedy_baseline=True, no_kmeans_baseline=True,
    )
    return parser


def parse_arguments(argv=None, *, parser=None, algorithm="local_pareto_shared_ppo_gae"):
    parser = build_parser() if parser is None else parser
    args = parser.parse_args(argv)
    if args.resume is not None:
        run_dir = args.resume.expanduser().resolve().parent.parent
        saved = json.loads((run_dir / "training_config.json").read_text())
        if saved.get("algorithm") != algorithm:
            parser.error("Resume requires a run created by this Pareto MAPPO launcher")
        excluded = {"help", "resume", "self_check", "print_config", "output_dir"}
        saved_defaults = {action.dest: saved[action.dest] for action in parser._actions
                          if action.dest in saved and action.dest not in excluded}
        parser.set_defaults(**saved_defaults, output_dir=str(run_dir))
        # Explicit command-line options still take precedence, then the runner
        # checks whether changing them is legal for an exact resume.
        args = parser.parse_args(argv)
        if Path(args.output_dir).expanduser().resolve() != run_dir:
            parser.error("Resume must use the original run directory")
    if not math.isfinite(args.gae_lambda) or not 0 <= args.gae_lambda <= 1:
        parser.error("--gae-lambda must be finite and in [0, 1]")
    if args.coordination_mode != "none":
        parser.error("Shared Pareto requires --coordination-mode none")
    if args.placement_only_episodes != 0:
        parser.error("This design has no allocation freeze; use --placement-only-episodes 0")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    if args.self_check:
        from test_pareto_mappo import self_check
        self_check()
        return
    print_config = args.print_config
    del args.self_check, args.print_config
    runner.validate_arguments(args)
    metadata = {
        "algorithm": "local_pareto_shared_ppo_gae",
        "framework": "five placement PPO actors + five local sampled-PPO allocation actors + two global V critics",
        "environment_file": str(Path(ver2.__file__).resolve()),
        "allocation_credit": "shared team GAE advantage; sampled-action clipped PPO; no counterfactual reward/state/Q evaluation",
        "allocation_value_metric_semantics": {
            "allocation_value": "V(s) on actual rollout states after critic fitting",
            "allocation_target": "GAE advantage + frozen pre-update V(s), scaled by reward_scale",
            "allocation_q": "not computed by this baseline; legacy CSV columns are unavailable",
            "allocation_return_mean_units": "unscaled Monte Carlo reward-to-go",
            "value_target_units": "environment reward divided by reward_scale",
            "reward_scale": physics.num_ues,
        },
        "baseline_controls": "compact 11D local inputs, residual, no critic cost table, shared 135 PRBs, own 2W, Pareto masks, PPO-GAE",
        "allocation_mask_mode": "pareto",
        "allocation_mask_scope": "coverage + QoS + remaining shared PRBs/own power + resource Pareto frontier; reject always available",
        "allocation_action_order": "participate -> masked PRB -> singleton power from Pareto frontier",
        "allocation_actor_feature_version": "compact_qos_link_residual_pareto_v1",
        "allocation_actor_local_feature_dim": 11,
        "allocation_critic_observation_scope": "global UAV/UE physical state and links; no Pareto mask or serving-cost table as network input",
        "allocation_critic_feature_version": "global_attention_no_cost_v1",
        "allocation_critic_cost_feature_dim": 0,
        "allocation_power_head": "no trainable parameters; singleton categorical at the Pareto-selected power",
        "popart_scope": "placement critic only",
        "maps_per_global": runner.MAPS_PER_GLOBAL,
        "placements_per_map": runner.PLACEMENTS_PER_MAP,
        "placement_samples_per_global": runner.PLACEMENTS_PER_GLOBAL,
        "shared_runner_file": str(Path(runner.__file__).resolve()),
    }
    if print_config:
        print(json.dumps({**vars(args), **metadata}, indent=2, default=str))
        return
    print(f"Pareto MAPPO | GAE lambda={args.gae_lambda} | device={args.device} | "
          f"output={Path(args.output_dir).expanduser().resolve()}", flush=True)
    runner.main(
        args=args, learner_class=pareto_mappo.ParetoMAPPO, environment_class=pareto_mappo.SharedParetoEnv,
        learner_kwargs={"gae_lambda": args.gae_lambda}, configuration_overrides=metadata,
        source_paths=[Path(__file__), *(Path(module.__file__) for module in
                     (baseline, pareto_mappo, physics, ver2, networks, learning_credit, kmeans_baseline))],
    )


if __name__ == "__main__":
    main()
