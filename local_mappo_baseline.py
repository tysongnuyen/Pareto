"""Local MAPPO control experiment, sharing ver2.gym_env and the existing networks.

Only allocation learning changes: sampled-action PPO + GAE, no simulator
counterfactuals and no COMA baseline. Placement learning, local actor inputs,
QoS/link residual branch, coverage mask, critic cost features, reward scaling,
entropy regularizer and per-actor KL stopping are inherited unchanged. This is
an architecture-matched MAPPO baseline, not a reproduction of a paper's network.

Allocation equations (r is scaled by reward_scale):
    delta_t = r_t + gamma * (1-done_t) * V_old(s_{t+1}) - V_old(s_t)
    A_t = delta_t + gamma * lambda * (1-done_t) * A_{t+1}
    value_target_t = A_t + V_old(s_t)
    actor_loss = -mean(min(ratio*A, clip(ratio, 1-eps, 1+eps)*A))
One team advantage is shared by all UAVs. Forced-reject actors have no policy
loss for that transition; the critic still trains on every real transition.

Only COMPLETE allocation episodes are accepted, including batches containing
multiple episodes. V_old and all targets are frozen before either network is
updated. The inherited action catalogs are probability tables for exact KL and
entropy only: no alternative action rewards/states/Q-values are evaluated.

Use LocalMAPPOBaseline.from_env(gym_env(coordination_mode="none"), ...).
The existing train_frozen_placement_batch can collect/update this learner;
its compatibility hook prepare_counterfactuals deliberately does nothing.
Do not run the old training CLI expecting it to select this class automatically.

Review/self-check only (no experiment runner or persistent training outputs):
    python local_mappo_baseline.py --self-check
"""
from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from learning_credit import categorical_entropy, parameter_vector, stack_states
from strict_local_independent_mappo import StrictLocalIndependentMAPPO, _masked_distribution


def complete_episode_gae(
    rewards: Tensor, old_values: Tensor, dones: Tensor, *, gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """GAE with zero bootstrap at EACH terminal boundary, in scaled units."""
    if rewards.ndim != 1 or not rewards.numel() or old_values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("GAE requires equally sized, nonempty 1-D tensors")
    if (not rewards.is_floating_point() or rewards.dtype != old_values.dtype
            or rewards.device != old_values.device or rewards.device != dones.device):
        raise ValueError("GAE rewards/values must share a floating dtype and all tensors a device")
    if dones.dtype != torch.bool or not bool(dones[-1]):
        raise ValueError("GAE requires boolean dones and complete episodes")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    if not bool(torch.isfinite(rewards).all() and torch.isfinite(old_values).all()):
        raise ValueError("GAE rewards/values must be finite")
    with torch.no_grad():
        advantages = torch.zeros_like(rewards)
        running = rewards.new_zeros(())
        next_value = rewards.new_zeros(())
        for t in reversed(range(len(rewards))):
            continuation = (~dones[t]).to(rewards.dtype)
            delta = rewards[t] + gamma * continuation * next_value - old_values[t]
            running = delta + gamma * gae_lambda * continuation * running
            advantages[t] = running
            next_value = old_values[t]
        return advantages, advantages + old_values.detach()


def sampled_ppo_loss(log_probs: Tensor, old_log_probs: Tensor,
                     advantages: Tensor, clip_epsilon: float) -> tuple[Tensor, Tensor]:
    """PPO surrogate on sampled actions, not an expectation over alternatives."""
    ratios = (log_probs - old_log_probs.detach()).exp()
    advantages = advantages.detach()
    loss = -torch.minimum(ratios * advantages,
                          ratios.clamp(1-clip_epsilon, 1+clip_epsilon) * advantages).mean()
    return loss, ((ratios - 1).abs() > clip_epsilon).float().mean()


def sampled_action_log_prob(actor, context, decision) -> Tensor:
    """Sum head log-probabilities, matching collection without probability underflow."""
    action = decision.action
    log_prob = _masked_distribution(actor.participate_logits(context), decision.action_mask).log_prob(action[0])
    if int(action[0]) == 1:
        log_prob = log_prob + _masked_distribution(actor.prb_logits(context), decision.prb_mask).log_prob(action[1])
        log_prob = log_prob + _masked_distribution(
            actor.power_logits(context, action[1]), decision.pair_mask[action[1]]).log_prob(action[2])
    return log_prob


class LocalMAPPOBaseline(StrictLocalIndependentMAPPO):
    """Same actors/critics as the none model, different allocation estimator."""

    def __init__(self, *, gae_lambda: float = 0.95, **kwargs):
        if kwargs.get("coordination_mode", "none") != "none":
            raise ValueError("This baseline requires coordination_mode='none'")
        if not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1:
            raise ValueError("gae_lambda must be finite and in [0, 1]")
        super().__init__(**kwargs)
        self.gae_lambda = float(gae_lambda)
        # Parent sets these per-instance: override after initialization.
        self.CHECKPOINT_FRAMEWORK = "local_mappo_sampled_ppo_gae_v1"
        self.CHECKPOINT_SCHEMA_VERSION = 1

    def prepare_counterfactuals(self, environment, record):
        """Compatibility with the shared collector; never query the simulator."""
        del environment
        if record.reward is not None or record.outcome_states is not None:
            raise RuntimeError("Expected a fresh real transition")

    def store_allocation_outcome(self, record, *, team_reward: float, done: bool):
        if record.reward is not None or record.done is not None:
            raise RuntimeError("This allocation transition already has an outcome")
        if not math.isfinite(team_reward):
            raise ValueError("team_reward must be finite")
        record.reward, record.done = float(team_reward), bool(done)
        self._allocation_rollout.append(record)

    def update_allocation_rollout(self, *, bootstrap_global_state, bootstrap_done):
        del bootstrap_global_state
        records = self._allocation_rollout
        if not records or any(r.reward is None or r.done is None for r in records):
            raise RuntimeError("A complete allocation rollout is required")
        if not bootstrap_done or not records[-1].done:
            raise ValueError("Allocation updates require complete episodes")
        n = len(records)
        minibatch = min(n, self.allocation_minibatch_size)
        rewards = torch.tensor([r.reward for r in records], device=self.device) / self.reward_scale
        dones = torch.tensor([r.done for r in records], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            old_values = torch.cat([
                self.allocation_critic(stack_states([r.global_state for r in records[start:start+64]], self.device))
                for start in range(0, n, 64)
            ])
            raw_advantages, targets = complete_episode_gae(
                rewards, old_values, dones, gamma=self.gamma, gae_lambda=self.gae_lambda)
            advantages = (raw_advantages - raw_advantages.mean()) / raw_advantages.std(unbiased=False).clamp_min(1e-6)

        before = [parameter_vector(actor) for actor in self.allocation_actors]
        critic_losses, policy_losses, entropies, clips = [], [], [], []
        gradients = [[] for _ in range(self.num_uavs)]
        # Same critic minibatch limit and epochs as the current learner.
        for _ in range(self.allocation_ppo_epochs):
            for indices in torch.randperm(n).split(min(minibatch, 64)):
                prediction = self.allocation_critic(stack_states(
                    [records[i].global_state for i in indices.tolist()], self.device))
                loss = F.mse_loss(prediction, targets[indices.to(self.device)])
                self.allocation_critic_optimizer.zero_grad(set_to_none=True)
                (self.value_coef * loss).backward()
                nn.utils.clip_grad_norm_(self.allocation_critic.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                self.allocation_critic_optimizer.step()
                self.optimizer_step_totals["allocation_critic"] += 1
                critic_losses.append(float(loss.detach()))

        stopped = [False] * self.num_uavs
        actor_steps = [0] * self.num_uavs
        steps = sample_uses = 0
        final_catalogs = []
        for _ in range(self.allocation_ppo_epochs):
            for indices in torch.randperm(n).split(minibatch):
                updated = False
                for uav, (actor, optimizer) in enumerate(zip(self.allocation_actors, self.allocation_actor_optimizers)):
                    if stopped[uav]:
                        continue
                    ids, logs, old_logs, entropy_terms, regularizers = [], [], [], [], []
                    for i in indices.tolist():
                        decision = records[i].decisions[uav]
                        if len(decision.counterfactual_probabilities) <= 1:
                            continue
                        context = actor.encode(decision.observation)
                        actions, probabilities = self._allocation_action_catalog(
                            actor, decision.observation, decision.action_mask, decision.pair_mask, context=context)
                        selected = (actions == decision.action).all(-1)
                        if int(selected.sum()) != 1:
                            raise RuntimeError("Sampled action is outside its stored mask")
                        ids.append(i)
                        logs.append(sampled_action_log_prob(actor, context, decision))
                        old_logs.append(decision.old_log_prob)
                        entropy_terms.append(categorical_entropy(probabilities))
                        regularizers.append(self._allocation_entropy_regularizer(actions, probabilities))
                    optimizer.zero_grad(set_to_none=True)
                    if not ids:
                        continue
                    loss, clip = sampled_ppo_loss(torch.stack(logs), torch.stack(old_logs),
                        advantages[torch.tensor(ids, device=self.device)], self.clip_epsilon)
                    entropy = torch.stack(entropy_terms).mean()
                    (loss - self.allocation_entropy_coef * torch.stack(regularizers).mean()).backward()
                    norm = nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                    gradients[uav].append(float(norm))
                    if float(norm) > 0:
                        optimizer.step()
                        self.optimizer_step_totals["allocation_actor"] += 1
                        actor_steps[uav] += 1
                        updated = True
                    sample_uses += len(ids)
                    policy_losses.append(float(loss.detach()))
                    entropies.append(float(entropy.detach()))
                    clips.append(float(clip.detach()))
                steps += int(updated)
                kl = self._mean_allocation_policy_kl(policy_catalogs=final_catalogs)
                for uav, actor_kl in enumerate(self._last_allocation_kl_by_agent):
                    stopped[uav] |= actor_kl > self.target_kl
                if all(stopped):
                    break
            if all(stopped):
                break

        mean = lambda xs: float(np.mean(xs)) if xs else 0.0
        with torch.no_grad():
            values = torch.cat([
                self.allocation_critic(stack_states([r.global_state for r in records[start:start+64]], self.device))
                for start in range(0, n, 64)])
            entropy_parts = [[float(x) for x in self._allocation_entropy_components(a, p)]
                             for a, p in final_catalogs]
        metrics = {
            "allocation_actor_loss": mean(policy_losses), "allocation_critic_loss": mean(critic_losses),
            "allocation_entropy": mean(entropies), "allocation_clip_fraction": mean(clips),
            "allocation_value_mean": float(values.mean()), "allocation_value_max_abs": float(values.abs().max()),
            "allocation_value_target_mean": float(targets.mean()),
            "allocation_return_mean": float(self._compute_allocation_returns().mean()) * self.reward_scale,
            "allocation_rollout_return": sum(r.reward for r in records), "allocation_rollout_size": float(n),
            "allocation_gradient_steps": float(steps), "allocation_critic_gradient_steps": float(len(critic_losses)),
            "allocation_policy_kl": kl, "allocation_kl_early_stop": float(any(stopped)),
            "allocation_policy_sample_uses": float(sample_uses),
            "allocation_gae_advantage_mean": float(raw_advantages.mean()), "allocation_gae_lambda": self.gae_lambda,
            "allocation_counterfactual_previews": 0.0,
            "allocation_reject_fraction": mean([int(d.action[0]) == 0 for r in records for d in r.decisions]),
        }
        for j, name in enumerate(("participation", "prb", "power")):
            metrics[f"allocation_{name}_entropy"] = mean([parts[j] for parts in entropy_parts])
        for uav, actor in enumerate(self.allocation_actors):
            prefix = f"allocation_uav_{uav}_"
            metrics.update({prefix + "gradient_norm": mean(gradients[uav]),
                prefix + "parameter_delta": float((parameter_vector(actor)-before[uav]).norm()),
                prefix + "policy_kl": self._last_allocation_kl_by_agent[uav],
                prefix + "kl_early_stop": float(stopped[uav]), prefix + "optimizer_steps": float(actor_steps[uav])})
        if self.diagnostics_callback is not None:
            self.diagnostics_callback("allocation", metrics)
        records.clear()
        return metrics

    def save_checkpoint(self, path, episode, *, trainer_state=None):
        # Stage before replacing: a metadata write failure must preserve the old file.
        path = Path(path)
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            staged = Path(directory) / "checkpoint.pt"
            super().save_checkpoint(staged, episode, trainer_state=trainer_state)
            checkpoint = torch.load(staged, map_location="cpu", weights_only=False)
            checkpoint.update(credit_estimator="sampled_ppo_gae", gae_lambda=self.gae_lambda)
            with staged.open("wb") as handle:
                torch.save(checkpoint, handle)
                handle.flush()
                os.fsync(handle.fileno())
            staged.replace(path)

    def load_checkpoint(self, path, *, map_location=None, restore_training_state=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint.get("framework") != self.CHECKPOINT_FRAMEWORK
                or checkpoint.get("credit_estimator") != "sampled_ppo_gae"):
            raise ValueError("Expected a local MAPPO baseline checkpoint, not a counterfactual model")
        if checkpoint.get("gae_lambda") != self.gae_lambda:
            raise ValueError("Checkpoint gae_lambda does not match this baseline")
        del checkpoint
        return super().load_checkpoint(path, map_location=map_location,
                                       restore_training_state=restore_training_state)


def self_check():
    """Small numerical and real-env checks; never launch a baseline experiment."""
    from unittest.mock import patch
    import strict_local_independent_mappo as original
    from ver2 import gym_env
    from kmeans_baseline import kmeans_centers, kmeans_placement
    from types import SimpleNamespace

    torch.set_num_threads(1)
    torch.manual_seed(7)
    rewards = torch.tensor([1., 2., 3., 4.])
    values = torch.tensor([.5, .6, .7, .8])
    dones = torch.tensor([False, True, False, True])
    advantages, targets = complete_episode_gae(rewards, values, dones, gae_lambda=.5)
    torch.testing.assert_close(advantages, torch.tensor([1.8, 1.4, 4.7, 3.2]))
    torch.testing.assert_close(targets, torch.tensor([2.3, 2., 5.4, 4.]))
    _, mc_targets = complete_episode_gae(rewards, values, dones, gae_lambda=1)
    torch.testing.assert_close(mc_targets, torch.tensor([3., 2., 7., 4.]))
    try:
        complete_episode_gae(rewards, values, torch.zeros(4, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete episodes were accepted")
    logs = torch.tensor([1.4, .6, 1.1, .9]).log().requires_grad_()
    loss, clip = sampled_ppo_loss(logs, torch.zeros(4), torch.tensor([2., -2., 2., -2.]), .2)
    torch.testing.assert_close(loss, torch.tensor(-.3))
    torch.testing.assert_close(clip, torch.tensor(.5))
    loss.backward()
    torch.testing.assert_close(logs.grad, torch.tensor([0., 0., -.55, .45]))

    environment = gym_env(coordination_mode="none", power_weight=.2)
    settings = dict(embed_dim=16, num_heads=4, allocation_ppo_epochs=1, allocation_minibatch_size=50)
    learner = LocalMAPPOBaseline.from_env(environment, **settings)
    reference = StrictLocalIndependentMAPPO.from_env(environment, **settings)
    assert {k: v.shape for k,v in learner.state_dict().items()} == {k: v.shape for k,v in reference.state_dict().items()}
    reference.load_state_dict(learner.state_dict())
    before = parameter_vector(learner.allocation_critic)
    with patch.object(environment, "preview_allocation_step", side_effect=AssertionError("Unexpected simulator preview")), \
         patch.object(learner, "_counterfactual_values", side_effect=AssertionError("Unexpected Q_cf")), \
         patch.object(original, "expected_clipped_loss", side_effect=AssertionError("Unexpected all-action PPO")):
        for seed in (434, 435):
            state, _ = environment.reset(options={"ue_seed": seed})
            centers = kmeans_centers(np.array([(u.x_m, u.y_m) for u in environment.ues]), 5, ue_seed=seed)
            placement = kmeans_placement(environment, centers, height=250, theta=55, assignment_order=range(5))
            state, _, _, _, _ = environment.step(placement)
            observations = environment.get_actor_observations()
            actual, _ = learner.act_allocation(state, observations, deterministic=True, store_for_update=False)
            expected, _ = reference.act_allocation(state, observations, deterministic=True, store_for_update=False)
            np.testing.assert_array_equal(actual, expected)
            original._collect_allocation_episode(environment, learner, state, defer_allocation_update=True)
        assert learner.allocation_rollout_size == 100
        assert all(r.outcome_states is None for r in learner._allocation_rollout)
        metrics = learner.update_allocation_rollout(bootstrap_global_state=None, bootstrap_done=True)
    assert learner.allocation_rollout_size == 0
    assert all(math.isfinite(v) for v in metrics.values())
    assert not torch.equal(before, parameter_vector(learner.allocation_critic))
    assert learner.optimizer_step_counts()["allocation_actor"] > 0
    assert any(metrics[f"allocation_uav_{i}_parameter_delta"] > 0 for i in range(5))
    # A sampled action may become very unlikely during PPO. Its log-probability
    # must remain -120, not log(1e-38)=-87.5 with a zero gradient.
    actor = reference.allocation_actors[0]
    with torch.no_grad():
        actor.participate_head.weight.zero_()
        actor.participate_head.bias.copy_(torch.tensor([-120., 0.]))
    actor.zero_grad(set_to_none=True)
    rare = SimpleNamespace(action=torch.tensor([0, 0, 0]), action_mask=torch.tensor([True, True]))
    rare_log = sampled_action_log_prob(actor, torch.zeros(settings["embed_dim"]), rare)
    torch.testing.assert_close(rare_log, torch.tensor(-120.))
    rare_log.backward()
    torch.testing.assert_close(actor.participate_head.bias.grad, torch.tensor([1., -1.]))
    # All UAVs forced to reject: still train V, never move the actors.
    actor_before = parameter_vector(learner.allocation_actors)
    environment.reset(options={"ue_seed": 434})
    state, _, _, _, _ = environment.step(placement)
    environment.link_features[:, :, 1] = 0
    state = environment._get_obs()
    original._collect_allocation_episode(environment, learner, state, defer_allocation_update=True)
    forced_metrics = learner.update_allocation_rollout(bootstrap_global_state=None, bootstrap_done=True)
    torch.testing.assert_close(actor_before, parameter_vector(learner.allocation_actors), rtol=0, atol=0)
    assert forced_metrics["allocation_policy_sample_uses"] == 0
    assert all(math.isfinite(v) for v in forced_metrics.values())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "baseline.pt"
        learner.save_checkpoint(path, episode=2)
        restored = LocalMAPPOBaseline.from_env(environment, **settings)
        assert restored.load_checkpoint(path) == 2
        for key, value in learner.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        wrong_lambda = LocalMAPPOBaseline.from_env(environment, gae_lambda=.5, **settings)
        try:
            wrong_lambda.load_checkpoint(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched GAE lambda was accepted")
        try:
            reference.load_checkpoint(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Original learner accepted a baseline checkpoint")
    print("PASS: hand-calculated GAE, episode boundaries, sampled PPO loss/gradient, identical actor architecture/actions,")
    print("100 real transitions, zero counterfactual previews, finite update, rare-action log-probability, forced reject, checkpoint isolation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="Run small checks only; do not launch training")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        parser.print_help()
