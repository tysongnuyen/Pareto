"""Run with python test_pareto_mappo.py; no persistent training outputs."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from pareto_mappo import SharedParetoEnv, ParetoMAPPO
from local_mappo_baseline import LocalMAPPOBaseline
from ver2 import gym_env
import strict_local_independent_mappo as networks
import env as physics
from kmeans_baseline import kmeans_centers, kmeans_placement


def self_check():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    env = SharedParetoEnv(power_weight=.2, prb_weight=1.)
    settings = dict(embed_dim=16, num_heads=4, allocation_ppo_epochs=1, allocation_minibatch_size=50)
    model = ParetoMAPPO.from_env(env, **settings)
    assert model.allocation_actors[0].local_vector_encoder[0].in_features == 11
    assert not env.include_allocation_cost_features
    assert model.allocation_critic.allocation_cost_dim == 0
    assert isinstance(model.placement_critic.popart_head, networks.PopArtLinear)
    assert not any(isinstance(m, networks.PopArtLinear) for m in model.allocation_critic.modules())
    assert all(actor.power_head is None for actor in model.allocation_actors)
    assert not any(".power_head." in key for key in model.state_dict())
    def place(seed):
        env.reset(options={"ue_seed": seed})
        centers = kmeans_centers(np.array([(u.x_m,u.y_m) for u in env.ues]), 5, ue_seed=seed)
        placement = kmeans_placement(env, centers, height=250, theta=55, assignment_order=range(5))
        return env.step(placement)[0]
    state = place(434)
    # No cost table reaches the critic, even if the Gym-compatible field is poisoned.
    state["allocation_cost_features"] = np.full_like(state["allocation_cost_features"], np.nan)
    tensors = model._global_state_to_tensors(state, include_candidate_features=False)
    assert "allocation_cost_features" not in tensors
    assert bool(torch.isfinite(model.allocation_critic(tensors)))
    # Removed geometry/limits/broadcast do not enter the compact MLP.
    obs = model._allocation_observation_to_tensors(env.get_actor_observations()[0])
    actor = model.allocation_actors[0]
    with torch.no_grad():
        expected = actor.encode(obs)
        obs["own_uav_features"][:5] = float("nan")
        obs["own_uav_limits"].fill_(float("nan"))
        obs["current_ue_features"][:2] = float("nan")
        obs["coordination_features"].fill_(float("nan"))
        torch.testing.assert_close(actor.encode(obs), expected, rtol=0, atol=0)
    # Scalar simulator oracle, including shrinking shared and private budgets.
    for prbs, power in [(135, 2.), (3, .3), (0, 0.)]:
        env.shared_prbs_remaining = prbs
        env.remaining_power_w.fill(power)
        for ue in range(env.num_ues):
            env.current_ue = ue
            masks = env.get_allocation_action_masks()
            assert masks['participate'][:,0].all()
            for u in range(5):
                feasible=[]
                for ni,n in enumerate(env.prb_levels):
                    for pi,p in enumerate(env.power_levels_w):
                        proposal=env._evaluate_allocation_proposal(u,pi,int(n),float(p))
                        if proposal['reason']=='accepted': feasible.append((ni,pi))
                frontier={(n,p) for n,p in feasible if not any(m<=n and q<=p and (m,q)!=(n,p) for m,q in feasible)}
                assert set(map(tuple,np.argwhere(masks['pair'][u]))) == frontier
                assert (masks['pair'][u].sum(axis=1)<=1).all()
    # Real PPO update must not invoke preview or counterfactual values.
    before=[p.detach().clone() for p in model.allocation_critic.parameters()]
    with patch.object(env,'preview_allocation_step',side_effect=AssertionError('counterfactual preview')), patch.object(model,'_counterfactual_values',side_effect=AssertionError('counterfactual critic')):
        for seed in (434,435):
            state=place(seed)
            networks._collect_allocation_episode(env,model,state,defer_allocation_update=True)
        assert model.allocation_rollout_size==100
        assert all(r.outcome_states is None for r in model._allocation_rollout)
        for record in model._allocation_rollout:
            assert "allocation_cost_features" not in record.global_state
            for decision in record.decisions:
                if int(decision.action[0]):
                    assert decision.pair_mask[decision.action[1], decision.action[2]]
                    assert decision.pair_mask[decision.action[1]].count_nonzero() == 1
        metrics=model.update_allocation_rollout(bootstrap_global_state=None,bootstrap_done=True)
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a,b) for a,b in zip(before,model.allocation_critic.parameters()))
    assert model.optimizer_step_counts()['allocation_actor']>0
    assert abs(metrics["allocation_power_entropy"]) < 1e-6
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'model.pt'
        model.save_checkpoint(path,1)
        restored=ParetoMAPPO.from_env(env,**settings)
        restored.load_checkpoint(path)
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,restored.state_dict()[k])
        assert model.optimizer_step_counts() == restored.optimizer_step_counts()
        state = place(436)
        observations = env.get_actor_observations()
        a, _ = model.act_allocation(state, observations, deterministic=True, store_for_update=False)
        b, _ = restored.act_allocation(state, observations, deterministic=True, store_for_update=False)
        np.testing.assert_array_equal(a, b)
        try: LocalMAPPOBaseline.from_env(gym_env(),**settings).load_checkpoint(path)
        except ValueError: pass
        else: raise AssertionError('Cross-framework checkpoint accepted')
        # Attention supports a different number of UEs without resizing a checkpoint.
        for count in (20, 70):
            with patch.object(physics, "num_ues", count):
                evaluation_env = SharedParetoEnv(power_weight=.2, prb_weight=1.)
                evaluation_model = ParetoMAPPO.from_env(
                    evaluation_env, reward_scale=model.reward_scale, **settings
                )
                evaluation_model.load_checkpoint(path)
                result = networks.evaluate_one_episode(
                    evaluation_env, evaluation_model, ue_seed=434
                )
                assert 0 <= result["served_count"] <= count
                assert np.isfinite(result["episode_return"])
    from train_pareto_mappo import parse_arguments
    import train_strict_local_mappo as runner
    args=parse_arguments([])
    runner.validate_arguments(args)
    assert (runner.MAPS_PER_GLOBAL,runner.PLACEMENTS_PER_MAP,runner.PLACEMENTS_PER_GLOBAL)==(2,5,10)
    assert args.placement_episodes==710 and args.placement_only_episodes==0
    assert args.allocation_batch_episodes==10 and args.placement_batch_episodes==10
    print('PASS: scalar Pareto oracle, budgets/reject, compact residual inputs, no-cost critic, '
          '100 real PPO transitions, no CF/power head, checkpoint isolation, '
          '20/70-UE inference, 2x5 schedule')

if __name__=='__main__': self_check()
