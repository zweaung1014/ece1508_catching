import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch
import numpy as np

from gym_dcmm.algs.ppo_dcmm.experience import ExperienceBuffer
from gym_dcmm.algs.ppo_dcmm.models_hoop import ActorCritic
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_hoop import PPO_Hoop, AdaptiveScheduler, policy_kl
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd


def make_model():
    return ActorCritic({
        'actor_units': [16, 8], 'actions_num': 5,
        'input_shape': (35,), 'separate_value_mlp': True,
    })


def make_update():
    """A real PPO optimizer update on a small, fixed synthetic rollout."""
    torch.manual_seed(42)
    agent = PPO_Hoop.__new__(PPO_Hoop)
    agent.model = make_model()
    agent.running_mean_std = RunningMeanStd((35,)).eval()
    agent.value_mean_std = RunningMeanStd((1,)).eval()
    agent.normalize_input = True
    agent.normalize_value = True
    agent.storage = ExperienceBuffer(2, 4, 8, 4, 35, 5, 'cpu')
    raw_obs = torch.randn(4, 2, 35) + 2.0
    with torch.no_grad():
        normalized = agent.running_mean_std(raw_obs.reshape(8, 35)).reshape(4, 2, 35)
        for t in range(4):
            result = agent.model.act({'obs': normalized[t]})
            agent.storage.update_data('obses', t, raw_obs[t])
            agent.storage.update_data('normalized_obses', t, normalized[t])
            for key in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                agent.storage.update_data(key, t, result[key])
        agent.storage.storage_dict['rewards'].copy_(torch.randn(4, 2, 1))
        agent.storage.storage_dict['dones'][-1] = 1
        agent.storage.compute_return(torch.zeros(2, 1), 0.99, 0.95)
        agent.storage.prepare_training()
    agent.mini_epochs_num = 3
    agent.bounds_loss_coef = 0.0001
    agent.e_clip = 0.2
    agent.clip_value_loss = False
    agent.critic_coef = 4.0
    agent.entropy_coef = 0.0
    agent.truncate_grads = True
    agent.grad_norm = 1.0
    agent.last_lr = agent.init_lr = 0.0005
    agent.lr_schedule = 'kl'
    agent.kl_threshold = 0.016
    agent.scheduler = AdaptiveScheduler(agent.kl_threshold)
    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=agent.last_lr)
    agent.data_collect_time = agent.rl_train_time = 0.0
    agent.extra_info = {}
    agent.play_steps = lambda: None
    return agent


class PPOUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_bounds_penalty_has_gradient_when_tanh_is_saturated(self):
        model = make_model()
        with torch.no_grad():
            model.mu.weight.zero_()
            model.mu.bias.fill_(100.0)
        result = model({'obs': torch.zeros(2, 35), 'prev_actions': torch.zeros(2, 5)})
        self.assertTrue(torch.equal(result['mus'], torch.ones(2, 5)))
        penalty = model.mean_bounds_loss(result['raw_mus']).mean()
        penalty.backward()
        self.assertGreater(penalty.item(), 0.0)
        self.assertTrue((model.mu.bias.grad > 0).all())

    def test_exploration_limits_and_projection(self):
        model = make_model()
        with torch.no_grad():
            model.sigma.fill_(10.0)
        result = model.act({'obs': torch.zeros(2, 35)})
        self.assertTrue((result['sigmas'] <= 1.0).all())
        model.project_log_std()
        self.assertTrue((model.sigma <= model.log_std_max).all())
        with torch.no_grad():
            model.sigma.fill_(-10.0)
        model.project_log_std()
        self.assertTrue((model.sigma >= model.log_std_min).all())

    def test_kl_matches_normal_distribution_reference(self):
        old_mu, new_mu = torch.zeros(3, 5), torch.full((3, 5), 0.4)
        old_std, new_std = torch.ones(3, 5), torch.full((3, 5), 0.8)
        self.assertEqual(policy_kl(old_mu, old_std, old_mu, old_std).item(), 0.0)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Normal(old_mu, old_std),
            torch.distributions.Normal(new_mu, new_std),
        ).sum(-1).mean()
        torch.testing.assert_close(policy_kl(old_mu, old_std, new_mu, new_std), expected)

    def test_rollout_reference_and_inputs_stay_fixed_during_update(self):
        agent = make_update()
        data = agent.storage.data_dict
        reference = {key: data[key].clone() for key in ['mus', 'sigmas', 'neglogpacs', 'normalized_obses']}
        count = agent.running_mean_std.count.item()
        first_ratios = []

        def inspect_forward(module, inputs, result):
            self.assertEqual(agent.running_mean_std.count.item(), count)
            self.assertFalse(agent.running_mean_std.training)
            if not first_ratios:
                old_logp = reference['neglogpacs'][:agent.storage.minibatch_size]
                first_ratios.append(torch.exp(old_logp - result['prev_neglogp']).detach())

        hook = agent.model.register_forward_hook(inspect_forward)
        try:
            losses = agent.train_epoch()
        finally:
            hook.remove()
        torch.testing.assert_close(first_ratios[0], torch.ones_like(first_ratios[0]))
        for key, value in reference.items():
            torch.testing.assert_close(data[key], value, rtol=0, atol=0)
        self.assertEqual(agent.running_mean_std.count.item(), count + agent.storage.batch_size)
        self.assertLessEqual(agent.last_lr, agent.init_lr * 1.5)
        for group in losses:
            self.assertTrue(torch.isfinite(torch.stack(group)).all())

    def test_kl_limit_stops_repeated_updates(self):
        agent = make_update()
        agent.kl_threshold = 1e-12
        losses = agent.train_epoch()
        self.assertEqual(len(losses[-1]), 1)
        self.assertEqual(agent.extra_info['info/ppo_epochs'], 1)

    def test_running_statistics_single_sample_and_common_units(self):
        rms = RunningMeanStd((1,)).eval()
        values = torch.tensor([[0.0], [1.0]])
        returns = torch.tensor([[10.0], [11.0]])
        rms.update(returns)
        state = {key: value.clone() for key, value in rms.state_dict().items()}
        normalized_values = rms(values)
        normalized_returns = rms(returns)
        for key, value in state.items():
            torch.testing.assert_close(rms.state_dict()[key], value)
        self.assertEqual(rms.count.item(), 3.0)
        delta = (returns - values) / torch.sqrt(rms.running_var.float() + rms.epsilon)
        torch.testing.assert_close(normalized_returns - normalized_values, delta)
        rms.update(torch.tensor([[12.0]]))
        self.assertTrue(torch.isfinite(rms.running_var).all())

    def test_disabled_normalization_is_respected(self):
        agent = make_update()
        agent.normalize_input = agent.normalize_value = False
        raw = torch.full((2, 35), 10.0)
        result = agent.model_act({'obs': raw})
        torch.testing.assert_close(result['normalized_obses'], raw)
        self.assertEqual(agent.running_mean_std.count.item(), 1.0)
        self.assertEqual(agent.value_mean_std.count.item(), 1.0)

    def resumable_agent(self):
        agent = make_update()
        agent.device = 'cpu'
        agent.seed = 42
        agent.batch_size = agent.storage.batch_size
        agent.ppo_config = {'resume_agent_steps': 0}
        agent.agent_steps = 4096
        agent.epoch_num = 511
        agent.best_rewards = 12.5
        agent.max_agent_steps = 25000000
        return agent

    def test_checkpoint_restores_optimizer_value_statistics_and_progress(self):
        source = self.resumable_agent()
        source.train_epoch()  # Populate Adam moments.
        source.best_rewards = np.float64(12.5)
        source.agent_steps = np.int64(source.agent_steps)
        source.epoch_num = np.int64(source.epoch_num)
        source.value_mean_std.update(torch.tensor([[10.0], [12.0]]))
        with TemporaryDirectory(dir=Path('outputs').resolve()) as folder:
            filename = str(Path(folder) / 'last')
            source.save(filename)
            self.assertFalse(Path(filename + '.pth.tmp').exists())
            target = self.resumable_agent()
            target.agent_steps = target.epoch_num = 0
            target.restore_train(filename + '.pth')
        self.assertEqual(target.agent_steps, source.agent_steps)
        self.assertEqual(target.epoch_num, source.epoch_num)
        self.assertEqual(target.last_lr, source.last_lr)
        self.assertEqual(target.best_rewards, source.best_rewards)
        for name, value in source.value_mean_std.state_dict().items():
            torch.testing.assert_close(target.value_mean_std.state_dict()[name], value)
        expected = source.optimizer.state_dict()
        actual = target.optimizer.state_dict()
        self.assertEqual(actual['param_groups'], expected['param_groups'])
        for index, state in expected['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(actual['state'][index][key], value)
        target.env = Mock()
        target.env.reset.return_value = (torch.zeros(2, 35), {})
        target.obs2tensor = lambda obs: obs
        target.train_epoch = Mock(side_effect=AssertionError('Progress was reset'))
        target.max_agent_steps = target.agent_steps
        target.train()
        target.train_epoch.assert_not_called()

    def test_legacy_checkpoint_restores_both_normalizers_and_manual_progress(self):
        source = self.resumable_agent()
        source.value_mean_std.update(torch.tensor([[10.0], [12.0]]))
        legacy = {'model': source.model.state_dict(),
                  'running_mean_std': source.running_mean_std.state_dict(),
                  'value_mean_std': source.value_mean_std.state_dict()}
        with TemporaryDirectory(dir=Path('outputs').resolve()) as folder:
            filename = str(Path(folder) / 'legacy.pth')
            torch.save(legacy, filename)
            target = self.resumable_agent()
            target.ppo_config['resume_agent_steps'] = 32
            target.restore_train(filename)
        self.assertEqual(target.agent_steps, 32)
        self.assertEqual(target.epoch_num, 3)
        self.assertEqual(len(target.optimizer.state_dict()['state']), 0)
        for name, value in source.value_mean_std.state_dict().items():
            torch.testing.assert_close(target.value_mean_std.state_dict()[name], value)

    def test_failed_checkpoint_write_preserves_previous_file(self):
        agent = self.resumable_agent()
        with TemporaryDirectory(dir=Path('outputs').resolve()) as folder:
            filename = str(Path(folder) / 'last')
            agent.save(filename)
            previous = Path(filename + '.pth').read_bytes()
            with patch.object(torch, 'save', side_effect=OSError('Simulated write failure')):
                with self.assertRaises(OSError):
                    agent.save(filename)
            self.assertEqual(Path(filename + '.pth').read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()
