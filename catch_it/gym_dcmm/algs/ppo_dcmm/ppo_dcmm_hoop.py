import os, sys
sys.path.append(os.path.abspath('../gym_dcmm'))
import math
import time
import torch
import torch.distributed as dist

import wandb

import numpy as np

from .experience import ExperienceBuffer
from .models_hoop import ActorCritic
from .utils import AverageScalarMeter, RunningMeanStd

from tensorboardX import SummaryWriter

# Must match OUTCOME_NAMES in gym_dcmm/envs/DcmmVecEnv.py
OUTCOME_NAMES = ["none", "catch", "ball_floor", "ball_robot", "collision", "timeout"]
# Per-episode values the env reports in its last info, averaged over the last episodes
EPISODE_STATS = ["success", "min_ball_dist", "min_path_dist", "rim_hits", "ik_fails"]

class PPO_Hoop(object):
    def __init__(self, env, output_dif, full_config):
        self.rank = -1
        self.device = full_config['rl_device']
        self.seed = int(full_config.seed)
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = int(self.ppo_config['num_actors'])
        print("num_actors: ", self.num_actors)
        self.actions_num = self.env.call("act_dim")[0]
        print("actions_num: ", self.actions_num)
        self.obs_shape = (self.env.call("obs_dim")[0],)
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'separate_value_mlp': self.network_config.get('separate_value_mlp', True),
            'log_std_min': self.network_config.get('log_std_min', -5.0),
            'log_std_max': self.network_config.get('log_std_max', 0.0),
            'raw_mean_soft_bound': self.network_config.get('raw_mean_soft_bound', 2.0),
        }
        print("net_config: ", net_config)
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'nn')
        self.tb_dif = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.init_lr = float(self.ppo_config['learning_rate'])
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.init_lr, eps=1e-5)
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        self.reward_scale_value = self.ppo_config['reward_scale_value']
        self.clip_value_loss = self.ppo_config['clip_value_loss']
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.lr_schedule = self.ppo_config['lr_schedule']
        self.kl_threshold = self.ppo_config['kl_threshold']
        if self.lr_schedule == 'kl':
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.lr_schedule == 'linear':
            self.scheduler = LinearScheduler(
                self.init_lr,
                self.ppo_config['max_agent_steps'])
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)
        self.episode_stats = {k: AverageScalarMeter(200) for k in EPISODE_STATS + ['outcome/' + n for n in OUTCOME_NAMES[1:]]}
        self.episode_test_rewards = AverageScalarMeter(self.ppo_config['test_num_episodes'])
        self.episode_test_lengths = AverageScalarMeter(self.ppo_config['test_num_episodes'])
        self.episode_test_stats = {k: AverageScalarMeter(self.ppo_config['test_num_episodes']) for k in EPISODE_STATS + ['outcome/' + n for n in OUTCOME_NAMES[1:]]}

        self.obs = None
        self.epoch_num = 0
        # print("self.obs_shape[0]: ", type(self.obs_shape[0]))
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size,
            self.obs_shape[0], self.actions_num, self.device,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.session_start_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.max_test_steps = self.ppo_config['max_test_steps']
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        log_dict = {
            'performance/RLTrainFPS': (self.agent_steps - self.session_start_steps) / self.rl_train_time,
            'performance/EnvStepFPS': (self.agent_steps - self.session_start_steps) / self.data_collect_time,
            'losses/actor_loss': torch.mean(torch.stack(a_losses)).item(),
            'losses/bounds_loss': torch.mean(torch.stack(b_losses)).item(),
            'losses/critic_loss': torch.mean(torch.stack(c_losses)).item(),
            'losses/entropy': torch.mean(torch.stack(entropies)).item(),
            'info/last_lr': self.last_lr,
            'info/e_clip': self.e_clip,
            'info/kl': torch.mean(torch.stack(kls)).item(),
        }
        for k, v in self.extra_info.items():
            log_dict[f'{k}'] = v

        # log to wandb
        wandb.log(log_dict, step=self.agent_steps)

        # log to tensorboard
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        # Normalization is part of the sampled policy. Do not change it per minibatch.
        self.running_mean_std.eval()
        self.value_mean_std.eval()

    def train(self):
        start_time = time.time()
        _t = time.time()
        _last_t = time.time()
        reset_obs, _ = self.env.reset(seed=self.seed)
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        if self.agent_steps == 0:
            self.agent_steps = self.batch_size
        self.session_start_steps = self.agent_steps

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls = self.train_epoch()
            self.storage.data_dict = None

            if self.lr_schedule == 'linear':
                self.last_lr = self.scheduler.update(self.agent_steps)
            
            all_fps = (self.agent_steps - self.session_start_steps) / (time.time() - _t)
            last_fps = (
                self.batch_size ) \
                / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e3):04}K | FPS: {all_fps:.1f} | ' \
                            f'Last FPS: {last_fps:.1f} | ' \
                            f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                            f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                            f'Current Best: {self.best_rewards:.2f} | ' \
                            f'Catch: {self.episode_stats["success"].get_mean():.3f} | ' \
                            f'LR: {self.last_lr:.1e} | ' \
                            f'Sigma: {torch.exp(self.model.sigma.clamp(self.model.log_std_min, self.model.log_std_max)).mean().item():.2f}'
            # flush: the log goes to a file during long runs, and buffered lines would arrive in bursts
            print(info_string, flush=True)

            self.write_stats(a_losses, c_losses, b_losses, entropies, kls)

            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            mean_success = self.episode_stats['success'].get_mean()
            # print("mean_rewards: ", mean_rewards)
            log_dict = {
                'metrics/episode_rewards_per_step': mean_rewards,
                'metrics/episode_lengths_per_step': mean_lengths,
                'metrics/episode_success_per_step': mean_success,
            }
            for k, meter in self.episode_stats.items():
                if k != 'success':
                    log_dict[f'episode/{k}'] = meter.get_mean()
            for k, v in log_dict.items():
                self.writer.add_scalar(k, v, self.agent_steps)
            wandb.log(log_dict, step=self.agent_steps)
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'

            if self.save_freq > 0:
                if (self.epoch_num % self.save_freq == 0) and (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, f'last'))

            if mean_rewards > self.best_rewards:
                print(f'save current best reward: {mean_rewards:.2f}')
                # remove previous best file
                prev_best_ckpt = os.path.join(self.nn_dir, f'best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, f'best_reward_{mean_rewards:.2f}'))

        print('max steps achieved')
        print('data collect time: %f min' % (self.data_collect_time / 60.0))
        print('rl train time: %f min' % (self.rl_train_time / 60.0))
        print('all time: %f min' % ((time.time() - start_time) / 60.0))

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        weights['training_state'] = {
            'agent_steps': int(self.agent_steps),
            'epoch_num': int(self.epoch_num),
            'best_rewards': float(self.best_rewards),
            'last_lr': float(self.last_lr),
            'optimizer': self.optimizer.state_dict(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        # Replace only after a complete write; retain the previous file if interrupted.
        temporary_path = f'{name}.pth.tmp'
        torch.save(weights, temporary_path)
        os.replace(temporary_path, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            if self.ppo_config.get('resume_agent_steps', 0):
                raise ValueError('resume_agent_steps requires a checkpoint')
            return
        checkpoint = torch.load(fn, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self.normalize_value:
            self.value_mean_std.load_state_dict(checkpoint['value_mean_std'])
        state = checkpoint.get('training_state')
        if state is not None:
            self.optimizer.load_state_dict(state['optimizer'])
            self.agent_steps = int(state['agent_steps'])
            self.epoch_num = int(state['epoch_num'])
            self.best_rewards = float(state['best_rewards'])
            self.last_lr = float(state['last_lr'])
            torch.set_rng_state(state['torch_rng_state'].cpu())
            if torch.cuda.is_available() and state['cuda_rng_states']:
                torch.cuda.set_rng_state_all([rng.cpu() for rng in state['cuda_rng_states']])
            print('Restored optimizer, learning rate, counters and normalization.')
        else:
            self.agent_steps = int(self.ppo_config.get('resume_agent_steps', 0))
            if self.agent_steps < 0 or self.agent_steps % self.batch_size:
                raise ValueError('resume_agent_steps must be a non-negative multiple of the rollout batch size')
            self.epoch_num = max(0, self.agent_steps // self.batch_size - 1)
            print('Legacy checkpoint: restored model and normalization; Adam starts fresh. '
                  'Use learning_rate and resume_agent_steps to restore the logged rate and progress.')
        for group in self.optimizer.param_groups:
            group['lr'] = self.last_lr
        print(f'Resuming at {self.agent_steps} / {self.max_agent_steps} agent steps; lr={self.last_lr:.10g}. '
              'Simulation episodes restart; mid-episode state is not restored.')

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location = self.device)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []
        for mini_ep in range(0, self.mini_epochs_num):
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs = self.storage[i]

                # These are the exact normalized inputs used during collection.
                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                }
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                if self.clip_value_loss:
                    value_pred_clipped = value_preds + \
                        (values - value_preds).clamp(-self.e_clip, self.e_clip)
                    value_losses = (values - returns) ** 2
                    value_losses_clipped = (value_pred_clipped - returns) ** 2
                    c_loss = torch.max(value_losses, value_losses_clipped)
                else:
                    c_loss = (values - returns) ** 2
                # bounded loss
                if self.bounds_loss_coef > 0:
                    b_loss = self.model.mean_bounds_loss(res_dict['raw_mus'])
                else:
                    b_loss = torch.zeros_like(a_loss)
                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef \
                    + b_loss * self.bounds_loss_coef

                self.optimizer.zero_grad()
                loss.backward()

                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()
                self.model.project_log_std()
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                entropies.append(entropy)
                b_losses.append(b_loss)

            # Compare the UPDATED policy against the unchanged rollout policy.
            # Evaluate the whole batch so all minibatches use the same policy version.
            data = self.storage.data_dict
            with torch.no_grad():
                updated = self.model({
                    'obs': data['normalized_obses'],
                    'prev_actions': data['actions'],
                })
                av_kls = policy_kl(data['mus'], data['sigmas'],
                                   updated['mus'], updated['sigmas'])
            kls.append(av_kls)
            if av_kls.item() > 2.0 * self.kl_threshold:
                break

        # Adjust once per rollout, not six times while reusing the same data.
        if self.lr_schedule == 'kl':
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
        elif self.lr_schedule == 'cos':
            self.last_lr = self.adjust_learning_rate_cos(mini_ep)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.last_lr

        self.extra_info.update({
            'policy/mean_saturation_fraction': (updated['mus'].abs() > 0.95).float().mean().item(),
            'policy/raw_mean_abs_max': updated['raw_mus'].abs().max().item(),
            'policy/std_min': updated['sigmas'].min().item(),
            'policy/std_max': updated['sigmas'].max().item(),
            'policy/action_clip_fraction': (data['actions'].abs() > 1).float().mean().item(),
            'info/ppo_epochs': len(kls),
            'info/kl_last': av_kls.item(),
        })
        # Use each raw observation once to prepare statistics for the NEXT rollout.
        if self.normalize_input:
            self.running_mean_std.update(data['obses'])

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls
    
    def obs2tensor(self, obs):
        # Map the step result to tensor (dim = 35, the order must stay fixed for saved checkpoints)
        obs_array = np.concatenate((
                    obs["hoop"]["pos3d"], obs["hoop"]["axis3d"], obs["hoop"]["v_lin_3d"],
                    obs["arm"]["joint_pos"], obs["arm"]["joint_vel"],
                    obs["object"]["pos3d"], obs["object"]["v_lin_3d"], obs["object"]["rel_pos3d"],
                    obs["prev_action"],
                    ), axis=1)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(self.device)
        return obs_tensor

    def update_episode_stats(self, infos, meters):
        # The vector env resets finished envs automatically and moves their last info to infos["final_info"]
        if "final_info" not in infos:
            return
        for i in np.nonzero(infos["_final_info"])[0]:
            final_info = infos["final_info"][i]
            for k in EPISODE_STATS:
                meters[k].update(torch.tensor([float(final_info[k])]))
            for outcome_id, name in enumerate(OUTCOME_NAMES[1:], start=1):
                meters['outcome/' + name].update(torch.tensor([float(final_info["outcome"] == outcome_id)]))

    def model_act(self, obs_dict, inference=False):
        processed_obs = (self.running_mean_std(obs_dict['obs'])
                         if self.normalize_input else obs_dict['obs'])
        input_dict = {
            'obs': processed_obs,
        }
        if not inference:
            res_dict = self.model.act(input_dict)
            if self.normalize_value:
                res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        else:
            res_dict = {}
            res_dict['actions'] = self.model.act_inference(input_dict)
        res_dict['normalized_obses'] = processed_obs
        return res_dict

    def play_steps(self):
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # Collect o_t
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('normalized_obses', n, res_dict['normalized_obses'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # Do env step
            # Clamp the actions of the action space (the env scales them, see DcmmCfg.action_scale)
            actions = torch.clamp(res_dict['actions'], -1, 1)
            obs, r, terminates, truncates, infos = self.env.step(actions.cpu().numpy())
            # Map the obs
            self.obs = {'obs': self.obs2tensor(obs)}
            # Map the rewards
            r = torch.tensor(r, dtype=torch.float32).to(self.device)
            rewards = r.unsqueeze(1)
            # Map the dones
            dones = terminates | truncates
            self.dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)
            # Update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = self.reward_scale_value * rewards.clone()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            # print("self.dones: ", self.dones)
            done_indices = self.dones.nonzero(as_tuple=False)
            # print("done_indices: ", done_indices)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])
            assert isinstance(infos, dict), 'Info Should be a Dict'
            self.update_episode_stats(infos, self.episode_stats)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        self.agent_steps = (self.agent_steps + self.batch_size)
        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            # One update from returns, then normalize both tensors in the same units.
            self.value_mean_std.update(returns)
            self.value_mean_std.eval()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns

    def play_test_steps(self):
        for _ in range(self.horizon_length):
            res_dict = self.model_act(self.obs, inference=True)
            # Do env step
            # Clamp the actions of the action space 
            actions = torch.clamp(res_dict['actions'], -1, 1)
            obs, r, terminates, truncates, infos = self.env.step(actions.cpu().numpy())
            # Map the obs
            self.obs = {'obs': self.obs2tensor(obs)}
            # Map the rewards
            r = torch.tensor(r, dtype=torch.float32).to(self.device)
            rewards = r.unsqueeze(1)
            # Map the dones
            dones = terminates | truncates
            self.dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)
            # Update dones and rewards after env step
            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_test_rewards.update(self.current_rewards[done_indices])
            self.episode_test_lengths.update(self.current_lengths[done_indices])
            assert isinstance(infos, dict), 'Info Should be a Dict'
            self.update_episode_stats(infos, self.episode_test_stats)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
        res_dict = self.model_act(self.obs)
        self.agent_steps = (self.agent_steps + self.batch_size)

    def test(self):
        self.set_eval()
        reset_obs, _ = self.env.reset(seed=self.seed)
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.test_steps = self.batch_size

        while self.test_steps < self.max_test_steps:
            self.play_test_steps()
            self.test_steps += self.batch_size
            self.storage.data_dict = None
            mean_rewards = self.episode_test_rewards.get_mean()
            mean_lengths = self.episode_test_lengths.get_mean()
            print("## Sample Length %d ##" % len(self.episode_test_rewards))
            print("mean_rewards: ", mean_rewards)
            print("mean_lengths: ", mean_lengths)
            for k, meter in self.episode_test_stats.items():
                print("mean_{}: {:.3f}".format(k, meter.get_mean()))
            # wandb.log({
            #     'metrics/episode_test_rewards': mean_rewards,
            #     'metrics/episode_test_lengths': mean_lengths,
            # }, step=self.agent_steps)

    def adjust_learning_rate_cos(self, epoch):
        lr = self.init_lr * 0.5 * (
            1. + math.cos(
                math.pi * (self.agent_steps + epoch / self.mini_epochs_num) / self.max_agent_steps))
        return lr


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    """Mean KL(old || new), summed across independent action dimensions."""
    old = torch.distributions.Normal(p0_mu, p0_sigma)
    new = torch.distributions.Normal(p1_mu, p1_sigma)
    return torch.distributions.kl_divergence(old, new).sum(-1).mean()

class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr

class LinearScheduler:
    def __init__(self, start_lr, max_steps=1000000):
        super().__init__()
        self.start_lr = start_lr
        self.min_lr = 1e-06
        self.max_steps = max_steps

    def update(self, steps):
        lr = self.start_lr - (self.start_lr * (steps / float(self.max_steps)))
        return max(self.min_lr, lr)
