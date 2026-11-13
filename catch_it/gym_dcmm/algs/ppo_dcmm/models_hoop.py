import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

        # orthogonal init of weights
        # hidden layers scale np.sqrt(2)
        self.init_weights(self.mlp, [np.sqrt(2)] * len(units))

    def forward(self, x):
        return self.mlp(x)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        separate_value_mlp = kwargs.pop('separate_value_mlp')
        self.separate_value_mlp = separate_value_mlp

        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        self.log_std_min = kwargs.get('log_std_min', -5.0)
        self.log_std_max = kwargs.get('log_std_max', 0.0)
        self.raw_mean_soft_bound = kwargs.get('raw_mean_soft_bound', 2.0)
        mlp_input_shape = input_shape[0]

        out_size = self.units[-1]

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        if self.separate_value_mlp:
            self.value_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

        # policy output layer with scale 0.01
        # value output layer with scale 1
        torch.nn.init.orthogonal_(self.mu.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.value.weight, gain=1.0)
    
    def save_actor(self, actor_mlp_path, actor_head_path):
        """
        Save actor and critic model parameters to files.

        actor_path: Path to save actor model parameters.
        """
        torch.save(self.actor_mlp.state_dict(), actor_mlp_path)
        torch.save(self.mu.state_dict(), actor_head_path)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, raw_mu = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
            'raw_mus': raw_mu,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']

        x = self.actor_mlp(obs)
        raw_mu = self.mu(x)
        if self.separate_value_mlp:
            x = self.value_mlp(obs)
        value = self.value(x)

        logstd = self.sigma.clamp(self.log_std_min, self.log_std_max)
        # Normalize to (-1,1)
        mu = torch.tanh(raw_mu)
        return mu, logstd.expand_as(mu), value, raw_mu

    @torch.no_grad()
    def project_log_std(self):
        # Keep the parameter itself in range so a clamp cannot trap it outside.
        self.sigma.clamp_(self.log_std_min, self.log_std_max)

    def mean_bounds_loss(self, raw_mu):
        # Penalize BEFORE tanh: this still has a gradient when tanh saturates.
        return torch.clamp_min(raw_mu.abs() - self.raw_mean_soft_bound, 0.0).square().sum(-1)

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value, raw_mu = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': prev_neglogp,
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'raw_mus': raw_mu,
        }
        return result
