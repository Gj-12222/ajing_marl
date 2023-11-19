from .. import BaseAgent
import torch
import random
import torch.nn.functional as F
from algorithms.Trainer.baseline.utils import copy_model_params
from algorithms.Trainer.baseline.replay_buffer import ReplayBuffer


class BaseDRL(BaseAgent):
    def __init__(self, config, algo_config, idx, agents):
        super(BaseDRL, self).__init__(config, algo_config, idx, agents)
        self.network_local = _Network()
        self.network_target = _Network()
        self.network_optim = None
        self.replay_buffer = ReplayBuffer(self.cur_agent['buffer_size'],
                                          self.cur_agent['batch_size'],
                                          self.obs_shape,
                                          self.config['device'])
        self.tau = self.cur_agent['tau']
        self.gamma = self.cur_agent['gamma']
        copy_model_params(source_model=self.network_local, target_model=self.network_target)

    def reset(self):
        self.current_phase = 0
        self.replay_buffer.reset()

    def action(self, n_obs):
        obs = n_obs[self.idx]
        if not isinstance(obs[0], torch.Tensor):
            obs = [torch.tensor(feature, dtype=torch.float32, device=self.config['device'])for feature in obs]
        assert obs[0].shape[0] == 1, 'should be mini-batch with size 1'
        self.network_local.eval()
        with torch.no_grad():
            b_q_value = self.network_local(obs)
            action = torch.argmax(b_q_value, dim=1).cpu().item()
            if self.on_training and random.random() < self.cur_agent['epsilon']:  # explore
                action = random.randint(0, self.num_phase - 1)
        self.network_local.train()
        self.current_phase = action
        return self.current_phase

    def save_replay_buffer(self, obs, action, reward, next_obs, done, terminal):
        self.replay_buffer.store_experience(obs, action, reward, next_obs, done or terminal)

    def _time_to_learn(self):
        return self.replay_buffer.current_size >= self.cur_agent['batch_size']

    def update(self, agents, agent_index=None, train_step=None):
        if self._time_to_learn():
            obs, act, rew, next_obs, done = self.replay_buffer.sample_experience()
            critic_loss = self._compute_critic_loss(obs, act, rew, next_obs, done)
            self.network_optim.zero_grad()
            critic_loss.backward()
            self.network_optim.step()
            for to_model, from_model in zip(self.network_target.parameters(), self.network_local.parameters()):
                to_model.data.copy_(self.tau * from_model.data + (1.0 - self.tau) * to_model.data)
            return critic_loss
        else:
            return 0

    def _compute_critic_loss(self, obs, act, rew, next_obs, done):
        with torch.no_grad():
            q_target_next = self.network_target(next_obs)
            q_target = rew + self.gamma * torch.max(q_target_next, dim=1, keepdim=True)[0] * (~done)
        q_expected = self.network_local(obs).gather(1, act.long())
        critic_loss = F.mse_loss(q_expected, q_target)
        return critic_loss


class _Network(torch.nn.Module):
    def __init__(self):
        super(_Network, self).__init__()

    def forward(self, obs):
        raise NotImplementedError('_Network should be implemented in subclass')
