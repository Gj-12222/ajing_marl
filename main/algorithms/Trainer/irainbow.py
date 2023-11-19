r'''

independent learning + rainbow

'''

import os
import torch
import numpy as np

from algorithms.Trainer import AgentTrainer
from algorithms.Network.net import RainbowCritic
from algorithms.rl_utils.ReplayBuffer import ReplayBufferTransition
from algorithms.rl_utils.batch import Batch


class RainbowAgentTrainer(AgentTrainer):
    # rainbow = d3qn + pre + noise net + distribution net + n_step td
    name = 'rainbow'

    def __init__(self, config, algo_config, agent_index, agents):
        self.cfg = config
        self.algo_cfg = algo_config
        self.agent_index = agent_index
        self.n = config['agent_config']['agent_num']
        self.device = config['device']
        self.hidden_dim = algo_config['hidden_dim']
        self.activate_fn = algo_config['activate_fn']
        self.n_atom = algo_config['atom']
        self.noise_layer = algo_config['noise_layer']
        self.q_lr = algo_config['q_lr']
        self.adv_lr = algo_config['adv_lr']
        self.obs_dim = agents[agent_index].obs_dim
        self.action_dim = agents[agent_index].action_dim
        self.v_min = algo_config['v_min']
        self.v_max = algo_config['v_max']
        self.v_quantization = torch.linspace(self.v_min, self.v_max, self.n_atom).to(self.device)
        self.detla_v = (self.v_max - self.v_min) / (self.n_atom - 1)
        # create network
        self.RainbowNetCritic = RainbowCritic(self.obs_dim,
                                              self.hidden_dim,
                                              self.n_atom,
                                              self.v_min,
                                              self.v_max,
                                              self.activate_fn,
                                              self.device)

        self.targetRainbowNetCritic = RainbowCritic(self.obs_dim,
                                                    self.hidden_dim,
                                                    self.n_atom,
                                                    self.v_min,
                                                    self.v_max,
                                                    self.activate_fn,
                                                    self.device)

        self.RainbowNetCritic_optimizer = torch.optim.Adam(self.RainbowNetCritic.parameters(), lr=self.q_lr)

        self.replay_buffer = ReplayBufferTransition(self.algo_cfg['buffer_size'])
        self.sample_indexs = None

    def action(self, obs_n):
        obs = obs_n[self.agent_index]
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs[None], device=self.device)

        q_dist = self.RainbowNetCritic(obs)['logit']
        action = torch.argmax(q_dist, dim=-1)

        return action

    def preUpdate(self):
        self.sample_indexs = None

    def save_replay_buffer(self, obs, action, reward, next_obs, done, terminal):
        transition = Batch()
        transition.obs = obs
        transition.action = action
        transition.next_obs = next_obs
        transition.reward = np.array(reward, dtype=np.float32)
        transition.done = np.array([done or terminal], dtype=np.float32)
        transition.to_numpy()
        self.replay_buffer.add(transition)

    def update(self, agents, train_step, agent_index=None):
        if len(self.replay_buffer) < self.algo_cfg['batch_size']:  # 256
            info = {'q_loss': 0.0,
                    'priority': 0.0}
            return info

        self.sample_indexs = self.replay_buffer.make_index(self.algo_cfg['batch_size'])
        indexs = self.sample_indexs
        obs_n, action_n, next_obs_n = [], [], []
        for i in range(self.n):
            buffer_data = agents[i].replay_buffer.sample_index(indexs)
            buffer_data.to_torch(device=self.device)
            obs_n.append(buffer_data.obs)
            action_n.append(buffer_data.action)
            next_obs_n.append(buffer_data.next_obs)
        obs_n = torch.cat(obs_n, dim=-1)
        action_n = torch.cat(action_n, dim=-1)
        next_obs_n = torch.cat(next_obs_n, dim=-1)
        agent_data = self.replay_buffer.sample_index(indexs)
        agent_data.to_torch(device=self.device)
        mask = self.algo_cfg['gamma'] * (1.0 - agent_data.done)

        q_dist = self.RainbowNetCritic(agent_data.obs)['distribution']
        with torch.no_grad():
        # double dqn
            next_action = self.RainbowNetCritic(agent_data.next_obs)['logit'].argmax(dim=-1)
            next_q_dist = self.targetRainbowNetCritic(agent_data.next_obs)['distribution']
        rainbow_loss, td_error_pre_sample = self.calculation_loss(q_dist,
                                                                  next_q_dist.detach(),
                                                                  next_action.detach(),
                                                                  agent_data, mask)
        self.RainbowNetCritic_optimizer.zero_grad()
        rainbow_loss.backward()  # 反向传播更新参数
        self.RainbowNetCritic_optimizer.step()

        self._soft_update(self.algo_cfg['tau'], self.RainbowNetCritic, self.targetRainbowNetCritic)


        info = {'rainbow_loss': rainbow_loss.item(),
                'priority': td_error_pre_sample}
        return info

    def train(self):
        self.RainbowNetCritic.train()

    def eval(self):
        self.RainbowNetCritic.eval()

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        save_name = ['RainbowNetCritic', 'targetRainbowNetCritic', 'RainbowNetCritic_optimizer']
        for _save_name in save_name:
            torch.save(getattr(self, _save_name), f"{path}/{_save_name}.pt")

    def load_model(self, path):
        load_name = ['RainbowNetCritic', 'targetRainbowNetCritic', 'RainbowNetCritic_optimizer']
        for _load_name in load_name:
            setattr(self, _load_name,
                    torch.load(f"{path}/{_load_name}.pt", map_location=torch.device(self.device)))

    def calculation_loss(self, q_dist, next_q_dist, next_action, agent_data, mask, weight=None):
        reward = agent_data.reward  # [batch, 1]
        action = agent_data.action # [batch, 1]
        reward = reward.unsqueeze(dim=-1).repeat(1, action.shape[1])  # [batch, 1, 1]
        mask = mask.unsqueeze(dim=-1).repeat(1, action.shape[1])  # [batch, 1, 1]
        batch_size = action.shape[0] * action.shape[1]
        batch_range = torch.arange(action.shape[0] * action.shape[1])
        n_atom = q_dist.shape[2]  # q_dist.shape = [Batch, action_dim, n_atom]
        q_dist = q_dist.reshape(action.shape[0] * action.shape[1], n_atom, -1)  # [batch, n_atom, action_dim]
        reward = reward.reshape(action.shape[0] * action.shape[1])
        mask = mask.reshape(action.shape[0] * action.shape[1])
        next_q_dist = next_q_dist.reshape(action.shape[0] * action.shape[1], n_atom, -1)

        next_action = next_action.reshape(action.shape[0] * action.shape[1])
        next_q_dist = next_q_dist[batch_range, next_action].detach()
        action = action.reshape(action.shape[0] * action.shape[1])
        target_n_atom = reward + mask * self.v_quantization
        target_n_atom = target_n_atom.clamp(min=self.v_min, max=self.v_max)
        b = (target_n_atom - self.v_min) / self.detla_v
        l = b.floor().long()
        u = b.ceil().long()

        l[(u > 0) * (l ==u)] -= 1
        u[(l < (self.n_atom - 1)) * (l == u)] += 1

        zero_dist = torch.zeros_like(next_q_dist).to(self.device)
        offset = torch.linspace(0, (batch_size - 1) * self.n_atom, batch_size).unsqueeze(dim=1).expand(batch_size, self.n_atom).long().to(self.device)

        zero_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_q_dist * (u.float() - b)).view(-1))
        zero_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_q_dist * (b - l.float())).view(-1))

        assert (q_dist[batch_range, action] > 0.0).all(), ('q_dist action', q_dist[batch_range, action], 'dist:', q_dist)
        log_p = torch.log(q_dist[batch_range, action])

        td_error_pre_sample = -(log_p * zero_dist).sum(dim=-1)
        if weight is None:
            weight = torch.ones_like(reward).to(self.device)
        loss = -(log_p * zero_dist * weight).sum(dim=-1).mean()

        return loss, td_error_pre_sample

    def _soft_update(self, tau, critic_models, target_critic_models):
        # update target critic
        for critic_var, target_critic_var in zip(critic_models.parameters(), target_critic_models.parameters()):
            target_critic_var.data.copy_(critic_var.data * tau + target_critic_var.data * (1.0 - tau))
