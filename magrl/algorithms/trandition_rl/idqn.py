r'''

value-based rl algorithm, eg. dqn

'''
import os

import torch
import torch.nn.functional as F
import random
import numpy as np

from algorithms.Trainer import AgentTrainer
from algorithms.Network.net import MLPCritic
from algorithms.rl_utils.ReplayBuffer import ReplayBufferTransition
from algorithms.rl_utils.batch import Batch

class IDQNAgentTrainer(AgentTrainer):
    name = 'idqn'
    def __init__(self, config, algo_config, agent_index, agents):
        self.cfg = config
        self.algo_cfg = algo_config
        self.agent_index = agent_index
        self.n = config['agent_config']['agent_num']
        self.device = config['device']
        self.hidden_dim = self.algo_cfg['hidden_dim']
        self.activate_fn = self.algo_cfg['activate_fn']
        self.critic_lr = self.algo_cfg['critic_lr']
        self.obs_dim = agents[agent_index].obs_dim
        self.action_dim = agents[agent_index].action_dim
        self.QNetCritic = MLPCritic(self.obs_dim, self.hidden_dim, sum(self.action_dim), self.activate_fn, self.device)
        self.targetQNetCritic = MLPCritic(self.obs_dim, self.hidden_dim, sum(self.action_dim), self.activate_fn, self.device)
        self.QNetCritic_optimizer = torch.optim.Adam(self.QNetCritic.parameters(), lr=self.critic_lr)

        self.replay_buffer = ReplayBufferTransition(self.algo_cfg['buffer_size'])
        self.sample_indexs = None
        self.epsilon = self.algo_cfg['epsilon']
        self.epsilon_min = self.algo_cfg['epsilon_min']
        self.epsilon_decay = self.algo_cfg['epsilon_decay']

    def action(self, obs_n):
        obs = obs_n[self.agent_index]
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs[None], device=self.device)
        if random.random() > self.epsilon:
            q_values = self.QNetCritic(obs)
            action = torch.argmax(q_values, dim=-1)
        else:
            action = torch.randint(0, self.action_dim[0], (1,), device=self.device)

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
            info = {'q_value': 0.0,
                'q_loss': 0.0,
                'epsilon': self.epsilon,
                'batch_reward': 0.0}
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
        agent_data = self.replay_buffer.sample_index(indexs)
        agent_data.to_torch(device=self.device)
        q_values = self.QNetCritic(agent_data.obs).gather(1, agent_data.action)  # Q值
        with torch.no_grad():
            mask = self.algo_cfg['gamma'] * (1.0 - agent_data.done)
            max_next_q_values = self.targetQNetCritic(agent_data.next_obs).max(1)[0].view(-1, 1)  # 下个状态的最大Q值
            q_targets = agent_data.reward + mask * max_next_q_values  # TD目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets.detach()))  # 均方误差损失函数
        self.QNetCritic_optimizer.zero_grad()
        dqn_loss.backward()  # 反向传播更新参数
        self.QNetCritic_optimizer.step()

        self._soft_update(self.algo_cfg['tau'], self.QNetCritic, self.targetQNetCritic)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        info = {'q_value': torch.mean(q_values).item(),
                'q_loss': dqn_loss.item(),
                'epsilon': self.epsilon,
                'batch_reward': torch.mean(agent_data.reward).item()}
        return info

    def train(self):
        self.QNetCritic.train()

    def eval(self):
        self.QNetCritic.eval()

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        save_name = ['QNetCritic', 'QNetCritic_optimizer']
        for _save_name in save_name:
            torch.save(getattr(self, _save_name), f"{path}/{_save_name}.pt")

    def load_model(self, path):
        load_name = ['QNetCritic', 'QNetCritic_optimizer']
        for _load_name in load_name:
            setattr(self, _load_name, torch.load(f"{path}/{_load_name}.pt", map_location=torch.device(self.device)))

    def _soft_update(self, tau, critic_models, target_critic_models):
        # update target critic
        for critic_var, target_critic_var in zip(critic_models.parameters(), target_critic_models.parameters()):
            target_critic_var.data.copy_(critic_var.data * tau + target_critic_var.data * (1.0 - tau))
