r'''

dqn + share network

'''

import os

import torch
import torch.nn.functional as F
import random
import numpy as np

from Algorithms.Trainer import AgentTrainer
from Algorithms.Network.net import MLPCritic
from Algorithms.rl_utils.ReplayBuffer import ReplayBufferTransition, SampleIndexs
from Algorithms.rl_utils.batch import Batch


class DQNShareAgentTrainer(AgentTrainer):
    name = 'dqn_share'

    def __init__(self, config, algo_config, agent_index, agents):
        super().__init__()
        self.cfg = config
        self.algo_cfg = algo_config
        self.n = self.cfg['agent_config']['agent_num']
        self.agent_index = agent_index
        self.device = config['device']
        self.hidden_dim = self.algo_cfg['hidden_dim']
        self.activate_fn = self.algo_cfg['activate_fn']
        self.critic_lr = self.algo_cfg['critic_lr']
        self.soft_update_freq = self.algo_cfg['soft_update_freq']
        self.obs_dim = agents[0].obs_dim
        self.action_dim = agents[0].action_dim
        self.QNetCritic = MLPCritic(self.obs_dim, self.hidden_dim, sum(self.action_dim), self.activate_fn, self.device)
        self.targetQNetCritic = MLPCritic(self.obs_dim, self.hidden_dim, sum(self.action_dim), self.activate_fn,
                                          self.device)
        self.QNetCritic_optimizer = torch.optim.Adam(self.QNetCritic.parameters(), lr=self.critic_lr)

        self.replay_buffer = ReplayBufferTransition(self.algo_cfg['buffer_size'])
        self.sample_indexs = SampleIndexs  # 类变量可共享，实例化变量无法共享
        self.epsilon = self.algo_cfg['epsilon']
        self.epsilon_min = self.algo_cfg['epsilon_min']
        self.epsilon_decay = self.algo_cfg['epsilon_decay']

    def action(self, obs_n):
        obs = obs_n[self.agent_index]
        if len(obs.shape) == 1:
            obs = obs[None]
        obs = torch.tensor(obs, device=self.device)
        if random.random() > self.epsilon:
            q_values = self.QNetCritic(obs)
            action = torch.argmax(q_values, dim=-1)
        else:
            action = torch.randint(0, self.action_dim[0], (1,), device=self.device)

        return action

    def preUpdate(self):
        # self.sample_indexs.sample_indexs = None
        self.sample_indexs.sample_indexs = self.replay_buffer.make_index(self.algo_cfg['batch_size'])

    def save_replay_buffer(self, obs, action, reward, next_obs, done, terminal):
        transition = Batch()
        transition.obs = obs
        transition.action = action
        transition.next_obs = next_obs
        transition.reward = np.array(reward, dtype=np.float32)
        transition.done = np.array([done or terminal], dtype=np.float32)
        transition.to_numpy()
        self.replay_buffer.add(transition)

    def update(self, trainers, train_step, agent_index=None):
        if len(self.replay_buffer) < self.algo_cfg['batch_size']:  # 256 * 11
            info = {'q_value': 0.0,
                    'q_loss': 0.0,
                    'epsilon': self.epsilon,
                    'batch_reward': 0.0}
            return info

        indexs = self.sample_indexs.sample_indexs

        all_obs_n, all_action_n, all_next_obs_n, all_reward_n, all_done_n = [], [], [], [], []
        for i in range(self.n):
            agent_data = trainers[i].replay_buffer.sample_index(indexs)
            agent_data.to_torch(device=self.device)
            all_obs_n.append(agent_data.obs[indexs])
            all_action_n.append(agent_data.action[indexs])
            all_reward_n.append(agent_data.reward[indexs])
            all_next_obs_n.append(agent_data.next_obs[indexs])
            all_done_n.append(agent_data.done[indexs])


        # n, bs, x -> n*bs, x
        all_done = torch.cat(all_done_n, dim=0)
        all_obs = torch.cat(all_obs_n, dim=0)
        all_action = torch.cat(all_action_n, dim=0)
        all_next_obs = torch.cat(all_next_obs_n, dim=0)
        all_reward = torch.cat(all_reward_n, dim=0)

        mask = self.algo_cfg['gamma'] * (1.0 - all_done)
        q_values = self.QNetCritic(all_obs).gather(1, all_action)  # Q值
        max_next_q_values = self.targetQNetCritic(all_next_obs).max(1)[0].view(-1, 1)  # 下个状态的最大Q值
        q_targets = all_reward + mask * max_next_q_values  # TD目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.QNetCritic_optimizer.zero_grad()
        dqn_loss.backward()  # 反向传播更新参数
        self.QNetCritic_optimizer.step()

        # if train_step % self.soft_update_freq == 0:
        #     self._soft_update(self.algo_cfg['tau'], self.QNetCritic, self.targetQNetCritic)

        self._soft_update(self.algo_cfg['tau'], self.QNetCritic, self.targetQNetCritic)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


        info = [{'q_value': torch.mean(q_values[index * indexs: (index+1) * indexs, :]).item(),
                'q_loss': dqn_loss.item(),
                'epsilon': self.epsilon,
                'batch_reward': torch.mean(all_reward_n[index]).item()} for index in range(self.n)]

        return info

    def train(self):
        self.QNetCritic.train()

    def eval(self):
        self.QNetCritic.eval()

    @property
    def model(self):
        return [self.QNetCritic, self.targetQNetCritic, self.QNetCritic_optimizer]

    @model.setter
    def model(self, NetWorkList):
        print(self.agent_index, 'is setter Network!')
        self.QNetCritic, self.targetQNetCritic, self.QNetCritic_optimizer = NetWorkList

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
