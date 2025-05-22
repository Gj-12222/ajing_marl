r'''

masac


'''

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from algorithms.Trainer import AgentTrainer
from algorithms.Network.net import MLPCritic, SoftDiscreteActor
from algorithms.rl_utils.ReplayBuffer import ReplayBufferTransition
from algorithms.rl_utils.batch import Batch

class MASACAgentTrainer(AgentTrainer):
    name = 'masac'
    def __init__(self, config, algo_config, agent_index, env):
        super().__init__()
        self.cfg = config
        self.algo_cfg = algo_config
        self.agent_index = agent_index
        self.n = config.total_num_agent
        self.device = config.device
        self.hidden_dim = self.algo_cfg['hidden_dim']

        self.actor_lr =self.algo_cfg['actor_lr']
        self.critic_lr = self.algo_cfg['critic_lr']
        self.alpha_lr = self.algo_cfg['alpha_lr']

        self.obs_dim = env.observation_space[agent_index].shape[0]
        self.action_dim = env.action_space[agent_index].shape[0]

        self.PolicyActor = SoftDiscreteActor(self.obs_dim, self.hidden_dim, self.action_dim,  distribution_fn=self.algo_cfg['distribution_fn'], device=self.device)
        self.PolicyActor_optimizer = torch.optim.Adam(self.PolicyActor.parameters(), lr=self.actor_lr)

        self.q_input_dim = sum([obs_shape.shape[0] + action_shape.shape[0] for obs_shape, action_shape in zip(env.observation_space, env.action_space)])
        self.Q1NetCritic = MLPCritic(self.q_input_dim, self.hidden_dim, 1, device=self.device)
        self.targetQ1NetCritic = MLPCritic(self.q_input_dim, self.hidden_dim, 1, device=self.device)
        self.Q1NetCritic_optimizer = torch.optim.Adam(self.Q1NetCritic.parameters(), lr=self.critic_lr)
        self.Q2NetCritic = MLPCritic(self.q_input_dim, self.hidden_dim, 1, device=self.device)
        self.targetQ2NetCritic = MLPCritic(self.q_input_dim, self.hidden_dim, 1, device=self.device)
        self.Q2NetCritic_optimizer = torch.optim.Adam(self.Q2NetCritic.parameters(), lr=self.critic_lr)

        # trainable parameter
        self.log_alpha = torch.tensor((-np.log(self.action_dim),),
                                      dtype=torch.float32,
                                      requires_grad=True,
                                      device=self.device)
        self.log_alpha_optim = torch.optim.Adam((self.log_alpha,), lr=self.alpha_lr)
        self.target_entropy = np.prod(self.action_dim)

        self.replay_buffer = ReplayBufferTransition(self.algo_cfg['buffer_size'])
        self.sample_indexs = None

    def action(self, obs_n):
        obs = obs_n[self.agent_index]
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        action_dist = self.PolicyActor(obs)
        action = action_dist.sample().squeeze(dim=0)

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
                    'p_loss':0.0,
                    'alpha_loss': 0.0,
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
        obs_n = torch.cat(obs_n, dim=-1)
        next_action_n = torch.cat([agent.PolicyActor(next_obs).sample() for agent, next_obs in zip(agents, next_obs_n)],
                                  dim=-1)
        next_obs_n = torch.cat(next_obs_n, dim=-1)
        agent_data = self.replay_buffer.sample_index(indexs)
        agent_data.to_torch(device=self.device)

        q1_value = self.Q1NetCritic(torch.cat([obs_n, torch.cat(action_n, dim=-1)], dim=-1))  # Q1值
        q2_value = self.Q2NetCritic(torch.cat([obs_n, torch.cat(action_n, dim=-1)], dim=-1))  # Q2值
        with torch.no_grad():
            next_q1_target_value = self.targetQ1NetCritic(torch.cat([next_obs_n, next_action_n], dim=-1))  # 下个状态的Q1值
            next_q2_target_value = self.targetQ2NetCritic(torch.cat([next_obs_n, next_action_n], dim=-1))  # 下个状态的Q2值

            mask = self.algo_cfg['gamma'] * (1.0 - agent_data.done)
            next_q_value = torch.min(next_q1_target_value, next_q2_target_value)
            next_act_dist = self.PolicyActor(agent_data.next_obs)
            next_act_log_prob = next_act_dist.log_prob(next_act_dist.rsample().argmax(dim=-1))
            q_targets = agent_data.reward + mask * (next_q_value - next_act_log_prob * self.log_alpha.exp().detach()) # TD目标

        q1_loss = torch.mean(F.mse_loss(q1_value, q_targets.detach()))  # 均方误差损失函数
        q2_loss = torch.mean(F.mse_loss(q2_value, q_targets.detach()))  # 均方误差损失函数

        self.Q1NetCritic_optimizer.zero_grad()
        q1_loss.backward()  # 反向传播更新参数
        self.Q1NetCritic_optimizer.step()

        self.Q2NetCritic_optimizer.zero_grad()
        q2_loss.backward()  # 反向传播更新参数
        self.Q2NetCritic_optimizer.step()

        action_dist = self.PolicyActor(agent_data.obs)
        new_action = action_dist.rsample()
        action_n[self.agent_index] = new_action
        log_act_prob = action_dist.log_prob(new_action.argmax(dim=-1))

        # alpha
        alpha_loss = - (self.log_alpha * (log_act_prob- self.target_entropy).detach()).mean()

        self.log_alpha_optim.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optim.step()
        with torch.no_grad():
            self.log_alpha[:] = self.log_alpha.clamp(-20, 2)

        # actor
        action_n = torch.cat(action_n, dim=-1)
        q_value = torch.min(self.Q1NetCritic(torch.cat([obs_n, action_n], dim=-1)),
                            self.Q2NetCritic(torch.cat([obs_n, action_n], dim=-1)))
        policy_loss = - torch.mean(q_value - self.log_alpha.detach().exp() * log_act_prob)

        self.PolicyActor_optimizer.zero_grad()
        policy_loss.backward()
        self.PolicyActor_optimizer.step()


        self._soft_update(self.algo_cfg['tau'], self.Q1NetCritic, self.targetQ1NetCritic)
        self._soft_update(self.algo_cfg['tau'], self.Q2NetCritic, self.targetQ2NetCritic)


        info = {'q_value': torch.mean(q_value).item(),
                'q_loss': (q1_loss + q2_loss).item() / 2,
                'p_loss': policy_loss.item(),
                'alpha_loss': alpha_loss.item(),
                'batch_reward': torch.mean(agent_data.reward).item()}
        return info

    def train(self):
        self.Q1NetCritic.train()
        self.Q2NetCritic.train()
        self.PolicyActor.train()

    def eval(self):
        self.Q1NetCritic.eval()
        self.Q2NetCritic.eval()
        self.PolicyActor.eval()

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        save_name = ['PolicyActor', 'PolicyActor_optimizer', 'Q1NetCritic', 'Q1NetCritic_optimizer', 'Q2NetCritic', 'Q2NetCritic_optimizer']
        for _save_name in save_name:
            torch.save(getattr(self, _save_name), f"{path}/{_save_name}.pt")

    def load_model(self, path):
        load_name = ['PolicyActor', 'PolicyActor_optimizer', 'Q1NetCritic', 'Q1NetCritic_optimizer', 'Q2NetCritic', 'Q2NetCritic_optimizer']
        for _load_name in load_name:
            setattr(self, _load_name, torch.load(f"{path}/{_load_name}.pt", map_location=torch.device(self.device)))

    def _soft_update(self, tau, critic_models, target_critic_models):
        # update target critic
        for critic_var, target_critic_var in zip(critic_models.parameters(), target_critic_models.parameters()):
            target_critic_var.data.copy_(critic_var.data * tau + target_critic_var.data * (1.0 - tau))



