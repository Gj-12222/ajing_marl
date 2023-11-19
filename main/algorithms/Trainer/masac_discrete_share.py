r'''

masac discrete


'''

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from algorithms.Trainer import AgentTrainer
from algorithms.Network.net import MLPCritic, DiscreteActor
from algorithms.rl_utils.ReplayBuffer import ReplayBufferTransition, SampleIndexs
from algorithms.rl_utils.batch import Batch

class MASACShareDiscreteAgentTrainer(AgentTrainer):
    name = 'masac_discrete_share'
    def __init__(self, config, algo_config, agent_index, agents):
        super().__init__()
        self.cfg = config
        self.algo_cfg = algo_config
        self.agent_index = agent_index
        self.n = config['agent_config']['agent_num']
        self.device = config['device']
        self.hidden_dim = self.algo_cfg['hidden_dim']
        self.activate_fn = self.algo_cfg['activate_fn']
        self.actor_lr =self.algo_cfg['actor_lr']
        self.critic_lr = self.algo_cfg['critic_lr']
        self.alpha_lr = self.algo_cfg['alpha_lr']
        self.obs_dim = agents[0].obs_dim
        self.action_dim = agents[0].action_dim

        self.PolicyActor = DiscreteActor(self.obs_dim, self.hidden_dim, sum(self.action_dim),  device=self.device)
        self.PolicyActor_optimizer = torch.optim.Adam(self.PolicyActor.parameters(), lr=self.actor_lr)

        self.q_input_dim = sum([agent.obs_dim for agent in agents])
        self.Q1NetCritic = MLPCritic(self.q_input_dim, self.hidden_dim, sum(self.action_dim), device=self.device)
        self.targetQ1NetCritic = MLPCritic(self.q_input_dim, self.hidden_dim, sum(self.action_dim), device=self.device)
        self.Q1NetCritic_optimizer = torch.optim.Adam(self.Q1NetCritic.parameters(), lr=self.critic_lr)
        self.Q2NetCritic = MLPCritic(self.q_input_dim, self.hidden_dim, sum(self.action_dim), device=self.device)
        self.targetQ2NetCritic = MLPCritic(self.q_input_dim, self.hidden_dim, sum(self.action_dim), device=self.device)
        self.Q2NetCritic_optimizer = torch.optim.Adam(self.Q2NetCritic.parameters(), lr=self.critic_lr)

        # trainable parameter
        self.log_alpha = torch.tensor((-np.log(sum(self.action_dim)),),
                                      dtype=torch.float32,
                                      requires_grad=True,
                                      device=self.device)
        self.log_alpha_optim = torch.optim.Adam((self.log_alpha,), lr=self.alpha_lr)
        self.target_entropy = np.prod(self.action_dim)

        self.replay_buffer = ReplayBufferTransition(self.algo_cfg['buffer_size'])
        self.sample_indexs = SampleIndexs

    def action(self, obs_n):
        obs = obs_n[self.agent_index]
        if len(obs.shape) == 1:
            obs = torch.tensor(obs[None], device=self.device)
        action_dist = self.PolicyActor(obs)
        action = action_dist.sample().squeeze(dim=0)
        return action

    def preUpdate(self):
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

    def update(self, agents, train_step, agent_index=None):
        # batch_size * 10
        if len(self.replay_buffer) < self.algo_cfg['batch_size']:
            info = {'q_value': 0.0,
                    'q_loss': 0.0,
                    'p_loss':0.0,
                    'alpha_loss': 0.0,
                    'batch_reward': 0.0}
            return info

        indexs = self.sample_indexs.sample_indexs
        obs_n, action_n, next_obs_n = [], [], []
        for i in range(self.n):
            buffer_data = agents[i].replay_buffer.sample_index(indexs)
            buffer_data.to_torch(device=self.device)
            obs_n.append(buffer_data.obs)
            action_n.append(buffer_data.action)
            next_obs_n.append(buffer_data.next_obs)
        obs_n = torch.cat(obs_n, dim=-1)
        action_n = torch.cat(action_n, dim=-1)
        next_obs_n =torch.cat(next_obs_n, dim=-1)
        agent_data = self.replay_buffer.sample_index(indexs)
        agent_data.to_torch(device=self.device)

        q1_values = self.Q1NetCritic(obs_n)  # Q1值
        q2_values = self.Q2NetCritic(obs_n)  # Q2值

        old_action = agent_data.action.unsqueeze(dim=-1)
        # Ep[r + gamma * Ep[Q(st+1, a+1) - logp] - Ep[Q(st,at) - logp] ]
        # Q(st,at)  - r + gamma * Ep[Q(st+1, a+1) - logp] = 0
        # action = 0, 1  通常是MC近似Ep，128个样本取均值 近似Ep act_prob = Ep
        with torch.no_grad():
            next_q1_target_values = self.targetQ1NetCritic(next_obs_n)  # 下个状态的Q1值
            next_q2_target_values = self.targetQ2NetCritic(next_obs_n)  # 下个状态的Q2值
            next_act_dist = self.PolicyActor(agent_data.next_obs)
            next_act_prob = next_act_dist.probs

            next_log_act_prob = torch.log(next_act_prob + 1e-8)

            next_q_target_value = (next_act_prob * (torch.min(next_q1_target_values, next_q2_target_values) -
                                                    self.log_alpha.detach().exp() * next_log_act_prob.squeeze(dim=-1))
                                   ).sum(dim=-1).unsqueeze(dim=-1)

            mask = self.algo_cfg['gamma'] * (1.0 - agent_data.done)
            q_targets = agent_data.reward + mask * next_q_target_value  # TD目标

        q1_loss = torch.mean(F.mse_loss(q1_values.gather(-1, old_action), q_targets.detach()))  # 均方误差损失函数
        q2_loss = torch.mean(F.mse_loss(q2_values.gather(-1, old_action), q_targets.detach()))  # 均方误差损失函数

        self.Q1NetCritic_optimizer.zero_grad()
        q1_loss.backward()  # 反向传播更新参数
        self.Q1NetCritic_optimizer.step()

        self.Q2NetCritic_optimizer.zero_grad()
        q2_loss.backward()  # 反向传播更新参数
        self.Q2NetCritic_optimizer.step()

        # policy
        action_dist = self.PolicyActor(agent_data.obs)
        act_prob = action_dist.probs
        log_act_prob = F.log_softmax(action_dist.logits, dim=-1)
        # alpha 既要保证max return  max entropy  log_act_prob ==> 1
        alpha_loss = (- act_prob.detach() * (self.log_alpha * (log_act_prob - self.target_entropy).detach())).sum(dim=-1).mean()
        self.log_alpha_optim.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optim.step()

        # policy max Q(St, at)
        with torch.no_grad():
            self.log_alpha[:] = self.log_alpha.clamp(-20, 2)

            q_value = torch.min(self.Q1NetCritic(obs_n), self.Q2NetCritic(obs_n))

        policy_loss = (act_prob * (self.log_alpha.detach().exp() * log_act_prob - q_value.detach())).sum(dim=-1).mean()

        self.PolicyActor_optimizer.zero_grad()
        policy_loss.backward()
        self.PolicyActor_optimizer.step()

        # target q1, q2
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

    @property
    def model(self):
        return [self.PolicyActor, self.Q1NetCritic, self.targetQ1NetCritic, self.Q2NetCritic, self.targetQ2NetCritic, self.log_alpha,
                self.PolicyActor_optimizer, self.Q1NetCritic_optimizer, self.Q2NetCritic_optimizer, self.log_alpha_optim]

    @model.setter
    def model(self, NetWorkList):
        print(self.agent_index, 'is setter Network!')
        self.PolicyActor, self.Q1NetCritic, self.targetQ1NetCritic, self.Q2NetCritic, self.targetQ2NetCritic, self.log_alpha, \
        self.PolicyActor_optimizer, self.Q1NetCritic_optimizer, self.Q2NetCritic_optimizer, self.log_alpha_optim = NetWorkList

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        save_name = ['log_alpha', 'log_alpha_optim', 'PolicyActor', 'PolicyActor_optimizer', 'Q1NetCritic', 'Q1NetCritic_optimizer', 'Q2NetCritic', 'Q2NetCritic_optimizer']
        for _save_name in save_name:
            torch.save(getattr(self, _save_name), f"{path}/{_save_name}.pt")

    def load_model(self, path):
        load_name = ['log_alpha', 'log_alpha_optim', 'PolicyActor', 'PolicyActor_optimizer', 'Q1NetCritic', 'Q1NetCritic_optimizer', 'Q2NetCritic', 'Q2NetCritic_optimizer']
        for _load_name in load_name:
            setattr(self, _load_name, torch.load(f"{path}/{_load_name}.pt", map_location=torch.device(self.device)))

    def _soft_update(self, tau, critic_models, target_critic_models):
        # update target critic
        # q target_q tau=0.001   target_critic_var.data = 0.999 * target_critic_var.data  + 0.001 critic_var.data
        for critic_var, target_critic_var in zip(critic_models.parameters(), target_critic_models.parameters()):
            target_critic_var.data.copy_(critic_var.data * tau + target_critic_var.data * (1.0 - tau))



