r'''

qmix

'''

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from algorithms.Trainer import AgentTrainer
from algorithms.Network.net import RainbowCritic
from algorithms.rl_utils.ReplayBuffer import ReplayBufferTransition
from algorithms.rl_utils.batch import Batch

class QmixAgentTrainer(AgentTrainer):
    name = 'qmix'

    def __init__(self, config, algo_config, agent_index, agents):
        pass

    def action(self, obs):
        pass

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
        pass

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
