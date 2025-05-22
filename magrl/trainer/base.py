import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from magrl.envs.make_env import make_env
from magrl.config.model_config import set_global_random_seed, get_rl_algorithm
from magrl.utils.logger import get_logger

logger = get_logger(__name__)


# trainers
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.epochs = self.cfg.epochs
        self.episodes = self.cfg.episodes
        self.update_frequence = self.cfg.update_step

        self.episode_rewards = [0.0]
        self.loss_infos = []
        self.info = []
        self.train_step = 0
        self.update_times = 0
        self.best_reward = -np.inf
        logger.info("***********  use algo name is ", self.cfg.algo_name)
        logger.info('starting train policy to use rl algorithm')
        logger.info('Env initialling...')
        set_global_random_seed(self.cfg.seed)
        self.wrapper_env = make_env(self.cfg.scenario_name)
        self.agents_rewards = [[0.0] for _ in range(self.wrapper_env.n)]
        logger.info('Env initial successfully!')
        logger.info('MARL algorithm initialing...')
        self.agents = self.get_trainers(self.wrapper_env)
        logger.info('MARL algorithm initial successfully!')
        if not os.path.exists(self.cfg.save_data_dir):
            os.makedirs(self.cfg.save_data_dir)
        if not os.path.exists(self.cfg.save_model_dir):
            os.makedirs(self.cfg.save_model_dir)
        self.tb_logger = SummaryWriter(self.cfg.save_data_dir)
        # load model
        if self.cfg.load_model:
            for i, agent in enumerate(self.agents):
                file_list = os.listdir(f"{self.cfg.load_dir}/{self.cfg.algo_name}")
                model_index = file_list[-1]
                agent.load_model(f"{self.cfg.load_dir}/{self.cfg.algo_name}/{model_index}/agent_{i}")

    def get_trainers(self, env):
        trainers = []
        num_adversaries = min(env.n, self.cfg.num_adversaries)
        # 先red
        alg_trainer, alg_cfg = get_rl_algorithm(self.cfg.adv_algorithm)
        self.cfg.algo_config['red'] = alg_cfg
        for i in range(num_adversaries):  # i=0,1,...,num_adversaries
            trainers.append(alg_trainer(self.cfg, alg_cfg, i, env))
        # 再blue
        alg_trainer, alg_cfg = get_rl_algorithm(self.cfg.good_algorithm)
        self.cfg.algo_config['blue'] = alg_cfg
        for i in range(num_adversaries, env.n):  # i=num_adversaries, num_adversaries+1, ..., env.n
            trainers.append(alg_trainer(self.cfg, alg_cfg, i, env))

        return trainers

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

