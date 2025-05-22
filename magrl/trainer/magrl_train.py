r"""

pytorch trainer.py


"""

import os

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

from magrl.trainer.base import Trainer
from magrl.utils.logger import get_logger

logger = get_logger(__name__)


# trainers
class MAGRLTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def start_train(self):
        logger.info('*********** start training ***********')
        for agent in self.agents:
            agent.train()

        for epoch in range(self.epochs):
            for episode in range(self.episodes):
                obs_n = self.wrapper_env.reset()
                episode_step = 0
                start_time = time.time()
                while True:
                    action_n = [agent.action(obs_n) for agent, obs in zip(self.agents, obs_n)]
                    next_obs_n, reward_n, done_n, terminal_n, info_n = self.wrapper_env.step(action_n)
                    for i, agent in enumerate(self.agents):
                        agent.save_replay_buffer(obs_n[i],
                                                 action_n[i],
                                                 reward_n[i],
                                                 next_obs_n[i],
                                                 done_n[i],
                                                 terminal_n[i],
                                                 )
                    episode_step += 1
                    obs_n = next_obs_n
                    # can use tensorboard log
                    for i, reward in enumerate(reward_n):
                        self.episode_rewards[-1] += reward
                        self.agents_rewards[i][-1] += reward
                    self.train_step += 1

                    if self.cfg.learn_type == 'off_policy':
                        # update agent model
                        if self.train_step % self.update_frequence == 0:
                            # TODO: 每个agent可以有不同算法
                            loss_info = []
                            if self.cfg.network_structure == 'share':
                                self.agents[0].preUpdate()
                                loss = self.agents[0].update(self.agents, self.train_step)
                                loss_info.extend(loss)
                            else:
                                for agent in self.agents:
                                    agent.preUpdate()
                                for agent_index, agent in enumerate(self.agents):
                                    loss = agent.update(self.agents, self.train_step, agent_index=agent_index)
                                    loss_info.append(loss)

                            self.loss_infos.append(loss_info)
                            for i, _loss in enumerate(loss_info):
                                for k, v in _loss.items():
                                    self.tb_logger.add_scalar('policy_training/agent_' + str(i) + '_' + k, v,
                                                              self.update_times)

                    if all(terminal_n) or episode_step > self.cfg.max_episode_len:
                        # save_best_model
                        if self.episode_rewards[-1] > self.best_reward:
                            self.best_reward = self.episode_rewards[-1]
                            for i, agent in enumerate(self.agents):
                                agent.save_model(f"{self.cfg.save_model_dir}/{self.cfg.algo_name}/{episode}/agent_{i}")
                            logger.info(f"**** best reward={self.best_reward}, save agents model successfully!")

                        for i, agent_reward in enumerate(self.agents_rewards):
                            self.tb_logger.add_scalar('policy_training/' + self.cfg.algo_name + '_' + str(i),
                                                      agent_reward[-1],
                                                      episode)
                        self.tb_logger.add_scalar('policy_training/' + self.cfg.algo_name, self.episode_rewards[-1],
                                                  episode)
                        self.episode_rewards.append(0.0)
                        for agent_reward in self.agents_rewards:
                            agent_reward.append(0.0)
                        break

                # train on-policy algorithm
                if self.cfg.learn_type == 'on_policy':
                    self.update_times += 1
                    loss_info = []
                    for trainer_index, agent in enumerate(self.agents):
                        loss = agent.update(self.agents, self.train_step, agent_index=trainer_index)
                        loss_info.append(loss)
                    self.loss_infos.append(loss_info)

                    # clear buffer
                    for agent in self.agents:
                        agent.buffer_clear()
                    # log
                    for i, _loss in enumerate(loss_info):
                        for k, v in _loss.items():
                            self.tb_logger.add_scalar('policy_training/agent_' + str(i) + '_' + k, v, self.update_times)

                # logger.info episode turn
                logger.info("steps: {}, episodes: {}, episode reward: {}, agent episode reward: {}, time: {}".format(
                    self.train_step, len(self.episode_rewards) - 1, np.mean(self.episode_rewards[-2]).round(3),
                    [np.mean(rew[-2]).round(3) for rew in self.agents_rewards], round(time.time() - start_time, 3)))
                self.info.append(info_n)

        self.save_and_show_data()
        self.wrapper_env.close()
        logger.info(f"{self.cfg.algo_name} train successfully!")

    def evaluate(self):
        ####### evalution
        for agent in self.agents:
            agent.eval()
        obs_n = self.wrapper_env.reset()
        reward_eval = [[0.0] for _ in range(self.wrapper_env.n)]
        while True:
            # action_n = [trainer.eval_action(obs_n) for trainer, obs in zip(trainers, obs_n)]
            action_n = [agent.action(obs_n) for agent, obs in zip(self.agents, obs_n)]
            next_obs_n, reward_n, done_n, terminal_n, info_n = self.wrapper_env.step(action_n)
            for i, reward in enumerate(reward_n):
                reward_eval[i][-1] += reward[0]
            if all(terminal_n) or all(done_n):
                break

        # logger.info(episode_rewards)
        # logger.info(agents_rewards)
        logger.info("eval:", np.sum(reward_eval), '\nagent_eval:', reward_eval)

    def save_and_show_data(self):
        logger.info(f"save train data...")
        file_name = self.cfg.save_data_dir + f"{self.cfg.algo_name}_info.pkl"
        with open(file_name, 'wb') as fp:
            pickle.dump(self.info, fp)
            logger.info('save train info successfully！')
        # # plot
        file_name = self.cfg.save_data_dir + f"{self.cfg.algo_name}/{self.cfg.algo_name}_episode_rewards.pkl"
        with open(file_name, 'wb') as fp:
            pickle.dump(self.episode_rewards, fp)
            logger.info('save episode_rewards successfully！')
        file_name = self.cfg.save_data_dir + f"{self.cfg.algo_name}/{self.cfg.algo_name}_agents_rewards.pkl"
        with open(file_name, 'wb') as fp:
            pickle.dump(self.agents_rewards, fp)
            logger.info('save agents_rewards successfully！')

        logger.info(f"save and show train curve...")
        plt.figure()
        plt.plot(self.episode_rewards[:-1], label='episode_rewards')
        plt.xlabel('episodes')
        plt.ylabel('Average Total Reward')
        plt.title(f"{self.cfg.algo_name}")
        plt.legend()
        plt.savefig(self.cfg.save_data_dir + f"{self.cfg.algo_name}/{self.cfg.algo_name}_episode_reward.png")
        plt.show()

        plt.figure()
        for i in range(len(self.agents_rewards)):
            plt.plot(self.agents_rewards[i][:-1], label='agent_' + str(i))
        plt.xlabel('episodes')
        plt.ylabel('mean reward')
        plt.title(f"{self.cfg.algo_name}")
        plt.legend()
        plt.savefig(self.cfg.save_data_dir + f"{self.cfg.algo_name}/{self.cfg.algo_name}_agents_reward.png")
        plt.show()


if __name__ == '__main__':
    from magrl.config.magrl_config import TrainConfig

    args = TrainConfig()
    trainer = MAGRLTrainer(args)
    # 开始训练
    trainer.start_train()
    # 训练完最后评估一次
    trainer.eval()
