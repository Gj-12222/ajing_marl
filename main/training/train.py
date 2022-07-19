# import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import time
import pickle
import random
import matplotlib.pyplot as plt
from Config import Config
import tools.tf_util as U
from algorithms.maddpg import MADDPGAgentTrainer

from algorithms.masac import MASACAgentTrainer
from algorithms.COMA import COMAAgentTrainer
import algorithms.COMA as coma

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None,activation_fn=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=activation_fn)
        return out

# 调用MPE
def make_env(scenario_name, arglist, benchmark=False):
    from envs.mpe_envs.multiagent.environment import MultiAgentEnv
    import envs.mpe_envs.multiagent.scenarios as scenarios
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multi-agent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,done_callback =scenario.done)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback =scenario.done)
    return env

# 创建算法
def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    if arglist.adv_algorithm == 'maddpg':
        adv_trainer = MADDPGAgentTrainer
    elif arglist.adv_algorithm == 'masac':
        adv_trainer = MASACAgentTrainer  # MASAC算法
    elif arglist.adv_algorithm == 'coma':  # COMA
        adv_trainer = COMAAgentTrainer
    # good
    if arglist.good_algorithm == 'maddpg':
        good_trainer = MADDPGAgentTrainer
    elif arglist.good_algorithm == 'masac':
        good_trainer = MASACAgentTrainer
    elif arglist.good_algorithm == 'coma':
        good_trainer = COMAAgentTrainer

    for i in range(num_adversaries):  # i=0,1,...,num_adversaries
        trainers.append(adv_trainer(
            "agent_%d" % i, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'CTDE')))
    for i in range(num_adversaries, env.n):  # i=num_adversaries, num_adversaries+1, ..., env.n
        trainers.append(good_trainer(
            "agent_%d" % i, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'CTDE')))
    return trainers

# 固定种子
def seed_np_tf_random(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

# 整个的训练过程
def train(arglist):
    with U.single_threaded_session():
        # 固定随机种子
        seed_np_tf_random(arglist.seed)
        # Create environment
        env = make_env(arglist.scenario, arglist)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

        num_adversaries = min(env.n, arglist.num_adversaries)

        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using adv policy {} and good policy {}'.format(arglist.adv_policy, arglist.good_policy))
        # Initialize
        U.initialize()

        if arglist.load_dir == " ":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore:
            print('初始化加载 Loading previous state...')
            U.load_state(arglist.load_dir)  # """从load_dir加载所有变量到当前会话"""
        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        # save losses
        agent_sum_loss = []
        # save rew/reward
        all_rew = []
        agrew = []
        # save win rate
        good_death_num = []
        adv_death_num = []
        # done_reset = None
        # ag_obs = []
        save_win = []
        win = [0, 0, 0]
        t_start = time.time()
        # done_reset = []

        print('Starting iterations...')
        # env.reset()

        # start training
        while True:
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1

            # print(done_n)
            if all(done_n[0:arglist.num_adversaries]) or  all(done_n[arglist.num_adversaries:]): # 红 or 蓝
                done = True
            else:
                done = False
            # print(done)

            terminal = (episode_step >= arglist.max_episode_len)
            # print(terminal)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                agent_adv_death_num = 0
                agent_good_death_num = 0

                for i,agent in enumerate(env.agents):
                    if done_n[i] == True:
                        if agent.adversary: #
                            agent_adv_death_num += 1
                        else:     # good
                            agent_good_death_num += 1
                good_death_num.append(agent_good_death_num)
                adv_death_num.append(agent_adv_death_num)
                if agent_good_death_num == arglist.num_agents - arglist.num_adversaries:
                    win[0] += 1
                elif agent_adv_death_num == arglist.num_adversaries:
                    win[1] += 1
                else:
                    win[2] += 1  #
                print('红方胜{}次，蓝方胜{}次，平局{}次！'.format(win[0],win[1],win[2]))
                save_win.append(win)
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)

                agent_info.append([[]])
            # increment global step counter
            
            train_step += 1

            # for displaying learned policies
            if arglist.display:
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            if train_step % 100 == 0:
                agent_loss = []
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step)
                    agent_loss.append(loss)
                agent_sum_loss.append(agent_loss)
            # save model, display training output
            # print(terminal)
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_model_dir + str(arglist.adv_algorithm)+'('+ str(arglist.adv_policy)+ ')-VS-'+str(arglist.good_algorithm) + '(' +str(arglist.good_policy) +')/' + str(len(episode_rewards)) + '/', saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time() - t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                            train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                            [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.save_data_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                    print('存储ep_rewards成功！')

                agrew_file_name = arglist.save_data_dir + arglist.exp_name + '_agrewards.pkl'

                with open(agrew_file_name, 'wb') as fp:

                    pickle.dump(final_ep_ag_rewards, fp)
                    print('存储ag_rewards成功！')
                agloss_file_name = arglist.save_data_dir + arglist.exp_name + '_agloss.pkl'
                with open(agloss_file_name, 'wb') as fp:
                    pickle.dump(agent_sum_loss, fp)
                    print('存储ag_loss成功！')

                good_file_name = arglist.save_data_dir + arglist.exp_name + '_good_death.pkl'
                with open(good_file_name, 'wb') as fp:
                    pickle.dump(good_death_num, fp)
                    print('存储good_death_num成功！')

                adv_file_name = arglist.save_data_dir + arglist.exp_name + '_adv_death.pkl'
                with open(adv_file_name, 'wb') as fp:
                    pickle.dump(adv_death_num, fp)
                    print('存储adv_death_num成功！')

                all_rew_name = arglist.save_data_dir + arglist.exp_name + '_everyep_allrew.pkl'
                with open(all_rew_name, 'wb') as fp:
                    pickle.dump(episode_rewards, fp)
                    print('存储all_rew成功！')

                agrew_name = arglist.save_data_dir + arglist.exp_name + '_everyep_agrew.pkl'
                with open(agrew_name, 'wb') as fp:
                    pickle.dump(agent_rewards, fp)
                    print('存储agrew成功！')

                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break
        env.close()
        print('terminal training !')


if __name__ == '__main__':
    arglist = Config()
    train(arglist)
