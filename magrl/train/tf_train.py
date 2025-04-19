import os

import numpy as np
import time
import pickle

import magrl.utils.tf_util as U
from config import TrainConfig
from magrl.envs.make_env import make_env
from magrl.utils.common import set_global_random_seed, get_rl_algorithm


# trainers
def get_trainers(env, args):
    trainers = []
    num_adversaries = min(env.n, args.num_adversaries)
    # 先red
    adv_trainer, adv_cfg = get_rl_algorithm(args.adv_algorithm)
    args.algo_config['red'] = adv_cfg
    for i in range(num_adversaries):  # i=0,1,...,num_adversaries
        trainers.append(adv_trainer(args, adv_cfg, i, env.agents))
    # 再blue    
    good_trainer, good_cfg = get_rl_algorithm(args.good_algorithm)
    args.algo_config['blue'] = adv_cfg
    for i in range(num_adversaries, env.n):  # i=num_adversaries, num_adversaries+1, ..., env.n
        trainers.append(good_trainer(args, good_cfg, i, env.agents))
    return trainers


def tf_train(args):
    with U.single_threaded_session():
        set_global_random_seed(args.seed)
        # Create environment
        env = make_env(args.scenario_name)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, args)
        print('Using adv policy {} and good policy {}'.format(args.adv_policy, args.good_policy))

        # Initialize
        U.initialize()
        if args.load_model:
            print('初始化加载 Loading previous state...')
            if args.load_dir == " ":
                args.load_dir = args.save_dir
            U.load_state(args.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        # loss
        agent_sum_loss = []
        # win rate
        good_death_num = []
        adv_death_num = []
        save_win = []
        win = [0, 0, 0]
        print('Starting iterations...')
        # env.reset()
        # start training
        t_start = time.time()
        while True:
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            if all(done_n[0:args.num_adversaries]) or all(done_n[args.num_adversaries:]):  # 红 or 蓝
                done = True
            else:
                done = False
            terminal = (episode_step >= args.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            # collect reward in step
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
            if done or terminal:
                agent_adv_death_num = 0
                agent_good_death_num = 0

                for i, agent in enumerate(env.agents):
                    if done_n[i]:
                        if agent.adversary:
                            agent_adv_death_num += 1
                        else:
                            agent_good_death_num += 1
                good_death_num.append(agent_good_death_num)
                adv_death_num.append(agent_adv_death_num)
                if agent_good_death_num == args.num_agents - args.num_adversaries:
                    win[0] += 1
                elif agent_adv_death_num == args.num_adversaries:
                    win[1] += 1
                else:
                    win[2] += 1  #
                print(' red number is {}，'
                      ' blue number is {}，'
                      ' draw number is {}！'.format(win[0], win[1], win[2]))
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
            if args.display:
                env.render()
                # continue

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
            if terminal and (len(episode_rewards) % args.save_rate == 0):
                U.save_state(args.save_model_dir + str(args.adv_algorithm) + '(' + str(args.adv_policy) + ')-VS-' + str(
                    args.good_algorithm) + '(' + str(args.good_policy) + ')/' + str(len(episode_rewards)) + '/')
                # print statement depends on whether or not there are adversaries
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-args.save_rate:]),
                    [np.mean(rew[-args.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-args.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-args.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > args.num_episodes:
                rew_file_name = args.save_data_dir + args.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                    print('save ep_rewards successfully！')

                agrew_file_name = args.save_data_dir + args.exp_name + '_agrewards.pkl'

                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                    print('save ag_rewards successfully！')
                agloss_file_name = args.save_data_dir + args.exp_name + '_agloss.pkl'
                with open(agloss_file_name, 'wb') as fp:
                    pickle.dump(agent_sum_loss, fp)
                    print('save ag_loss successfully！')

                good_file_name = args.save_data_dir + args.exp_name + '_good_death.pkl'
                with open(good_file_name, 'wb') as fp:
                    pickle.dump(good_death_num, fp)
                    print('save good_death_num successfully！')

                adv_file_name = args.save_data_dir + args.exp_name + '_adv_death.pkl'
                with open(adv_file_name, 'wb') as fp:
                    pickle.dump(adv_death_num, fp)
                    print('save adv_death_num successfully！')

                all_rew_name = args.save_data_dir + args.exp_name + '_everyep_allrew.pkl'
                with open(all_rew_name, 'wb') as fp:
                    pickle.dump(episode_rewards, fp)
                    print('save all_rew successfully！')

                agrew_name = args.save_data_dir + args.exp_name + '_everyep_agrew.pkl'
                with open(agrew_name, 'wb') as fp:
                    pickle.dump(agent_rewards, fp)
                    print('save agrew successfully！')

                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break
        env.close()
        print('terminal training !')


if __name__ == '__main__':
    args = TrainConfig()
    tf_train(args)
