import os

import numpy as np
import time
import pickle

from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig
import tools.tf_util as U
from envs.make_env import make_env
from tools.common import set_global_random_seed, regist_algorithm
import matplotlib.pyplot as plt

# trainers
def get_trainers(env, args):
    trainers = []
    num_adversaries = min(env.n, args.num_adversaries)
    # 先red
    adv_trainer, adv_cfg = regist_algorithm(args.adv_algorithm)
    args.algo_config['red'] = adv_cfg
    for i in range(num_adversaries):  # i=0,1,...,num_adversaries
        trainers.append(adv_trainer(args, adv_cfg, i, env.agents))
    # 再blue    
    good_trainer, good_cfg = regist_algorithm(args.good_algorithm)
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
            if all(done_n[0:args.num_adversaries]) or  all(done_n[args.num_adversaries:]): # 红 or 蓝
                done = True
            else:
                done = False
            terminal = (episode_step >= args.max_episode_len)
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
                      'close-game number is {}！'.format(win[0],win[1],win[2]))
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
                U.save_state(args.save_model_dir + str(args.adv_algorithm)+'('+ str(args.adv_policy)+ ')-VS-'+str(args.good_algorithm) + '(' +str(args.good_policy) +')/' + str(len(episode_rewards)) + '/')
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

def torch_train(config):
    print("***********  use algo name is ", config.algo_name)
    print('starting train tsc(policy) to use rl algorithm')
    print('Env initialling...')
    Env = make_env(args.scenario_name)
    print('Env initial successfully!')
    print('MARL algorithm initialing...')
    trainers = get_trainers(Env, args)
    print('MARL algorithm initial successfully!')
    if not os.path.exists(config.save_data_dir):
        os.makedirs(config.save_data_dir)
    if not os.path.exists(config.save_model_dir):
        os.makedirs(config.save_model_dir)
        
    logger = SummaryWriter(config.save_data_dir)
    episode_rewards = [0.0]
    agents_rewards = [[0.0] for _ in range(Env.n)]
    loss_infos = []
    info = []
    train_step = 0
    update_times = 0
    best_reward = -np.inf
    # load model
    if config.load_model:
        for i, trainer in enumerate(trainers):
            file_list = os.listdir(f"{config.load_dir}/{config.algo_name}")
            model_index = file_list[-1]
            trainer.load_model(f"{config.load_dir}/{config.algo_name}/{model_index}/agent_{i}")

    epochs = config.epochs
    episodes = config.episodes
    update_frequence = config.update_step
    print('*********** start training ***********')
    for trainer in trainers:
        trainer.train()

    for epoch in range(epochs):
        for episode in range(episodes):
            obs_n = Env.reset()
            episode_step = 0
            start_time = time.time()
            while True:
                action_n = [trainer.action(obs_n) for trainer, obs in zip(trainers, obs_n)]
                next_obs_n, reward_n, done_n, terminal_n, info_n = Env.step(action_n)
                for i, trainer in enumerate(trainers):
                    trainer.save_replay_buffer(obs_n[i], action_n[i], reward_n[i], next_obs_n[i], done_n[i], terminal_n[i])
                episode_step += 1
                obs_n = next_obs_n
                # can use tensorboard log
                for i, reward in enumerate(reward_n):
                    episode_rewards[-1] += reward[0]
                    agents_rewards[i][-1] += reward[0]
                train_step += 1

                if config.learn_type == 'off_policy':
                    # update agent model
                    if train_step % update_frequence == 0:
                        # TODO: 每个agent可以有不同算法
                        if config.network_structure == 'share':
                            trainers[0].preUpdate()
                        else:
                            for trainer in trainers:
                                trainer.preUpdate()
                        loss_info = []
                        for trainer_index, trainer in enumerate(trainers):
                            loss = trainer.update(trainers, train_step, agent_index=trainer_index)
                            loss_info.append(loss)
                        loss_infos.append(loss_info)
                        for i, _loss in enumerate(loss_info):
                            for k, v in _loss.items():
                                logger.add_scalar('policy_training/agent_' + str(i) + '_' + k, v, update_times)

                if all(terminal_n) or all(done_n):
                    # save_best_model
                    if episode_rewards[-1] > best_reward:
                        best_reward = episode_rewards[-1]
                        for i, trainer in enumerate(trainers):
                            trainer.save_model(f"{config.save_model_dir}/{config.algo_name}/{episode}/agent_{i}")
                        print(f"**** best reward={best_reward}, save agents model successfully!")

                    for i, agent_reward in enumerate(agents_rewards):
                        logger.add_scalar('policy_training/' + config.algo_name + '_' + str(i), agent_reward[-1],
                                          episode)
                    logger.add_scalar('policy_training/' + config.algo_name, episode_rewards[-1],
                                      episode)
                    episode_rewards.append(0.0)
                    for agent_reward in agents_rewards:
                        agent_reward.append(0.0)
                    break

            # train on-policy algorithm
            if config.learn_type == 'on_policy':
                update_times += 1
                loss_info = []
                for trainer_index, trainer in enumerate(trainers):
                    loss = trainer.update(trainers, train_step, agent_index=trainer_index)
                    loss_info.append(loss)
                loss_infos.append(loss_info)

                # clear buffer
                for trainer_index, trainer in enumerate(trainers):
                    trainer.buffer_clear()
                # log
                for i, _loss in enumerate(loss_info):
                    for k, v in _loss.items():
                        logger.add_scalar('policy_training/agent_' + str(i) + '_' + k, v, update_times)

            # print episode turn
            print("steps: {}, episodes: {}, episode reward: {}, agent episode reward: {}, time: {}".format(
                train_step, len(episode_rewards)-1, np.mean(episode_rewards[-2]).round(3),
                [np.mean(rew[-2]).round(3) for rew in agents_rewards], round(time.time() - start_time, 3)))
            info.append(info_n[0])
            
    file_name = config.save_data_dir + f"{config.algo_name}_info.pkl"
    with open(file_name, 'wb') as fp:
        pickle.dump(info, fp)
        print('save train info successfully！')
    # # plot
    file_name = config.save_data_dir + f"{config.algo_name}/{config.algo_name}_episode_rewards.pkl"
    with open(file_name, 'wb') as fp:
        pickle.dump(episode_rewards, fp)
        print('save episode_rewards successfully！')
    file_name = config.save_data_dir + f"{config.algo_name}/{config.algo_name}_agents_rewards.pkl"
    with open(file_name, 'wb') as fp:
        pickle.dump(agents_rewards, fp)
        print('save agents_rewards successfully！')
    plt.figure()
    plt.plot(episode_rewards[:-1], label='episode_rewards')
    plt.xlabel('episodes')
    plt.ylabel('Average Total Reward')
    plt.title(f"{config.algo_name}")
    plt.legend()
    plt.savefig(config.save_data_dir + f"{config.algo_name}/{config.algo_name}_episode_reward.png")
    plt.show()

    plt.figure()
    for i in range(len(agents_rewards)):
        plt.plot(agents_rewards[i][:-1], label='agent_' + str(i))
    plt.xlabel('episodes')
    plt.ylabel('mean reward')
    plt.title(f"{config.algo_name}")
    plt.legend()
    plt.savefig(config.save_data_dir + f"{config.algo_name}/{config.algo_name}_agents_reward.png")
    plt.show()

    Env.close()
    print(f"{config.algo_name} train successfully!")
    
    
    # file_name = config['save_data_dir'] + f"{config['agent_config']['algo_name']}_ATT.pkl"
    # with open(file_name, 'wb') as fp:
    #     pickle.dump(ATT, fp)
    #     print('save ATT successfully！')

    # plt.figure()
    # plt.plot(ATT, label='ATT')
    # plt.xlabel('episodes')
    # plt.ylabel('Average Travel Time')
    # plt.title(f"{config['agent_config']['algo_name']} in 6:00-7:00 of Huo Lin He of Tong Liao")
    # plt.legend()
    # plt.savefig(config['save_data_dir'] + f"{config['agent_config']['algo_name']}_ATT.png")
    # plt.show()

    ####### evalution
    for trainer in trainers:
        trainer.eval()
    obs_n = Env.reset()
    reward_eval =[[0.0] for _ in range(Env.n)]
    while True:
        # action_n = [trainer.eval_action(obs_n) for trainer, obs in zip(trainers, obs_n)]
        action_n = [trainer.action(obs_n) for trainer, obs in zip(trainers, obs_n)]
        next_obs_n, reward_n, done_n, terminal_n, info_n = Env.step(action_n)
        for i, reward in enumerate(reward_n):
            reward_eval[i][-1] += reward[0]
        if all(terminal_n) or all(done_n):
            break

    # print(episode_rewards)
    # print(agents_rewards)
    print("eval:", np.sum(reward_eval), '\nagent_eval:', reward_eval)



if __name__ == '__main__':
    args = TrainConfig()
    # tf_train(args)
    torch_train(args)
