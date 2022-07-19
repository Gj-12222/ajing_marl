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

# 定义训练所需的参数：
#     环境相关参数
#     训练用的超参数定义
#     checkpointing（用于存储数据和模型）
#     测试阶段的参数
# Environment环境相关参数

# def parse_args():
#     parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")
#     #     环境相关参数
#     parser.add_argument("--scenario", type=str, default="uavs_5v5", help="name of the scenario script")  # 场景名称
#     parser.add_argument("--max-episode-len", type=int, default=300, help="maximum episode length")  # 最大片段长度
#     parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")  # 总片段数
#     # parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")  # 敌方数量
#     parser.add_argument("--num-adversaries", type=int, default=4, help="number of adversaries")  # 敌方数量
#     parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")  # 对抗策略
#     parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")  # 合作方智能体策略
#     # 各类算法
#     parser.add_argument("--adv-algorithm", type=str, default="maddpg", help="name of algorithm")  # 红方使用算法      # adv-red
#     parser.add_argument("--good-algorithm", type=str, default="maddpg", help="name of algorithm")  # 蓝方使用算法     # good-blue
#     # 实验参数
#     parser.add_argument("--seed", type=int, default=3407, help="random of seed")  # 固定随机种子
#     parser.add_argument("--fix-alpha", action="store_true", default=True)  # 固定温度系数
#     parser.add_argument("--init-alpha", type=float, default=0.2, help="weight of wendu")  # 熵权重 温度系数
#     parser.add_argument("--target-actor", action="store_true", default=False)  # 使用策略目标网络
#     parser.add_argument("--td-lambda", type=float, default=0.8, help="weight of wendu")  # TD(λ)的 λ
#     # Core training parameters训练用的超参数定义
#     parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")  # 优化器中的学习率alpha-lr-0.01
#     """
#     可以看到  0.96, 0.98, 0.99, 0.995 的gamma值
#     分别对应    56,  114,  229,   459 的步数
#     """
#     parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")  # 折扣因子gamma
#     parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")  # 经验库中提取的数量
#     parser.add_argument("--num-units", type=int, default=128,help="number of units in the mlp")  # 在MLP(多层感知器-(Multi-Layer Perceptron)中的神经元数
#     # Checkpointing（用于存储数据和模型）
#     parser.add_argument("--exp-name", type=str, default='4V8', help="name of the experiment")  # 实验名称
#     parser.add_argument("--save-dir", type=str, default="./training/model/save/", help="directory in which training state and model should be saved")  # 保存训练状态和模型的目录
#     parser.add_argument("--save-rate", type=int, default=4, help="save model once every time this many episodes are completed")  # 每1次保存一次模型-神经网络
#     parser.add_argument("--load-dir", type=str, default="./training/model/load/", help="directory in which training state and model are loaded")  # 训练状态和模型的目录被载入
#     # Evaluation-测试阶段的参数
#     parser.add_argument("--restore", action="store_true", default=False)  # 重置
#     parser.add_argument("--display", action="store_true", default=False)  # 输出display
#     # parser.add_argument("--display", action="store_true", default=True)  # 输出display
#     parser.add_argument("--benchmark", action="store_true", default=False)  # 标准检查程序-基准
#     return parser.parse_args()


# 定义了agent所需的网络结构，用全连接层MLP

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None,activation_fn=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=activation_fn)
        return out

# 调用MPE（多智能体处理环境Multi-Agent Processing Environment）环境
def make_env(scenario_name, arglist, benchmark=False):
    from envs.mpe_envs.multiagent.environment import MultiAgentEnv
    import envs.mpe_envs.multiagent.scenarios as scenarios
    # load scenario from script
    # 从脚本加载场景
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world——建立场景
    world = scenario.make_world()
    # create multi-agent environment——建立多智能体环境
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data,done_callback =scenario.done)
    else:  # 不使用
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback =scenario.done)
    return env

# 创建算法
def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []  # 预置敌方trainer的空间
    if arglist.adv_algorithm == 'maddpg':
        #adv_model = mlp_model  # 定义model神经网络-2层全连接层64个神经元，1个输出层1个神经元
        adv_trainer = MADDPGAgentTrainer  # 定义多智能体MADDPG算法-集中训练方式
    elif arglist.adv_algorithm == 'masac':
        #adv_model = sac_nn  # 定义model神经网络-3层全连接层128个神经元，1个输出层1个神经元
        adv_trainer = MASACAgentTrainer  # MASAC算法
    elif arglist.adv_algorithm == 'coma':  # COMA
        #adv_model = [coma.coma_actor_rnn, coma.coma_critic_mlp]
        adv_trainer = COMAAgentTrainer  # 定义多智能体MADDPG算法-集中训练方式
    # good-蓝方
    if arglist.good_algorithm == 'maddpg':
        #good_model = mlp_model  # 定义model神经网络-2层全连接层64个神经元，1个输出层1个神经元
        good_trainer = MADDPGAgentTrainer  # 定义多智能体MADDPG算法-集中训练方式
    elif arglist.good_algorithm == 'masac':
        #good_model = sac_nn  # 定义model神经网络-3层全连接层128个神经元，1个输出层1个神经元
        good_trainer = MASACAgentTrainer  # MASAC算法
    elif arglist.good_algorithm == 'coma': # COMA
        #good_model = [coma.coma_actor_rnn, coma.coma_critic_mlp]
        good_trainer = COMAAgentTrainer  # 定义多智能体MADDPG算法-集中训练方式

    # 定义了敌方的agent的训练算法MADDPG
    for i in range(num_adversaries):  # i=0,1,...,num_adversaries
        trainers.append(adv_trainer(
            "agent_%d" % i, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'CTDE')))
    # 定义了除敌方agent之外的所有agent的训练算法MADDOG
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
    with U.single_threaded_session():  # 返回一个只使用单个CPU的会话
        # 固定随机种子
        # seed_np_tf_random(arglist.seed)
        # Create environment-创建scenario场景的多智能体交互的环境
        env = make_env(arglist.scenario, arglist)
        # Create agent trainers-创建智能体集中训练
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        # 敌方agent数量num_adversaries
        num_adversaries = min(env.n, arglist.num_adversaries)
        # 创建敌方agent的集中训练模型-class类
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using adv policy {} and good policy {}'.format(arglist.adv_policy, arglist.good_policy))
        # Initialize-初始化操作
        U.initialize()
        # 是否加载已有模型
        if arglist.load_dir == " ":  # 如果load_dir是空的
            arglist.load_dir = arglist.save_dir  # 就把上episodes保存的数据给load_dir
        if arglist.display or arglist.restore:  # 如果display=benchmark，restore=benchmark
            print('初始化加载 Loading previous state...')  # 打印输出'Loading previous state...'
            U.load_state(arglist.load_dir)  # """从load_dir加载所有变量到当前会话"""
        # 定义训练变量
        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        # 保存loss数据
        agent_sum_loss = []
        # 保存每步rew
        all_rew = []
        agrew = []
        # 保存胜率
        good_death_num = []
        adv_death_num = []
        # 边界标志
        # done_reset = None
        # ag_obs = []
        save_win = []
        win = [0, 0, 0]
        t_start = time.time()
        # done_reset = []

        print('Starting iterations...')
        # env.reset()  # 初始化环境

        # 主体循环部分： 训练开始
        while True:
            ####guojing set  防止越界
            # if not any(done_reset):
            #     obs_n = env.reset()
            # get action-对每一个agent的动作action-->根据定义的agent模型及交互算法与一一对应的agent初始观测状态O，绑定一起返回位元组方式：
            # [[agent1,obs1];[agent2,obs2];[agent3,obs3];...;[agentN,obsN]]
            # 通过trainers.action(self,obs) 输入对应状态obs_i,输出对应经过策略网络p_act得到的动作action_i,(i=1,2,...,n)
            # if arglist.display:  # 显示策略标志
            #     time.sleep(0.01)
            #     env.render()
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]  # 获取所有智能体的动作列表（每一个智能体）
            # environment step-输入所有agent的动作集合action_n,与环境交互，
            # 得到下一时刻的观测状态集合new_obs_n,奖励集合rew_n,终止回合标志集合done_n,调试的诊断信息-反馈信息集合info_n
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            
              # 渲染出当前的智能体以及环境的状态-绘制图形
            #for i in range(env.n):
            #    ag_obs = [abs(new_obs_n[i][2]), abs(new_obs_n[i][3])]
            #    if max(ag_obs) > 1.5:
            #        done_reset.append(True)
            #        ag_obs = np.random.uniform(-1.5, +1.5, 2)
            #        new_obs_n[i][2] = ag_obs[0]
            #        new_obs_n[i][3] = ag_obs[1]
                    # break

            # 回合内步数+1
            episode_step += 1
            # all()判断done_n是否都为TRUE，是 返回True，否 返回False
            #红蓝任意一方全灭，终止回合
            # print(done_n)
            if all(done_n[0:arglist.num_adversaries]) or  all(done_n[arglist.num_adversaries:]): # 红 or 蓝
                done = True
            else:
                done = False
            # print(done)
            # 终止条件：回合内步数大于等于最大回合步数 terminal=1，否则 terminal=0
            terminal = (episode_step >= arglist.max_episode_len)
            # print(terminal)
            # collect experience-分别对每个agent收集经验到不同的经验库
            for i, agent in enumerate(trainers):  # 给trainer编序号 ([0,agent1];[1,agent2];[2,agent3];...;[n,agentn])
                # 存储记忆库-第i个agent的obs_n[i]观测状态，动作action_n[i],奖励汇报rew_n[i],下一回合步状态new_obs_n[i],判决标志done_n[i],终止条件terminal
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            # 更新状态obs_n = new_obs_n  S <- S'
            obs_n = new_obs_n
            # 对每个agent的奖励回报编序号([0,rew_n[0]];[1,rew_n[1]];[2,rew_n[2]];...;[n,rew_n[n])
            for i, rew in enumerate(rew_n):
                # 对回合中每个agent奖励回报求和得到回合求和回报
                episode_rewards[-1] += rew
                # 对当前回合每个agent的奖励生成二维数据--行为agent序号，列为对应的奖励回报
                agent_rewards[i][-1] += rew
            # # 判断终止条件
            # if train_step >198:
            #      print(train_step)
            if done or terminal:  # done 与 terminal 任一为True
                # 判断agent死亡状态
                agent_adv_death_num = 0
                agent_good_death_num = 0

                # arglist.num_adversaries = 8
                for i,agent in enumerate(env.agents):
                # for ag_done in done_n:
                    if done_n[i] == True:
                        if agent.adversary:  # 是敌方-红方
                            agent_adv_death_num += 1  # 红方无人机死的数量
                        else:     # 是友军-蓝方
                            agent_good_death_num += 1  # 蓝方无人机死的数量
                good_death_num.append(agent_good_death_num)
                adv_death_num.append(agent_adv_death_num)
                if agent_good_death_num == 2:  # 红方赢
                    win[0] += 1
                elif agent_adv_death_num == 2:   # 蓝方赢
                    win[1] += 1
                else:
                    win[2] += 1  # 平局
                print('红方胜{}次，蓝方胜{}次，平局{}次！'.format(win[0],win[1],win[2]))
                save_win.append(win)
                obs_n = env.reset()  # 重置环境，obs_n为初始环境观测状态
                episode_step = 0  # 重置回合数为0
                episode_rewards.append(0)  # 回合总回报为0
                for a in agent_rewards:  # 对每个agent的奖励回报为0
                    a.append(0)
                # 环境的反馈信息info 为0
                agent_info.append([[]])
            # env.render()  # 渲染环境，显示环境
            # time.sleep(0.05)
            # increment global step counter-全局步进计数增加器
            # 回合内部步数更新
            
            train_step += 1

            # for displaying learned policies-显示学习到的策略
            if arglist.display:  # 显示策略标志
                # time.sleep(0.01)
                env.render()  # 渲染出当前的智能体以及环境的状态-绘制图形
                continue

            # update all trainers, if not in display or benchmark mode
            # 如果不是在显示模式或基准模式，更新所有的训练器，
            if train_step % 100 == 0:
            # if False:
                agent_loss = []
                for agent in trainers:  # 遍历所有agent
                    agent.preupdate()  # 预清空-记忆库取样的索引（下标）
                for agent in trainers: # 计算损失函数loss
                    loss = agent.update(trainers, train_step)
                    agent_loss.append(loss)  # 每100 step 存储12个agent的list形式的loss，这就意味着24w个数据存储在agent_loss
                agent_sum_loss.append(agent_loss) # 在最后一个append中，存储了从1~terminal的所有loss
            # save model, display training output-保存模型，显示训练结果
            # 条件：terminal=True & 总回报长度episode_rewards = 指定保存长度save_rate
            # print(terminal)
            # 回合终止，且训练回合数超过save_rate
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            #if False:
                # 保存状态save_state和训练模型，save_dir为目录
                U.save_state(arglist.save_model_dir + str(arglist.adv_algorithm)+'('+ str(arglist.adv_policy)+ ')-VS-'+str(arglist.good_algorithm) + '(' +str(arglist.good_policy) +')/' + str(len(episode_rewards)) + '/', saver=saver)
                # print statement depends on whether or not there are adversaries
                # 输出声明取决于是否有对手
                # if num_adversaries == 0:  # 没有对手
                #     # 显示步数train_step，回合数episode_rewards，平均回报episode_rewards，时间-保留小数点3位(四舍五入)的用时
                #     print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                #         train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time() - t_start, 3)))
                # else:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward-追加最终所有回合的总奖励
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:  # 追加save_rate长的每个agent奖励回报求均值
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            # 保存最终的奖励，以便稍后绘制训练曲线
            # 回合数大于指定回合数
            if len(episode_rewards) > arglist.num_episodes:
                # 总回报文件命名rew_file_name
                rew_file_name = arglist.save_data_dir + arglist.exp_name + '_rewards.pkl'
                # rew_file_name = 'D:/software/PyCharm/pyCharmProject/MADDPG/testPaper-MADDPG/venv' + arglist.save_data_dir +  '_rewards.pkl'
                # 打开奖励文件-wb以二进制写方式打开，只能写文件， 如果文件不存在，创建该文件；如果文件已存在，则覆盖写。
                with open(rew_file_name, 'wb') as fp:
                    # 序列化对象，将对象final_ep_rewards保存到文件指针fp指向的文件中去
                    pickle.dump(final_ep_rewards, fp)
                    print('存储ep_rewards成功！')
                # agent的奖励文件命名agrew_file_name
                agrew_file_name = arglist.save_data_dir + arglist.exp_name + '_agrewards.pkl'
                # agrew_file_name = 'D:/software/PyCharm/pyCharmProject/MADDPG/testPaper-MADDPG/venv' + arglist.save_data_dir  + '_agrewards.pkl'
                # 打开agent的奖励文件
                with open(agrew_file_name, 'wb') as fp:
                    # 序列化对象，将对象final_ep_ag_rewards保存到文件指针fp指向的文件中去
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
        # f = open(rew_file_name)
        # data = pickle.load(f)
        #  print('final_ep_rewards=',data)  # show file
        # print('loss=', data)
        # plt.plot(loss)
        # plt.show()


if __name__ == '__main__':
    # 初始化所有参数parse_args
    # arglist = parse_args()
    arglist = Config()
    # 传递所有参数arglist，开始集中训练
    train(arglist)
