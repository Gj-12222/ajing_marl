"""
MADQN算法核心

传统的DQN应用CT框架得到的MADQN
"""
from algorithms import AgentTrainer
from tools import tf_util as U
import tensorflow as tf
from tools.distributions import make_pdtype
from tools.replay_buffer import ReplayBuffer
import numpy as np
import tensorflow.contrib.layers as layers

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None,activation_fn=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=activation_fn)
        return out

def soft_update_vars(q_vars, q_target_vars,tau):
    soft_update = []
    for q_var, q_target_var in zip(sorted(q_vars,key=lambda v:v.name),sorted(q_target_vars,key=lambda v:v.name)):
        soft_update.append(q_target_var.assign((1 - tau)*q_target_var + tau*q_var))
    soft_update = tf.group(*soft_update)
    return U.function([],[],updates=[soft_update])

def q_train(make_obs_n,
                act_space_n,
                q_model,
                agent_index,
                optimizer,
                gram_normal_clipping=0.5,
                local_q_func=False,
                num_units=64,
                scope="trainers",
                tau=0.001):
    with tf.variable_scope(scope,reuse=False):
        # obs_ph_n,
        # 获取动作分布
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # 获取动作分布同类型的占位符placeholder
        # [100,1]
        act_ph_n = [act_pdtype_n[i].sample_placeholder(prepend_shape=[None], name="action" +str(i)) for i in range(len(act_space_n))]

        action_index_ph = tf.placeholder(dtype=tf.int64, shape=[None,2], name="action_index")
        # Q网络的输入-MARL
        obs_ph_n = make_obs_n
        # q_input = obs_ph_n
        # if local_q_func: # DQN
            # q_input = obs_ph_n[agent_index]
        q_input = obs_ph_n
        # 生成q值 batch-size= 100, q= [100,act_dim] act_dim = [(w,0),(s,0),(a,0),(d,0),(0,0),(w,1),(s,1),(a,1),(d,1),(0,1)]
        q = q_model(q_input,int(act_pdtype_n[agent_index].param_shape()[0]),scope="q_model", num_units=num_units,activation_fn=None)
        # 网络的weight + bias
        q_vars = U.scope_vars(U.absolute_scope_name("q_model"))
        # 获取q值分布
        # q_pd = act_pdtype_n[agent_index].pdfromflat(q)
        # q_sampe = q_pd.sample()  # 分布 [-1,1]
        # q_loss
        # u_ph = tf.placeholder(dtype=tf.float32, shape=[None],name="U")
        # 选q(st,at) [100,act_dim] ——> [100,1]
        q_temp = tf.placeholder(dtype=tf.float32, shape=[None],name="q_s_a")

        q_predit = tf.gather_nd(q, action_index_ph)  # 提取对应index下的q

        q_loss = - tf.reduce_mean(tf.square(q_predit - q_temp)) # [1]
        epilon = tf.reduce_mean(q)
        loss = q_loss + 1e-3 * epilon

        # update
        optimizers_var = U.minimize_and_clip(optimizer=optimizer,objective=loss,var_list=q_vars,clip_val=gram_normal_clipping)
        # 用计算图更新
        q_probs_sample = U.function(inputs=obs_ph_n, outputs=q)

        train = U.function(inputs=obs_ph_n + [action_index_ph] + [q_temp],outputs=loss,updates=[optimizers_var])
        # q_values
        # 目标网络-输出所有动作的q值 [100, act-dim]
        # q_target_input = tf.concat(obs_ph_n,1) # MADQN
        # q_target_input = obs_ph_n[agent_index]
        # if local_q_func: # DQN
        #     # q_target_input =tf.concat(obs_ph_n[agent_index],1)
        #     q_target_input = obs_ph_n[agent_index]
        target_q_prb = q_model(q_input, int(act_pdtype_n[agent_index].param_shape()[0]), scope="q_target_model", num_units=num_units, activation_fn=None)
        target_q_max = tf.reduce_max(target_q_prb, axis=1, keep_dims=False)
        # 更新网络
        q_target_vars = U.scope_vars(U.absolute_scope_name("q_target_model"))
        # q_next_target_predit = tf.gather_nd(target_q, action_index_ph)  # argmanxQ(s‘,a')不需要提取对应index下的q

        q_target_soft_update = soft_update_vars(q_vars=q_vars,q_target_vars=q_target_vars,tau=tau)
        q_target_max_values = U.function(inputs=obs_ph_n, outputs=target_q_max)

        return q_probs_sample, train, q_target_soft_update, q_target_max_values
    # pass


class MADQNAgentTrainer(AgentTrainer):
    def __init__(self, name, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        # ①定义变量
        self.name=name
        self.agent_index = agent_index
        self.n = len(obs_shape_n)
        self.args = args
        self.actor_space_n = act_space_n
        # ②创建Q网络
        # Q(s,a) - s, a
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        self.q_probs_sample, self.q_train, self.q_update, self.q_target_values = q_train(scope=name,
                make_obs_n=obs_ph_n,
                act_space_n=act_space_n,
                q_model=mlp_model,
                agent_index=agent_index,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr * 0.05),
                gram_normal_clipping=0.5,
                local_q_func=local_q_func,
                num_units=args.num_units,
                tau=args.tau)

        self.delta = args.delta
        # ③记忆库
        self.dqn_replay_buffer = ReplayBuffer(size=1e6)
        self.dqn_sample = args.batch_size * args.max_episode_len


    # 产生动作 dleta-greedy
    def action(self, obs_n):

        if np.random.rand() <= self.delta:  # epsilon-greedy
            actions = [np.random.randint(low=0,high=(self.actor_space_n[self.agent_index].n)-1, dtype=np.int64) ]# (5,1)是动作
        else:
            obs_n = np.array(obs_n)
            q_probs = self.q_probs_sample(obs_n)
            actions = [np.argmax(q_probs)]
        return actions

    # 加入记忆库
    def experience(self, obs, act, rew, obs_next, done, terminal):
        self.dqn_replay_buffer.add(obs, act, rew, obs_next, float(done and terminal))

    # 清空索引
    def preupdate(self):
        self.replay_sample_index = None
    # 更新网络
    def update(self, agents, t):
        #  当记忆库长度大于最大记忆库的长度的时候 需要更新
        if len(self.dqn_replay_buffer) < self.dqn_sample:
            return
        #  时间步到达一定次数：t = 100 步更新
        if not t % 10 == 0:
            return
        # 每次更新退火
        if self.delta <= 0.001:
            self.delta = 0.001
        elif t % 6000:
            self.delta = self.delta - 0.005
        # 获取每一个智能体的（状态Si，动作Ai，奖励Ri，下一状态Si+1）
        # 这一步有问题 # [100,1]
        self.replay_sample_index =self.dqn_replay_buffer.make_index(self.args.batch_size)
        # 收集所有智能体的replay样本
        # 首先建立空表，然后在索引添加
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        # 然后开始收集数据
        for i in range(self.n):
            obs, act, _, obs_next, _ = agents[i].dqn_replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        # act_n= obs_n = [n, [batch_size,obs_dim]] = [n, batch_size, obs_dim]
        # done = rew [batch_size, 1]
        _, self_act, rew, _, done = self.dqn_replay_buffer.sample_index(index)

        #
        # 开始训练Q网络
        num_sample = 1   # 单样本的数量
        target_u = 0.0    # 设置初始Q值

        # 开始训练
        for i in range(num_sample):
            # 获取下一时刻的联合动作
            # target_act_next_n = [agents[i].q_act_sample(obs_next_n[i]) for i in range(self.n)]
            q_target_next_max = self.q_target_values(obs_next_n)  # [batch_size,1] = q_next_max
            # q_next_max = np.max(q_next_probs,1) # [ batch_size,1]
            # 计算Q（St+1，At+1）
            # target_q_next = q_target_values(*(obs_next_n+target_act_next_n))
            # 计算Q值 Q = R + gamma * (1 - done——终止条件) * Q
            target_u += rew + self.args.gamma * (1 - done) * q_target_next_max  # [batch_size,1]
        # 求平均值
        target_u /= num_sample
        # q(si,ai)
        # q_probs = self.q_probs_sample(obs_n) # [batch_size,act_dim]
        # q_probs_T = q_probs.T # [act_dim,batch_size]
        # q = np.choose(self_act,q_probs.T)
        # q = q_probs[self_act]  # [batch_size, 1]
        # 首先获得一个等差序列，为batch_size的数据排序，以方便提取对应索引下的q值
        action_index_arithmetic_sequence = np.linspace(start=0,
                                                       stop=len(self_act)-1,
                                                       num=len(self_act),
                                                       dtype=np.int64).reshape(-1,1)
        self_act = self_act.reshape(-1, 1)
        # 然后按列 axis=1 拼接
        action_indexs = np.concatenate((action_index_arithmetic_sequence,self_act), axis=1)
        # 计算损失函数
        q_loss = self.q_train(*(obs_n + [action_indexs] + [target_u]))
        # 软更新目标网络
        self.q_update()

        return [q_loss, np.mean(target_u), np.mean(rew),  np.std(target_u)]












