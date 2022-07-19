# IDDQN算法核心

# 传统的DQN应用DTDE框架得到的IDDQN
#  idoubledqn   适用--> 连续状态-离散动作

from algorithms import AgentTrainer
from tools import tf_util as U
import tensorflow as tf
from tools.distributions import make_pdtype
from .replay_buffer import ReplayBuffer
import numpy as np


def soft_update_vars(q_vars, q_target_vars,tau):
    soft_update = []
    for q_var, q_target_var in zip(sorted(q_vars,key=lambda v:v.name),sorted(q_target_vars,key=lambda v:v.name)):
        soft_update.append(q_target_var.assign((1 - tau) * q_target_var + tau * q_var))
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

        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        act_ph_n = [act_pdtype_n[i].sample_placeholder(prepend_shape=[None], name="action" +str(i)) for i in range(len(act_space_n))]
        action_index_ph = tf.placeholder(dtype=tf.int64, shape=[None,2], name="action_index")

        obs_ph_n = make_obs_n
        q_input = obs_ph_n[agent_index]
        q = q_model(q_input,int(act_pdtype_n[agent_index].param_shape()[0]),scope="q_model", num_units=num_units,activation_fn=None)

        q_vars = U.scope_vars(U.absolute_scope_name("q_model"))
        # 获取q值分布
        q_temp = tf.placeholder(dtype=tf.float32, shape=[None],name="q_s_a")
        q_predit = tf.gather_nd(q, action_index_ph)  # 提取对应index下的q
        q_loss = - tf.reduce_mean(tf.square(q_predit - q_temp)) # [1]
        epilon = tf.reduce_mean(q)
        loss = q_loss + 1e-3 * epilon
        optimizers_var = U.minimize_and_clip(optimizer=optimizer,objective=loss,var_list=q_vars,clip_val=gram_normal_clipping)
        # 用计算图更新
        q_probs_sample = U.function(inputs=[obs_ph_n[agent_index]], outputs=q)
        train = U.function(inputs=[obs_ph_n[agent_index]] + [action_index_ph] + [q_temp],outputs=loss,updates=[optimizers_var])

        target_q_prb = q_model(q_input, int(act_pdtype_n[agent_index].param_shape()[0]), scope="q_target_model", num_units=num_units, activation_fn=None)
        q_target_vars = U.scope_vars(U.absolute_scope_name("q_target_model"))
        q_target_soft_update = soft_update_vars(q_vars=q_vars,q_target_vars=q_target_vars,tau=tau)

        q_target_rpb_values = U.function(inputs=[obs_ph_n[agent_index]],outputs=target_q_prb)

        return q_probs_sample, train, q_target_soft_update, q_target_rpb_values

class IDoubleDQNAgentTrainer(AgentTrainer):
    def __init__(self, name,
                 adv_model,
                 obs_shape_n,
                 act_space,  # n个act_space
                 agent_index,
                 args,
                 local_q_func=False):
        # ①定义变量
        self.name=name
        self.agent_index = agent_index
        self.n = len(obs_shape_n)
        self.args = args
        self.actor_space_n = act_space
        # ②创建Q网络
        # Q(s,a) - s, a
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        self.q_probs_sample, self.q_train, self.q_update, self.q_target_values = q_train(scope=name,
                make_obs_n=obs_ph_n,
                act_space_n=act_space,
                q_model=adv_model,
                agent_index=agent_index,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr * 0.05),
                gram_normal_clipping=0.5,
                local_q_func=local_q_func,
                num_units=args.num_units,
                tau=args.tau)

        self.delta = args.delta
        # ③记忆库
        self.dqn_replay_buffer = ReplayBuffer(size=2e6)
        self.dqn_sample = args.batch_size * args.max_episode_len


    # 产生动作 dleta-greedy
    def action(self, obs):

        if np.random.rand() <= self.delta:  # epsilon-greedy
            action_random_probs = np.random.random(size=[self.actor_space_n[self.agent_index].n,])
            actions = [np.argmax(action_random_probs)]
        else:
            q_probs = self.q_probs_sample(obs[None])[0]
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
        if len(self.dqn_replay_buffer) < self.dqn_sample: return
        if not t % 100 == 0: return
        self.replay_sample_index =self.dqn_replay_buffer.make_index(self.args.batch_size)
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, _, obs_next, _ = agents[i].dqn_replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        _, self_act, rew, _, done = self.dqn_replay_buffer.sample_index(index)

        num_sample = 1
        target_u = 0.0

        for i in range(num_sample):

            """计算double DQN"""
            q_target_next_prb = self.q_target_values(obs_next_n[self.agent_index])
            q_next_prb = self.q_probs_sample(obs_next_n[self.agent_index])
            action_next_max_indexs = np.argmax(q_next_prb, axis=1)
            q_target_next_max = q_target_next_prb[np.arange(len(action_next_max_indexs)),action_next_max_indexs]
            # 计算Q值 Q = R + gamma * (1 - done——终止条件) * Q
            target_u += rew + self.args.gamma * (1 - done) * q_target_next_max
        target_u /= num_sample
        action_index_arithmetic_sequence = np.linspace(start=0, stop=len(self_act)-1, num=len(self_act), dtype=np.int64).reshape(-1,1)
        self_act = self_act.reshape(-1, 1)
        action_indexs = np.concatenate((action_index_arithmetic_sequence,self_act), axis=1)
        # 计算损失函数
        q_loss = self.q_train(*([obs_n[self.agent_index]] + [action_indexs] + [target_u]))
        # 软更新目标网络
        self.q_update()

        return [q_loss, np.mean(target_u), np.mean(rew),  np.std(target_u)]

    # 获取神经网络的变量
    def get_scope_var(self, scope):
        return U.scope_vars(U.absolute_scope_name(scope))

    # 保存模型
    def save_model(self, file_name, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        var = self.get_scope_var(scope=self.name)
        if saver is None:
            saver = tf.train.Saver(var)
        saver.save(U.get_session(), file_name + self.name + '/')

    # 加载模型
    def load_model(self, file_name, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        var = self.get_scope_var(scope=self.name + '/')
        if saver is None:
            saver = tf.train.Saver(var)
        saver.restore(U.get_session(), file_name + self.name + '/')

    # 保存部分模型-如只保存Actor网络，且不需要目标网络
    def saver_actor_model(self, file_name, steps, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        with tf.variable_scope(self.name, reuse=None):
            var = self.get_scope_var(scope=self.actor_vars)
            if saver is None:
                saver = tf.train.Saver(var)
            saver.save(U.get_session(), file_name + self.name + '/actor/', global_step=steps)

    # 加载actor模型
    def load_actor_model(self, file_name, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        if saver is None:
            saver = tf.train.Saver()
        saver.restore(U.get_session(), file_name + self.name + '/actor/')
