# ID3DQN算法核心

# 传统的DQN应用DTDE框架得到的ID3DQN
#  id3qn 适用--> 连续状态-离散动作

from algorithms import AgentTrainer
import tools.tf_util as U
import tensorflow as tf
import tensorflow.contrib.layers as layers

from tools.distributions import make_pdtype
from tools.replay_buffer import ReplayBuffer
import numpy as np

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None, activation_fn=None):
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

def av_mlp_nn(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None, activation_fn=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out_v = layers.fully_connected(out, num_outputs=1, activation_fn=activation_fn)
        out_a_prb = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=activation_fn)
        return out_v, out_a_prb


def q_train(make_obs_n,
                act_space_n,
                a_v_model,
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

        action_index_ph = tf.placeholder(dtype=tf.int64, shape=[None,1], name="action_index")
        obs_ph_n = make_obs_n
        input_data = obs_ph_n[agent_index]
        v, a_prb = a_v_model(input_data,int(act_pdtype_n[agent_index].param_shape()[0]),scope="av_model", num_units=num_units,activation_fn=None)
        av_vars = U.scope_vars(U.absolute_scope_name("av_model"))
        q_target_temp = tf.placeholder(dtype=tf.float32, shape=[None],name="q_s_a")

        a_predit = tf.gather(a_prb, indices=action_index_ph, axis=1,batch_dims=1)
        # 计算 mean(A(s,a))
        a_mean = tf.reduce_mean(a_prb, axis=-1, keepdims=True)
        # 计算 q = v + a - mean(A(s,a))
        q_predit = v + a_predit - a_mean # 计算q_prb
        # 删除1维度
        q_predit = tf.squeeze(q_predit, axis=-1)
        q_loss = - tf.reduce_mean(tf.square(q_predit - q_target_temp)) # [1]
        epilon = tf.reduce_mean(a_prb)
        loss = q_loss + 1e-3 * epilon
        # update
        optimizers_var = U.minimize_and_clip(optimizer=optimizer,objective=loss,var_list=av_vars,clip_val=gram_normal_clipping)
        # 用计算图更新
        a_probs = U.function(inputs=[obs_ph_n[agent_index]], outputs=a_prb)
        train = U.function(inputs=[obs_ph_n[agent_index]] + [action_index_ph] + [q_target_temp],outputs=loss,updates=[optimizers_var])
        target_v, target_a_prb = a_v_model(input_data, int(act_pdtype_n[agent_index].param_shape()[0]), scope="av_target_model", num_units=num_units, activation_fn=None)
        # 更新网络
        av_target_vars = U.scope_vars(U.absolute_scope_name("av_target_model"))
        av_target_soft_update = soft_update_vars(q_vars=av_vars,q_target_vars=av_target_vars,tau=tau)
        av_target_values = U.function(inputs=[obs_ph_n[agent_index]],outputs=[target_v, target_a_prb])

        return a_probs, train, av_target_soft_update, av_target_values

class ID3QNAgentTrainer(AgentTrainer):
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

        self.a_probs, self.av_train, self.av_update, self.av_target_values = q_train(scope=name,
                make_obs_n=obs_ph_n,
                act_space_n=act_space_n,
                a_v_model = mlp_model,
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
            a_probs = self.a_probs(obs[None])[0]
            actions = [np.argmax(a_probs)]
        return actions

    # 加入记忆库
    def experience(self, obs, act, rew, obs_next, done, terminal):
        self.dqn_replay_buffer.add(obs, act, rew, obs_next, float(done and terminal))

    # 清空索引
    def preupdate(self):
        self.replay_sample_index = None
    # 更新网络
    def update(self, agents, t):
        if len(self.dqn_replay_buffer) < self.dqn_sample:return
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
            """计算double dueling DQN"""
            [v_target, a_target_next_prb] = self.av_target_values(obs_next_n[self.agent_index])
            v_target = np.squeeze(v_target)
            a_next_prb = self.a_probs(obs_next_n[self.agent_index])
            action_next_max_indexs = np.argmax(a_next_prb, axis=1)
            a_target_next = a_target_next_prb[np.arange(len(action_next_max_indexs)),action_next_max_indexs]
            a_target_next_mean = np.mean(a_target_next_prb, axis=-1,keepdims=False)
            q_target_max = v_target + a_target_next - a_target_next_mean
            # 计算Q值 Q = R + gamma * (1 - done——终止条件) * Q
            target_u += rew + self.args.gamma * (1 - done) * q_target_max
        target_u /= num_sample
        # 计算损失函数
        av_loss = self.av_train(*([obs_n[self.agent_index]] + [self_act] + [target_u]))
        self.av_update()
        return [av_loss, np.mean(target_u), np.mean(rew),  np.std(target_u)]

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


