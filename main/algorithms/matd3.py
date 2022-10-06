r"""MATD3"""

import os
import numpy as np
import tensorflow as tf
import tools.tf_util as U
import tensorflow.contrib.layers as layers

from algorithms import AgentTrainer
from tools.distributions import make_pdtype
from tools.replay_buffer import ReplayBuffer


def mlp_model(inputs, num_outputs, scope, reuse=False, num_units=64, activation_fn=None, rnn_cell=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inputs
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=activation_fn)
        return out


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    # polyak = 1.0 - 1e-2  # 这里建议小点
    polyak = 1.0 - 3e-4  # 够小了
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False,
            num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        act_tanh_sample = tf.tanh(act_sample)

        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q1 = q_func(q_input, 1, scope="q1_func", reuse=True, num_units=num_units)[:, 0]
        q2 = q_func(q_input, 1, scope="q2_func", reuse=True, num_units=num_units)[:, 0]
        q = tf.minimum(q1, q2)

        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_tanh_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        # 目标策略平滑：在policy输出前加噪声
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act_tanh_sample = tf.tanh(target_act_sample)
        # 加入clip的噪声 在policy输出前加噪声，不应该在Q网络输入前
        epsilon_sample = act_pdtype_n[p_index].pdfromflat(target_p).epsilon_sample()  # 噪声采样
        target_epsilon_act_sample = tf.clip_by_value(target_act_tanh_sample + epsilon_sample, -1, 1)  # clip到-1~1

        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_epsilon_act_sample)  # 带有噪声的目标策略输出

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}, "p_func"


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False,
            scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)

        q1 = q_func(q_input, 1, scope="q1_func", num_units=num_units)[:, 0]
        q2 = q_func(q_input, 1, scope="q2_func", num_units=num_units)[:, 0]

        q1_func_vars = U.scope_vars(U.absolute_scope_name("q1_func"))
        q2_func_vars = U.scope_vars(U.absolute_scope_name("q2_func"))

        q = tf.minimum(q1, q2)

        q1_loss = tf.reduce_mean(tf.square(q1 - target_ph))
        q2_loss = tf.reduce_mean(tf.square(q2 - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q1_reg = tf.reduce_mean(tf.square(q1))
        q2_reg = tf.reduce_mean(tf.square(q2))

        loss1 = q1_loss + 1e-3 * q1_reg
        loss2 = q2_loss + 1e-3 * q2_reg

        optimize1_expr = U.minimize_and_clip(optimizer, loss1, q1_func_vars, grad_norm_clipping)
        optimize2_expr = U.minimize_and_clip(optimizer, loss2, q2_func_vars, grad_norm_clipping)

        # Create callable functions
        train1 = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss1, updates=[optimize1_expr])
        train2 = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss2, updates=[optimize2_expr])
        min_q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network

        target_q1 = q_func(q_input, 1, scope="target_q1_func", num_units=num_units)[:, 0]
        target_q2 = q_func(q_input, 1, scope="target_q2_func", num_units=num_units)[:, 0]
        target_q = tf.minimum(target_q1, target_q2)

        target_q1_func_vars = U.scope_vars(U.absolute_scope_name("target_q1_func"))
        target_q2_func_vars = U.scope_vars(U.absolute_scope_name("target_q2_func"))

        update_target_q1 = make_update_exp(q1_func_vars, target_q1_func_vars)
        update_target_q2 = make_update_exp(q2_func_vars, target_q2_func_vars)

        min_target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return [train1, train2], [update_target_q1, update_target_q2], {'min_q_values': min_q_values,
                                                                        'min_target_q_values': min_target_q_values}


def get_scope_var(scope):
    return U.scope_vars(U.absolute_scope_name(scope))


class MATD3AgentTrainer(AgentTrainer):
    def __init__(self, name, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        super(MATD3AgentTrainer, self).__init__(name, obs_shape_n, act_space_n, agent_index, args, local_q_func)
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=mlp_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug, self.actor_vars = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=mlp_model,
            q_func=mlp_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.replay_buffer = ReplayBuffer(int(1e6))
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def action(self, obs):
        return self.act(obs[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, 1.0 if done or terminal else 0.0)

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        update_frequency = 10  # 更新频率
        critic_interval_actor = 3
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0

        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in
                                 range(self.n)]  # 在计算图里给目标策略网络带了噪声
            target_q_next = self.q_debug['min_target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q1_loss = self.q_train[0](*(obs_n + act_n + [target_q]))  # 噪声在target_q里

        min_q = self.q_debug["min_q_values"](*(obs_n + act_n))
        if t % (update_frequency * critic_interval_actor):  # 延迟更新actor critic2
            q2_loss = self.q_train[1](*(obs_n + act_n + [target_q]))
            # train p network
            p_loss = self.p_train(*(obs_n + act_n))
            self.p_update()
        else:
            q2_loss = [None]
            p_loss = [None]

        self.q_update[0]()
        if t % (update_frequency * critic_interval_actor):  # 延迟更新actor critic2
            self.q_update[1]()

        return [q1_loss, q2_loss, p_loss, np.mean(target_q), np.mean(min_q), np.mean(rew), np.std(target_q)]

    def save_model(self, file_name, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        var = get_scope_var(scope=self.name)
        if saver is None:
            saver = tf.train.Saver(var)
        saver.save(U.get_session(), file_name + self.name + '/')

    def load_model(self, file_name, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        var = get_scope_var(scope=self.name)
        if saver is None:
            saver = tf.train.Saver(var)
        saver.restore(U.get_session(), file_name + self.name + '/')

    def saver_actor_model(self, file_name, steps, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        with tf.variable_scope(self.name, reuse=None):
            var = get_scope_var(scope=self.actor_vars)
            if saver is None:
                saver = tf.train.Saver(var)
            saver.save(U.get_session(), file_name + self.name + '/actor/', global_step=steps)

    def load_actor_model(self, file_name, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        if saver is None:
            saver = tf.train.Saver()
        saver.restore(U.get_session(), file_name + self.name + '/actor/')
