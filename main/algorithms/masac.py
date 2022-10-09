r'''
core code for the MASAC algorithm
（MASAC算法的核心代码）
自适应alpha (未实现)
'''

import os
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tools.tf_util as U
from tools.distributions import make_pdtype
from tools.replay_buffer import ReplayBuffer
from algorithms import AgentTrainer

# Actor-critic
def mlp_model(input,num_outputs,scope,  reuse=False, num_units=64, rnn_cell=None,activation_fn=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=activation_fn)
        return out


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

#  参数更新函数
def make_update_exp(vals, target_vals):
    polyak = 1.0 - (5*1e-3)
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))   #
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

# log_prob
def log_gaussian_policy(act_resample, act_mu, act_logstd, epsilon):
    # act_resample.shape = [ batch_size , action.shape ]
    # log_s = -0.5 *[( x - u)/ (e^log var)]^2 + 2*(log var) + 2*logπ # 高斯分布对数形式
    log_normal_sum = -0.5 * (((act_resample - act_mu) / (tf.exp(act_logstd) + epsilon)) ** 2 + 2 * act_logstd + np.log(2 * np.pi))
    # return [batch_size, 1]
    return tf.reduce_mean(log_normal_sum, axis=1)

# 修正 log_prob
def euler_transformation(logp_act_resample, act_resample):
    logp_act_resample -= tf.reduce_mean(2 * (np.log(2) - act_resample - tf.nn.softplus(-2 * act_resample)), axis=1)
    return logp_act_resample

# alpha
def masac_alpha_log(make_obs_ph_n,
                    act_space_n,
                    p_index,
                    p_func,
                    q_func,
                    optimizer,
                    grad_norm_clipping=None,
                    local_q_func=False,
                    deterministic=False,  # 固定alpha
                    target_entropy=1,  # 目标熵
                    init_alpha_log=np.log(0.2),  # 初始化alpha_log
                    num_units=64,
                    scope="trainer",
                    reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if deterministic:
            alpha_log = init_alpha_log
            np_alpha_log = np.array([init_alpha_log])
            return np_alpha_log, None
        else:
            target_entropy *= np.log(act_space_n[p_index].shape)  # agent对应的动作维度？还是 所有agent的动作维度？
            alpha_log = tf.Variable(init_alpha_log,name="alpha",dtype=tf.float32)
            alpha = tf.convert_to_tensor(tf.exp(alpha_log))
            act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
            obs_ph_n = make_obs_ph_n

            p_input = obs_ph_n[p_index]

            act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in
                        range(len(act_space_n))]
            p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units,
                       activation_fn=None)

            act_pd = act_pdtype_n[p_index].pdfromflat(p)
            actpd_mu = act_pd.mean
            actpd_logstd = act_pd.logstd

            act_resample = act_pd.reparameterization()
            act_resample = tf.tanh(act_resample)

            logp_act_resample = log_gaussian_policy(act_resample, actpd_mu, actpd_logstd, epsilon=1e-8)
            logp_act_resample = euler_transformation(logp_act_resample, act_resample)

            alpha_losses = -(alpha * tf.stop_gradient(logp_act_resample - target_entropy))
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)
            optimize_expr = U.minimize_and_clip(optimizer, alpha_loss, alpha_log, grad_norm_clipping)

            train = U.function(inputs=obs_ph_n + act_ph_n, outputs=alpha_loss, updates=[optimize_expr])
            alpha_values = U.function(inputs=obs_ph_n + act_ph_n, outputs=alpha_log)
            return alpha, train

# policy
def masac_p_train(make_obs_ph_n,
                  act_space_n,
                  p_index,
                  p_func,
                  q_func,
                  optimizer,
                  grad_norm_clipping=None,
                  fix=False,
                  local_q_func=False,
                  num_units=64,
                  scope="trainer",
                  reuse=None):
    with tf.variable_scope(scope, reuse=reuse):

        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]
        if fix:  # 固定α
            alpha = tf.placeholder(tf.float32, [None],name="alpha")
            p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), reuse=False, scope="p_func",
                       num_units=num_units,
                       activation_fn=None)
        else: # 不固定
            alpha = tf.get_variable("alpha",shape=[None,],dtype=tf.float32)
            p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), reuse=True, scope="p_func",
                       num_units=num_units,
                       activation_fn=None)
        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        actpd_mu = act_pd.mean
        actpd_logstd = act_pd.logstd # (-20, 2)
        act_resample = act_pd.reparameterization()
        act_resample = tf.tanh(act_resample)

        logp_act_resample = log_gaussian_policy(act_resample, actpd_mu, actpd_logstd)
        logp_act_resample = euler_transformation(logp_act_resample, act_resample)
        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_resample
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q1 = q_func(q_input, 1, scope="q1_func", reuse=True, num_units=num_units)[:, 0]
        q2 = q_func(q_input, 1, scope="q2_func", reuse=True, num_units=num_units)[:, 0]

        q = tf.minimum(q1, q2)
        pg_loss = -tf.reduce_mean(q - alpha * logp_act_resample)
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
        loss = pg_loss + p_reg * 1e-3

        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        train = U.function(inputs=obs_ph_n + act_ph_n + [alpha], outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_resample)
        logp_act = U.function(inputs=[obs_ph_n[p_index]], outputs=logp_act_resample)
        p_values = U.function([obs_ph_n[p_index]], p)

        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=num_units, activation_fn=None)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))

        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_actpd_mu = act_pdtype_n[p_index].pdfromflat(target_p).mean
        target_actpd_logstd = act_pdtype_n[p_index].pdfromflat(target_p).logstd
        target_act_resample = act_pdtype_n[p_index].pdfromflat(target_p).reparameterization()
        target_logp_act_resample = log_gaussian_policy(target_act_resample, target_actpd_mu,target_actpd_logstd)
        target_logp_act_resample = euler_transformation(target_logp_act_resample, target_act_resample)
        target_act_mu = tf.tanh(target_actpd_mu)
        target_act_resample = tf.tanh(target_act_resample)
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_resample)
        target_logp_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_logp_act_resample)
        target_p = U.function([obs_ph_n[p_index]], target_p)
        return act, logp_act, train, update_target_p, {'target_act': target_act, 'target_logp_act': target_logp_act}, 'p_func'

def masac_q_train(make_obs_ph_n,
                  act_space_n,
                  q_index,
                  q_func,
                  optimizer,
                  grad_norm_clipping=None,
                  local_q_func=False,
                  scope="trainer",
                  reuse=None,
                  num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
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

        min_q = tf.minimum(q1, q2)
        q1_loss = tf.reduce_mean(tf.square(q1 - target_ph))
        q2_loss = tf.reduce_mean(tf.square(q2 - target_ph))
        # viscosity solution to Bellman differential equation in place of an initial condition
        q1_reg = tf.reduce_mean(tf.square(q1))
        q2_reg = tf.reduce_mean(tf.square(q2))
        loss1 = q1_loss + 1e-3 * q1_reg
        loss2 = q2_loss + 1e-3 * q2_reg

        optimize1_expr = U.minimize_and_clip(optimizer, loss1, q1_func_vars, grad_norm_clipping)
        optimize2_expr = U.minimize_and_clip(optimizer, loss2, q2_func_vars, grad_norm_clipping)
        train1 = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss1, updates=[optimize1_expr])
        train2 = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss2, updates=[optimize2_expr])

        min_q_values = U.function(obs_ph_n + act_ph_n, min_q)

        target_q1 = q_func(q_input, 1, scope="target_q1_func", num_units=num_units)[:, 0]
        target_q2 = q_func(q_input, 1, scope="target_q2_func", num_units=num_units)[:, 0]

        target_q1_func_vars = U.scope_vars(U.absolute_scope_name("target_q1_func"))
        target_q2_func_vars = U.scope_vars(U.absolute_scope_name("target_q2_func"))

        update_target_q1 = make_update_exp(q1_func_vars, target_q1_func_vars)
        update_target_q2 = make_update_exp(q2_func_vars, target_q2_func_vars)

        min_target_q = tf.minimum(target_q1, target_q2)
        min_target_q_values = U.function(obs_ph_n + act_ph_n, min_target_q)
        return [train1, train2], [update_target_q1, update_target_q2], {'min_q_values': min_q_values, 'min_target_q_values': min_target_q_values},

class MASACAgentTrainer(AgentTrainer):
    def __init__(self,name, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        # target_entorpy目标熵（未用到）
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())
        """① alpha自适应更新(暂缺)"""
        self.alpha_log, self.alpha_loss = masac_alpha_log(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=mlp_model,
            q_func=mlp_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr*0.03),
            grad_norm_clipping=0.5,
            deterministic=args.fix_alpha,  # 固定alpha
            target_entropy=1,
            init_alpha_log=np.log(args.init_alpha),
            local_q_func=local_q_func,
            num_units=args.num_units
        )

        self.q_train, self.q_update, self.q_debug = masac_q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=mlp_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr * 3 * 0.1),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # 用p_train建立了 策略训练网络（policy tarining network）含有（策略网络，目标策略网络）
        self.act, self.act_prob, self.p_train, self.p_update, self.p_debug, self.actor_vars = masac_p_train(
            scope=self.name,  # agent的序号 agent%d
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=mlp_model,
            q_func=mlp_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr * 3 * 0.5),
            grad_norm_clipping=0.5,
            fix=args.fix_alpha,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer 创建经验缓冲区-记忆库
        self.replay_buffer = ReplayBuffer(5e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None


    def action(self, obs):
        return self.act(obs[None])[0]

    # 将经验（S,A,R,S'）存入经验记忆库replay_buffer
    def experience(self, obs, act, rew, new_obs, done, terminal):

        self.replay_buffer.add(obs, act, rew, new_obs, float(1.0)if done==True or terminal==True else float(0.0))

    def preupdate(self):
        self.replay_sample_index = None

    # 网络更新
    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return
        if not t % 10 == 0:
            return
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        # buffer_data
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        num_sample = 1
        Y_target = 0.0

        if self.args.fix_alpha == True:
            alpha = np.exp(self.alpha_log)
        else:
            alpha = np.exp(self.alpha_log(*(obs_n + act_n)))
        for i in range(num_sample):
            # Y = r + (1-d)γ(min_q - alpha*logp_act)
            if self.args.target_actor: # use tar_policy
                target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
                target_logp_act_next_n = agents[self.agent_index].p_debug['target_logp_act'](obs_next_n[self.agent_index])
                target_min_q_next = self.q_debug['min_target_q_values'](*(obs_next_n + target_act_next_n))
            else:
                act_next_n = [agents[i].act(obs_next_n[i]) for i in range(self.n)]
                target_logp_act_next_n = agents[self.agent_index].act_prob(obs_next_n[self.agent_index])
                target_min_q_next = self.q_debug['min_target_q_values'](*(obs_next_n + act_next_n))
            Y_target += rew + (1 - done) * self.args.gamma * (target_min_q_next - alpha * target_logp_act_next_n)  # alpha需要降维取平均
        Y_target /= num_sample
        q1_loss = self.q_train[0](*(obs_n + act_n + [Y_target]))
        q2_loss = self.q_train[1](*(obs_n + act_n + [Y_target]))

        p_loss = self.p_train(*(obs_n + act_n + [alpha]))
        if not self.args.fix_alpha:
            log_alpha_loss = self.alpha_loss(*(obs_n + act_n))
        else:
            log_alpha_loss = [None]

        self.p_update()
        for i in range(2):
            self.q_update[i]()
        return [q1_loss, q2_loss, p_loss, np.mean(Y_target), np.mean(rew), np.std(Y_target)]

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






























