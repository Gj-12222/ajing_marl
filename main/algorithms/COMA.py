"""
COMA algorithm
"""
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn

import numpy as np
import random
import tools.tf_util as U
import copy

from tools.distributions import make_pdtype
from algorithms import AgentTrainer

"""
RT = rT + γVT
for t = T-1, T-2,...,0:
    Gt =rt + γ(1-λ)Vt
    Rt =Gt + γλRt+1
Rt = TD(λ)

wait testing
"""
def normal_td_lamda(rewards, dones, gamma, lamda, target_q_next):
    pass

def td_lambda_discount_reward(rewards, dones, gamma, lamda_seq, gamma_seq, target_q_next):

    discount_reward = []
    T = rewards.shape[0]  # batch_size = T
    for t in range(T):
        r = 0  # n+t=T=batch_size
        for n in range(T - t):
            r = rewards[n + t] * np.sum(lamda_seq[n: T - t]) * (1 - dones[n + t]) + r * gamma  # 加done,回合可能提前结束
        discount_reward.append(r)
    for t in range(T - 1):  # t=0,1,...,T-1-1 相当于 t=1,2,...,T-1 时
        target_q_temp = 0.0
        for i in range(T - t - 1):
            # 计算每个Gt(λ)的累加 ∑ i=1~T-t-1 λ^(i-1) * γ^i * Q(st+i,at+i)
            target_q_temp += lamda_seq[i] * gamma_seq[1 + i] * target_q_next[t + i]
        discount_reward[t] = discount_reward[t] + target_q_temp
    """[::-1] 表示正序所有行，倒序所有列"""
    return np.array(discount_reward[::-1])

# MC-return
def monte_carlo_returns(rewards, gamma, lamda):
    max_epsiode_len = rewards.shape[0]
    G_t = []
    for t in range(max_epsiode_len):  # t= 1,2,...,T  index =  0,1,....,T-1
        monte_carlo_temp = 0.0
        for i in range(max_epsiode_len - t):  # Gt=rt+1 + γ^(1) * rt+2 +... = ∑i=1~T-t γ^(i-1) * rt+i
            monte_carlo_temp += gamma ** (i) * rewards[t + i]  # γ^(i-1) * rt+i
        monte_carlo_temp *= lamda ** (max_epsiode_len - t - 1)  # 再乘以 λ^(T-t-1) * ( ∑i=1~T-t γ^(i-1) * rt+i )
        G_t.append(monte_carlo_temp)
    return G_t

def soft_update_exp(values, target_values):
    tau = 1.0 - 5e-4  # 0.9995
    target_vars = []
    for var, target_var in zip(sorted(values, key=lambda v: v.name),
                               sorted(target_values, key=lambda v: v.name)):
        target_vars.append(target_var.assign(tau * target_var + (1 - tau) * var))
    target_vars = tf.group(*target_vars)
    return U.function([], [], updates=[target_vars])


def coma_actor_rnn(inputs, num_outputs, scope, num_units=256, reuse=False, rnn_cell=None, last_h_state=None,
                   time_step=4, activation_fn=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.shape.dims[1].value  # 最后一维
        out = tf.reshape(inputs, (-1, time_step, int(inputs_shape / time_step)))  # out = [?, time_step, obs_shape]
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)  # out = [?, time_step, 128]

        h_out, h_state = tf.nn.dynamic_rnn(rnn_cell, out, initial_state=last_h_state, dtype=tf.float32)
        last_out = h_out[:, -1]

        out = layers.fully_connected(last_out, num_outputs=num_outputs, activation_fn=activation_fn)

        return out, h_state

# Critic network
def coma_critic_mlp(inputs, num_outputs, scope, num_units=128, reuse=False, rnn_cell=None, activation_fn=None):
    with tf.variable_scope(scope, reuse=reuse):
        outs = inputs
        outs = layers.fully_connected(outs, num_outputs=num_units, activation_fn=tf.nn.relu)
        outs = layers.fully_connected(outs, num_outputs=num_units, activation_fn=tf.nn.relu)
        outs = layers.fully_connected(outs, num_outputs=num_outputs, activation_fn=activation_fn)
        return outs

def q_train(scope,
            obs_ph_n,
            act_space_n,
            q_func,
            q_index,
            optimizer,
            grad_norm_clipping,
            local_func=False,
            num_units=64,
            reuse=None):
    with tf.variable_scope(scope, reuse=reuse):

        act_pdtpye_n = [make_pdtype(act_shape) for act_shape in act_space_n]
        act_ph_n = [act_pdtpye_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]

        agent_onehot_ph_n = [tf.placeholder(tf.float32, [None, len(act_space_n)], name="agent_onehot" + str(i)) for i in
                             range(len(act_space_n))]

        target_ph = tf.placeholder(tf.float32, [None], name="target")
        q_inputs = tf.concat(obs_ph_n + act_ph_n + agent_onehot_ph_n, 1)
        if local_func:
            q_inputs = tf.concat(obs_ph_n[q_index] + act_ph_n[q_index] + agent_onehot_ph_n[p_index], 1)
        q = q_func(q_inputs, 1, scope="q_func", num_units=num_units, activation_fn=None)[:, 0]  # 取值
        q_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        loss = tf.reduce_mean(tf.square(target_ph - q))
        epsilon = tf.reduce_mean(tf.square(q))
        q_loss = loss + epsilon * 1e-3
        optimizer_exp_var = U.minimize_and_clip(optimizer, q_loss, q_vars, grad_norm_clipping)

        target_q = q_func(q_inputs, 1, scope="target_q_func", num_units=num_units, activation_fn=None)[:, 0]
        target_q_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        update_target_q_net = soft_update_exp(q_vars, target_q_vars)

        train = U.function(inputs=obs_ph_n + act_ph_n + agent_onehot_ph_n + [target_ph], outputs=q_loss,
                           updates=[optimizer_exp_var])
        q_values = U.function(inputs=obs_ph_n + act_ph_n + agent_onehot_ph_n, outputs=q)
        target_q_values = U.function(inputs=obs_ph_n + act_ph_n + agent_onehot_ph_n, outputs=target_q)

        return train, update_target_q_net, {'q_values': q_values, 'target_q_values': target_q_values}

def counterfactual_train(scope,
                         obs_ph_n,
                         act_space_n,
                         counterfactual_func,
                         c_index,
                         optimizer,
                         grad_norm_clipping,
                         local_func=False,
                         num_units=64,
                         reuse=None):
    with tf.variable_scope(scope, reuse=reuse):

        act_pdtpye_n = [make_pdtype(act_shape) for act_shape in act_space_n]
        act_other_ph_n = [act_pdtpye_n[i].sample_placeholder([None], name="other_action" + str(i)) for i in
                          range(len(act_space_n))]

        agent_onehot_ph = [tf.placeholder(tf.float32, [None, len(act_space_n)], name="agent_onehot" + str(i)) for i in
                           range(len(act_space_n))]

        counterfactual_target_ph = tf.placeholder(tf.float32, [None], name="counterfactual_target")
        c_inputs = tf.concat(obs_ph_n + act_other_ph_n + agent_onehot_ph, 1)
        if local_func:
            c_inputs = tf.concat(obs_ph_n[c_index] + act_other_ph_n[c_index] + agent_onehot_ph[p_index], 1)
        counterfactual = counterfactual_func(c_inputs, 1, scope="counterfactual_func", num_units=num_units,
                                             activation_fn=None)[:, 0]  # 取值
        counterfactual_vars = U.scope_vars(U.absolute_scope_name("counterfactual_func"))

        loss = tf.reduce_mean(tf.square(counterfactual_target_ph - counterfactual))
        epsilon = tf.reduce_mean(tf.square(counterfactual))
        counterfactual_loss = loss + epsilon * 1e-4
        optimizer_exp_var = U.minimize_and_clip(optimizer, counterfactual_loss, counterfactual_vars, grad_norm_clipping)

        target_counterfactual = counterfactual_func(c_inputs, 1, scope="target_counterfactual_func",
                                                    num_units=num_units, activation_fn=None)[:, 0]
        target_counterfactual_vars = U.scope_vars(U.absolute_scope_name("target_counterfactual_func"))

        update_target_counterfactual_net = soft_update_exp(counterfactual_vars, target_counterfactual_vars)

        train = U.function(inputs=obs_ph_n + act_other_ph_n + agent_onehot_ph + [counterfactual_target_ph],
                           outputs=counterfactual_loss, updates=[optimizer_exp_var])
        counterfactual_values = U.function(inputs=obs_ph_n + act_other_ph_n + agent_onehot_ph, outputs=counterfactual)
        target_counterfactual_values = U.function(inputs=obs_ph_n + act_other_ph_n + agent_onehot_ph,
                                                  outputs=target_counterfactual)

        return train, update_target_counterfactual_net, {'counterfactual_values': counterfactual_values, 'target_counterfactual_values': target_counterfactual_values}

def p_train(scope,
            obs_ph_n,
            warpper_obs_ph,
            act_space_n,
            p_index,
            p_func,
            q_func,
            b_func,
            optimizer,
            grad_norm_clipping,
            local_func=False,
            num_units=64,
            rnn_time_step=4,
            obs_shape=64,
            reuse=None):
    with tf.variable_scope(scope, reuse=reuse):

        act_pdtpye_n = [make_pdtype(act_shape) for act_shape in act_space_n]
        act_ph_n = [act_pdtpye_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]

        act_ph_other = [act_pdtpye_n[i].sample_placeholder([None], name="other_action" + str(i)) for i in
                        range(len(act_space_n))]

        agent_onehot_ph = [tf.placeholder(tf.float32, [None, len(act_space_n)], name="agent_onehot" + str(i)) for i in
                           range(len(act_space_n))]
        # GRU_cell
        gru_cell = rnn.GRUCell(num_units=num_units,
                               kernel_initializer=tf.random_normal_initializer(-0.1, 0.1, seed=2),
                               bias_initializer=tf.random_normal_initializer(-0.1, 0.1, seed=2),
                               name="policy_gru_cell")

        p_inputs = warpper_obs_ph[:, :obs_shape]
        p, hidden_p = p_func(p_inputs, int(act_pdtpye_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units,
                             rnn_cell=gru_cell, last_h_state=warpper_obs_ph[:, obs_shape:], time_step=rnn_time_step,
                             activation_fn=None)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        act_pd = act_pdtpye_n[p_index].pdfromflat(p)
        act_sample = act_pd.sample()

        act_sample = tf.tanh(act_sample)

        q_inputs = tf.concat(obs_ph_n + act_ph_n + agent_onehot_ph, 1)
        if local_func:
            q_inputs = tf.concat(obs_ph_n[p_index] + act_ph_n[p_index] + agent_onehot_ph[p_index], 1)
        q = q_func(q_inputs, 1, scope="q_func", reuse=True, num_units=num_units, activation_fn=None)[:, 0]
        # co_baseline 网络
        b_inputs = tf.concat(obs_ph_n + act_ph_other + agent_onehot_ph, 1)
        if local_func:
            b_inputs = tf.concat(obs_ph_n[p_index] + act_ph_n[p_index] + agent_onehot_ph[p_index], 1)
        counterfactual = b_func(b_inputs, 1, scope="counterfactual_func", reuse=True, num_units=num_units,
                                activation_fn=None)[:, 0]

        A_loss = q - counterfactual  # A_loss = [t_size, 1]
        pg_loss = - tf.reduce_mean(A_loss * tf.reduce_prod(act_sample, axis=1))
        epsilon = tf.reduce_mean(tf.square(act_pd.flatparam()))
        loss = pg_loss + epsilon * 1e-3

        optimizer_p_vars = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # (选用) target_p_network
        target_gru_cell = rnn.GRUCell(num_units=num_units,
                                      kernel_initializer=tf.random_normal_initializer(-0.1, 0.1, seed=2),
                                      bias_initializer=tf.random_normal_initializer(-0.1, 0.1, seed=2),
                                      name="target_policy_gru_cell")
        target_p, target_hidden_p = p_func(p_inputs, int(act_pdtpye_n[p_index].param_shape()[0]), scope="target_p_func",
                                           num_units=num_units,
                                           rnn_cell=target_gru_cell, last_h_state=warpper_obs_ph[:, obs_shape:],
                                           time_step=rnn_time_step, activation_fn=None)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))

        target_act_sample = act_pdtpye_n[p_index].pdfromflat(target_p).sample()
        target_act_sample = tf.tanh(target_act_sample)

        target_p_soft_update = soft_update_exp(p_func_vars, target_p_func_vars)

        act = U.function(inputs=[warpper_obs_ph], outputs=[act_sample, hidden_p])
        p_prob = U.function(inputs=[warpper_obs_ph], outputs=p)
        target_p_prob = U.function(inputs=[warpper_obs_ph], outputs=target_p)
        target_act = U.function(inputs=[warpper_obs_ph], outputs=[target_act_sample, target_hidden_p])
        train = U.function(inputs=obs_ph_n + act_ph_n + act_ph_other + agent_onehot_ph + [warpper_obs_ph], outputs=loss,updates=[optimizer_p_vars])

        return act, train, target_p_soft_update, {'p_prob': p_prob, 'target_p_prob': target_p_prob, 'target_act': target_act}, "p_func"

class COMAAgentTrainer(AgentTrainer):
    def __init__(self, name,
                 actor_critic_model,
                 obs_space_n,
                 act_space_n,
                 agent_index,
                 args,
                 local_q_func=False):
        self.name = name
        self.n = len(obs_space_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_space_n[i], name="observation" + str(i)).get())
        warpper_obs_tuple_to_list = list(obs_space_n[agent_index])[0] * 4
        warpper_obs_ph = U.BatchInput([warpper_obs_tuple_to_list + args.num_units],
                                      name="warpper_obs_and_last_hidden_state" + str(self.agent_index)).get()
        self.c_train, self.c_target_update, self.c_target_values = counterfactual_train(
            scope=self.name,
            obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            c_index=agent_index,
            counterfactual_func=coma_critic_mlp,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr * 0.05),
            grad_norm_clipping=0.5,
            local_func=local_q_func,
            num_units=args.num_units)

        self.q_train, self.q_targrt_update, self.q_target_values = q_train(
            scope=self.name,
            obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=coma_critic_mlp,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr * 0.05),
            grad_norm_clipping=0.5,
            local_func=local_q_func,
            num_units=args.num_units)

        self.act, self.p_train, self.p_target_update, self.p_target_act, self.actor_vars = p_train(
            scope=self.name,
            obs_ph_n=obs_ph_n,
            warpper_obs_ph=warpper_obs_ph,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=coma_actor_rnn,
            q_func=coma_critic_mlp,
            b_func=coma_critic_mlp,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr * 0.1),
            grad_norm_clipping=0.5,
            local_func=local_q_func,
            num_units=args.num_units,
            rnn_time_step=args.rnn_time_step,
            obs_shape=warpper_obs_tuple_to_list)


        self.on_policy_replay_buffer = ReplayBuffer(self.args.max_episode_len)

        self.max_replay_buffer_len = args.max_episode_len
        self.replay_sample_index = None

        """加入 hidden_state 和 history_states，组成 time-step 为4的轨迹片段 """
        self.hidden_state = None
        self.history_states = None

    def action(self, obs):
        # 取前 3个 step 的 state + 当前 state
        if self.hidden_state is None:
            self.hidden_state = np.zeros((self.args.num_units,))
        self.warpper(obs) # [time_step, obs_dims]
        actor_inputs_feature = np.concatenate([self.history_states] + [self.hidden_state])
        [act, act_hidden] = self.act(actor_inputs_feature[None])
        self.hidden_state = act_hidden[0]
        return act[0]

    def warpper(self, obs_t):
        if self.history_states is None:
            self.history_states = np.zeros_like(np.array([obs_t] * self.args.rnn_time_step))
            self.history_states[-1] = copy.deepcopy(obs_t)
            self.history_states = self.history_states.reshape(-1, )
        else:
            temp_warpper = self.history_states.reshape(self.args.rnn_time_step, -1)
            temp_warpper = temp_warpper[1:]
            self.history_states = np.concatenate([temp_warpper, obs_t.reshape(1, -1)], axis=0).reshape(-1, )  # 在第0维度拼接

    def learn_warpper(self, agent_obs):
        agent_warpper_obs = []  #
        init_obs_zero = np.zeros_like(agent_obs[0])
        temp_history_state = np.concatenate([init_obs_zero, init_obs_zero, init_obs_zero, agent_obs[0]], axis=0)
        for i in range(len(agent_obs)):  # batch
            agent_warpper_obs += [temp_history_state.reshape(-1, )]
            temp_history_state = temp_history_state.reshape(self.args.rnn_time_step, -1)
            temp_history_state = temp_history_state[1:]
            temp_history_state = np.concatenate([temp_history_state, agent_obs[i].reshape(1, -1)],
                                                axis=0)
        agent_warpper_obs += [temp_history_state.reshape(-1, )]
        return agent_warpper_obs


    def experience(self, obs, act, rew, obs_next, done, done_all):
        self.on_policy_replay_buffer.add(obs, act, rew, obs_next, 1.0 if done == True or done_all == True else 0.0)

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, step):

        if len(self.on_policy_replay_buffer) < self.max_replay_buffer_len: return
        if not step % self.args.max_episode_len == 0:  return

        obs_n, act_n, obs_next_n = [], [], []
        self.replay_sample_index = self.on_policy_replay_buffer.on_policy_make_indexs_sample(self.args.max_episode_len)  # 在线采样
        indexs = self.replay_sample_index

        for i in range(self.n):
            obs, act, _, obs_next, _ = \
                agents[i].on_policy_replay_buffer.on_poicy_sampe(indexs)
            obs_n.append(obs)
            act_n.append(act)
            obs_next_n.append(obs_next)

        _, _, rew, _, mask = self.on_policy_replay_buffer.on_poicy_sampe(indexs)

        agent_obs = obs_n[self.agent_index]
        agent_warpper_obs = self.learn_warpper(agent_obs)
        # burn-in
        learn_hidden_state = np.zeros((self.args.num_units,))
        obs_warpper_state = []
        for i in range(len(act_n[self.agent_index])):
            warpper_obs = np.concatenate([agent_warpper_obs[i]] + [learn_hidden_state])
            [_, hidden_state] = self.act(warpper_obs[None])
            learn_hidden_state = hidden_state[0]
            obs_warpper_state.append(warpper_obs)
        obs_warpper_state = np.array(obs_warpper_state)
        # A = Q - counterfactual
        agent_onehot_n = [np.zeros((1, self.n), dtype=np.float32).repeat(len(obs_n[self.agent_index]), axis=0) for _ in
                          range(self.n)]
        for i, agent_onehot in enumerate(agent_onehot_n):
            agent_onehot[:, i] = 1.0

        num_sample = 1
        G_t_lamda = 0.0
        for i in range(num_sample):
            target_act_next_n = []
            for i, agent in enumerate(agents):
                if hasattr(agent, "history_states"):
                    agent_obs_next = obs_next_n[self.agent_index]
                    agent_warpper_obs_next = self.learn_warpper(agent_obs_next)
                    learn_hidden_target_stat = np.zeros((self.args.num_units,))
                    target_act_next = []
                    for i in range(len(agent_obs_next)):
                        agent_warpper_next = np.concatenate([agent_warpper_obs_next[i]] + [learn_hidden_target_stat])
                        [target_act, hidden_next_state] = agent.p_target_act['target_act'](agent_warpper_next[None])
                        learn_hidden_target_stat = hidden_next_state[0]
                        target_act_next.append(target_act[0])
                    target_act_next = np.array(target_act_next, copy=False)
                    target_act_next_n.append(target_act_next)
                else:
                    target_act_next_n.append(agent.p_target_act['target_act'](obs_next_n[i]))
            target_q_next = (1.0 - mask) * self.q_target_values['target_q_values'](*(obs_next_n + target_act_next_n + agent_onehot_n))
            gamma_seq = np.logspace(1, rew.shape[0], num=rew.shape[0], endpoint=True,base=self.args.gamma)
            tdlamda_seq = np.logspace(0, rew.shape[0] - 1, num=rew.shape[0], endpoint=True,base=self.args.tdlambda)
            G_t_n = (1 - self.args.tdlambda) * \
                    td_lambda_discount_reward(rew, mask, gamma=self.args.gamma, lamda_seq=tdlamda_seq, gamma_seq=gamma_seq, target_q_next=target_q_next)
            G_t_n += monte_carlo_returns(rew, self.args.gamma, self.args.tdlambda)
            G_t_lamda += G_t_n
        G_t_lamda = G_t_lamda / num_sample

        target_conuterfactual = 0.0
        for i in range(num_sample):

            target_act_other_next_n = copy.deepcopy(target_act_next_n)
            target_act_other_next_n[self.agent_index] = np.zeros_like(target_act_other_next_n[self.agent_index])

            target_counterfactual_next = (1.0 - mask) * self.c_target_values['target_counterfactual_values'](
                *(obs_next_n + target_act_other_next_n + agent_onehot_n))  # [200, 1]
            target_conuterfactual += rew + self.args.gamma * (1 - mask) * target_counterfactual_next
        target_conuterfactual /= num_sample

        act_other_n = copy.deepcopy(act_n)
        act_other_n[self.agent_index] = np.zeros_like(act_other_n[self.agent_index])

        c_loss = self.c_train(*(obs_n + act_other_n + agent_onehot_n + [target_conuterfactual]))

        q_loss = self.q_train(*(obs_n + act_n + agent_onehot_n + [G_t_lamda]))

        self.q_targrt_update()
        self.c_target_update()

        p_loss = self.p_train(*(obs_n + act_n + act_other_n + agent_onehot_n + [obs_warpper_state]))

        self.p_target_update()

        return [q_loss, p_loss, c_loss, np.mean(G_t), np.mean(rew), np.mean(target_q_next), np.std(G_t)]

    def get_scope_var(self, scope):
        return U.scope_vars(U.absolute_scope_name(scope))

    def save_model(self, file_name, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        var = self.get_scope_var(scope=self.name)
        if saver is None:
            saver = tf.train.Saver(var)
        saver.save(U.get_session(), file_name + self.name + '/')

    def load_model(self, file_name, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        var = self.get_scope_var(scope=self.name + '/')
        if saver is None:
            saver = tf.train.Saver(var)
        saver.restore(U.get_session(), file_name + self.name + '/')

    def saver_actor_model(self, file_name, steps, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        with tf.variable_scope(self.name, reuse=None):
            var = self.get_scope_var(scope=self.actor_vars)
            if saver is None:
                saver = tf.train.Saver(var)
            saver.save(U.get_session(), file_name + self.name + '/actor/', global_step=steps)

    def load_actor_model(self, file_name, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        if saver is None:
            saver = tf.train.Saver()
        saver.restore(U.get_session(), file_name + self.name + '/actor/')

""" ReplayBuffer"""
class  ReplayBuffer(object):

    def __init__(self, size):
        self._storage = []
        self._maxsize = int(size)
        self._next_index = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_index = 0

    def add(self, obs_t, action, reward, obs_t1, done_mask):  # replay buffer length not more than one episode
        data = (obs_t, action, reward, obs_t1, done_mask)
        if self._next_index >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_index] = data
        self._next_index = (self._next_index + 1) % self._maxsize

    # on-policy-sample
    def on_policy_make_indexs_sample(self, step):
        return [step]

    def _on_policy_lambda_encode_sampe(self, indexs):
        obs, act, rew, obs_next, mask = [], [], [], [], []
        for i in range(indexs[0]):
            data = self._storage[i]  # 正序就行
            obs_temp, act_temp, rew_temp, obs_next_temp, mask_temp = data
            obs.append(np.array(obs_temp, copy=False))
            act.append(np.array(act_temp, copy=False))
            rew.append(np.array(rew_temp))
            obs_next.append(np.array(obs_next_temp, copy=False))
            mask.append(np.array(mask_temp))
        return np.array(obs), np.array(act), np.array(rew), np.array(obs_next), np.array(mask)

    def on_poicy_sampe(self, indexs):
        return self._on_policy_lambda_encode_sampe(indexs)

    # off-policy-sample  n-step TD(λ)
    def make_sort_sample_index(self, batch_size):
        index_sample = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        list.sort(index_sample)
        return index_sample

    def TD_lambda_sample(self, indexs, step):
        return self._off_policy_lambda_encode_sample(indexs, step)

    def _off_policy_lambda_encode_sample(self, indexs, step):
        obs, act, rew, obs_next, done_mask = [], [], [], [], []
        obs_dim, act_dim, rew_dim, obs_next_dim, done_mask_dim = self._storage[0]

        obs_deep_dim = np.zeros_like(obs_dim)
        act_deep_dim = np.zeros_like(act_dim)
        for i in range(len(indexs)):
            obs_temp, act_temp, rew_temp, obs_next_temp, done_mask_temp = [], [], [], [], []
            for j in range(step):
                if indexs[i] + j > len(self._storage) - 1:
                    break
                data = self._storage[indexs[i] + j]
                obs_t, act_t, rew_t, obs_t1, done_mask_t = data
                obs_temp.append(np.array(obs_t, copy=False))
                act_temp.append(np.array(act_t, copy=False))
                rew_temp.append(rew_t)
                obs_next_temp.append(np.array(obs_t1, copy=False))
                done_mask_temp.append(done_mask_t)
                if done_mask_t == 1.0:  # beyond max_buffer_list
                    break
                if i < len(indexs) - 1:
                    if indexs[i] + j + 1 >= indexs[i + 1]:
                        break
            if len(obs_temp) < step and len(indexs) >= 2:
                for i in range(step - len(obs_temp)):
                    obs_temp.append(np.array(obs_deep_dim))
                    act_temp.append(np.array(act_deep_dim))
                    rew_temp.append(0.0)
                    obs_next_temp.append(np.array(obs_deep_dim))
                    done_mask_temp.append(1.0)

            obs.append(np.array(obs_temp, copy=False))
            act.append(np.array(act_temp, copy=False))
            rew.append(np.array(rew_temp, copy=False))
            obs_next.append(np.array(obs_next_temp, copy=False))
            done_mask.append(np.array(done_mask_temp, copy=False))
        return np.array(obs), np.array(act), np.array(rew), np.array(obs_next), np.array(done_mask)

    def make_sample_index(self, batch_size):
        index_sample = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return index_sample

    def make_last_jam_index(self, batch_size):
        indexs = [(self._next_index - i - 1) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(indexs)
        return indexs

    def collect(self):
        return self.sample(-1)

    def sample(self, indexs):
        return self._encode_sample(indexs)

    def _encode_sample(self, indexs):
        obs, act, rew, obs_next, done_mask = [], [], [], [], []
        for i in indexs:
            data = self._storage[i]
            obs_t, act_t, rew_t, obs_t1, done_mask_t = data
            obs.append(np.array(obs_t, copy=False))
            act.append(np.array(act_t, copy=False))
            rew.append(rew_t)
            obs_next.append(np.array(obs_t1, copy=False))
            done_mask.append(done_mask_t)
        return np.array(obs), np.array(act), np.array(rew), np.array(obs_next), np.array(done_mask)

