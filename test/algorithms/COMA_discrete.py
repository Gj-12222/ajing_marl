# COMA-discrete

"""目前，看起来运行成功了，但是，还需要实验验证是否正确"""

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn
import numpy as np
import random
import copy

import tools.tf_util as U
from tools.distributions import make_pdtype
from algorithms import AgentTrainer

"""
RT = rT + γVT
for t = T-1, T-2,...,0:
    Gt =rt + γ(1-λ)Vt
    Rt =Gt + γλRt+1
结果：Rt = TD(λ)
根据TD(λ)的计算公式编写的标准TD(λ)函数
"""

def normal_td_lamda(rewards, dones, gamma, lamda, target_q_next):
    pass

"""
###############################################
#################③额外： TD(λ)的迭代计算式
###############################################
"""
# n-step TD(λ)计算每step的折扣预期回报- G(n)t  t = T-1,T-2,...,1   n = 1,2,...,T-1

def td_lambda_discount_reward(rewards, dones, gamma, lamda_seq, gamma_seq,
                              target_q_next):  # reward.shape(0) = [t_size,1]
    discount_reward = []
    # c = rewards.shape[0]  # batch_size
    # lamda = lamda_seq[1]  # λ
    T = rewards.shape[0]  # batch_size = T
    for t in range(T):  # 每个t都有Gλ(t) t = 1,2,...,T
        r = 0  # n+t=T=batch_size
        for n in range(T - t):  # 每个t的计算都有 ∑(n=1,2,...,T-t-1) λ^(n-1)*Gn(t)
            # (1+λ+λ^2+λ^3+...+λ^T-1)*rt+1 + (λ+λ^2+λ^3+...+λ^T-1)*γ*rt+1 +...+ γ^t-1*rT
            r = rewards[n + t] * np.sum(lamda_seq[n: T - t]) * (1 - dones[n + t]) + r * gamma  # 加done,回合可能提前结束
            # r = rewards[n + t] * np.sum(lamda_seq[n: T - t]) + r * gamma
        discount_reward.append(r)
    #  δt = rt+1 + γ*rt+2 + γ^2*rt+3 + ...+ γ^n * rt+n+1 + γ^n*Q(st+n+1,at+n+1)
    # [G^λ(T),G^λ(T-1),...,G^λ(0)]  λ = 0.8
    """对每step都进行计算 累计 ∑τ=1~T-t-1 λγQ(st+τ,at+τ) """
    for t in range(T - 1):  # t=0,1,...,T-1-1 相当于 t=1,2,...,T-1 时
        target_q_temp = 0.0
        for i in range(T - t - 1):  # 计算每个Gt(λ)的累加 ∑ i=1~T-t-1 λ^(i-1) * γ^i * Q(st+i,at+i)
            target_q_temp += lamda_seq[i] * gamma_seq[1 + i] * target_q_next[t + i]
        discount_reward[t] = discount_reward[t] + target_q_temp
    """[::-1] 表示正序所有行，倒序所有列"""
    return np.array(discount_reward[::-1])


# MC-return
def monte_carlo_returns(rewards, gamma, lamda):
    max_epsiode_len = rewards.shape[0]
    G_t = []
    for t in range(max_epsiode_len):  # t= 1,2,...,T  数组索引 =  0,1,....,T-1
        monte_carlo_temp = 0.0
        for i in range(max_epsiode_len - t):  # 计算Gt=rt+1 + γ^(1) * rt+2 +... = ∑i=1~T-t γ^(i-1) * rt+i
            monte_carlo_temp += gamma ** (i) * rewards[t + i]  # γ^(i-1) * rt+i
        monte_carlo_temp *= lamda ** (max_epsiode_len - t - 1)  # 再乘以 λ^(T-t-1) * ( ∑i=1~T-t γ^(i-1) * rt+i )
        G_t.append(monte_carlo_temp)
    return G_t


# 软更新
def soft_update_exp(values, target_values):
    tau = 1.0 - 5e-4  # 0.9995
    target_vars = []
    """
    zip() 在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。用list(zip(a,b)),把打包为元组转为列表,元素个数与最短的列表一致
    sorted() 函数对所有可迭代的对象进行排序操作。对象类型不限制，默认升序排列
        key -- 主要是用来进行比较的元素，指定可迭代对象中的一个元素来进行排序。
    .assign()赋值，是一个op节点，返回了1个ojbect，只有sess.run才能运行
    tf.group() 返回一个操作，用于批量完成一些张量计算  
    用group()组合 soft update 操作，一次性更新所有权重/偏置
    """
    for var, target_var in zip(sorted(values, key=lambda v: v.name),
                               sorted(target_values, key=lambda v: v.name)):
        # 软更新 θ_target = tau * θ_target + (1 - tau) * θ
        target_vars.append(target_var.assign(tau * target_var + (1 - tau) * var))
    #  用group()组合 soft update 操作，一次性更新所有权重/偏置
    target_vars = tf.group(*target_vars)
    # 通过feed_dict运行see.run，进行更新计算
    return U.function([], [], updates=[target_vars])


# Actor神经网络-1层全连接+1层GRU+1层全连接
def coma_actor_rnn(inputs, num_outputs, scope, num_units=256, reuse=False, rnn_cell=None, last_h_state=None,
                   time_step=4, activation_fn=None):
    # ①声明变量空间名称
    with tf.variable_scope(scope, reuse=reuse):
        # inputs.shape= [batch,s]
        inputs_shape = inputs.shape.dims[1].value  # 最后一维

        out = tf.reshape(inputs, (-1, time_step, int(inputs_shape / time_step)))  # out = [?, 4, obs_shape]
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)  # out = [?, 4, 128]
        # 计算GRU
        # 增加1维，是为了满足批处理的次数, 一次把batch_size条数据都处理了？
        # [batch_size, time_step, max_obs_dim]  batch条数据，每条数据中，每个数据前后有时序关系，每一个数据是state_dim维的向量
        # tf.nn.dynamic_rnn 的
        # input是 [batch_size个序列, 每个序列最大长度-即状态维度， 该条序列的维度-1维标量]]
        # output是: h_out = [batch_size, max_state, cell.output_size]
        # gru:      last_state = [batch_size, cell.output_size] 输出的是最后一个batch的输出隐藏层状态，不需要
        h_out, h_state = tf.nn.dynamic_rnn(rnn_cell, out, initial_state=last_h_state, dtype=tf.float32)
        last_out = h_out[:, -1]
        # 计算全连接
        out = layers.fully_connected(last_out, num_outputs=num_outputs, activation_fn=activation_fn)
        # 输出是 out为动作, h_out是隐藏状态输出
        """
        ###############################################
        #################①需要改进的地方-GRU正确使用
        ##############################################
        使用了R2D2里面的2种方法：先试试①，①比较麻烦是多了保存隐藏层数据， 不过MF-MARL也需要保存额外的平均场动作数据，迟早都要弄。
        ① Stored-state方法，保存轨迹对应的隐藏层；        第①种，在update时，提取batch个hidden
        ② burn-in方法，保存2次轨迹，一次轨迹用来恢复隐藏层； 第②种，在update时，恢复hidden
        """
        return out, h_state


def coma_elegant_actor_rnn(inputs, num_outputs, scope, num_units=256, reuse=False, rnn_cell=None, last_h_state=None,
                           activation_fn=None):
    # ①声明变量空间名称
    with tf.variable_scope(scope, reuse=reuse):
        """来自小雅大佬 曾的idea ：
        把 part state + hidden state 视为完整的state(未完成)"""
        inputs_shape = inputs.shape.dims[1].value  # 最后一维
        out = tf.concat(out + last_h_state, axis=1)  # 拼接
        out = tf.reshape(inputs, (-1, 4, int(inputs_shape / 4)))
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        h_out, h_state = tf.nn.dynamic_rnn(rnn_cell, out, initial_state=last_h_state, dtype=tf.float32)
        last_out = h_out[:, -1]
        """把 hidden state + action 视为完整的 action(未完成)"""
        full_action_heature = tf.concat(last_out + h_state, axis=1)
        out = layers.fully_connected(full_action_heature, num_outputs=num_outputs, activation_fn=activation_fn)
        return out, h_state


# Critic网络-3层全连接-只为计算Q
def coma_critic_mlp(inputs, num_outputs, scope, num_units=128, reuse=False, rnn_cell=None, activation_fn=None):
    with tf.variable_scope(scope, reuse=reuse):
        outs = inputs
        outs = layers.fully_connected(outs, num_outputs=num_units, activation_fn=tf.nn.relu)
        outs = layers.fully_connected(outs, num_outputs=num_units, activation_fn=tf.nn.relu)
        outs = layers.fully_connected(outs, num_outputs=num_outputs, activation_fn=activation_fn)
        return outs


# Critic网络的输出和更新过程：
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
        # ①获取动作的act_ph_n
        act_pdtpye_n = [make_pdtype(act_shape) for act_shape in act_space_n]
        act_ph_n = [act_pdtpye_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        # agent的onehot编码
        agent_onehot_ph = [tf.placeholder(tf.float32, [None, len(act_space_n)], name="agent_onehot" + str(i)) for i in
                           range(len(act_space_n))]
        # ②创建Critic输入和构建网络
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        q_inputs = tf.concat(obs_ph_n + act_ph_n + agent_onehot_ph, 1)
        if local_func:
            q_inputs = tf.concat(obs_ph_n[q_index] + act_ph_n[q_index] + agent_onehot_ph[p_index], 1)
        q = q_func(q_inputs, 1, scope="q_func", num_units=num_units, activation_fn=None)[:, 0]  # 取值
        q_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        # ③建立Critic网络的损失函数loss 梯度下降得到最小MSE
        loss = tf.reduce_mean(tf.square(target_ph - q))
        epsilon = tf.reduce_mean(tf.square(q))
        q_loss = loss + epsilon * 1e-3
        optimizer_exp_var = U.minimize_and_clip(optimizer, q_loss, q_vars, grad_norm_clipping)
        # ④创建Critic_target网络
        target_q = q_func(q_inputs, 1, scope="target_q_func", num_units=num_units, activation_fn=None)[:, 0]
        target_q_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        # ⑤软更新Critic_target网络
        update_target_q_net = soft_update_exp(q_vars, target_q_vars)
        # ⑥输出所需变量及操作
        train = U.function(inputs=obs_ph_n + act_ph_n + agent_onehot_ph + [target_ph], outputs=q_loss,
                           updates=[optimizer_exp_var])
        q_values = U.function(inputs=obs_ph_n + act_ph_n + agent_onehot_ph, outputs=q)
        target_q_values = U.function(inputs=obs_ph_n + act_ph_n + agent_onehot_ph, outputs=target_q)
        # ⑦返回
        return train, update_target_q_net, {'q_values': q_values, 'target_q_values': target_q_values}


# Critic网络的输出和更新过程：
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
        # ①获取动作的act_ph_n
        act_pdtpye_n = [make_pdtype(act_shape) for act_shape in act_space_n]
        act_other_ph_n = [act_pdtpye_n[i].sample_placeholder([None], name="other_action" + str(i)) for i in
                          range(len(act_space_n))]
        # agent的onehot编码
        agent_onehot_ph = [tf.placeholder(tf.float32, [None, len(act_space_n)], name="agent_onehot" + str(i)) for i in
                           range(len(act_space_n))]
        # ②创建 counterfactual 输入和构建网络
        counterfactual_target_ph = tf.placeholder(tf.float32, [None], name="counterfactual_target")
        c_inputs = tf.concat(obs_ph_n + act_other_ph_n + agent_onehot_ph, 1)
        if local_func:
            c_inputs = tf.concat(obs_ph_n[c_index] + act_other_ph_n[c_index] + agent_onehot_ph[p_index], 1)
        counterfactual = counterfactual_func(c_inputs, 1, scope="counterfactual_func", num_units=num_units,
                                             activation_fn=None)[:, 0]  # 取值
        counterfactual_vars = U.scope_vars(U.absolute_scope_name("counterfactual_func"))
        # ③建立 counterfactual 网络的损失函数loss 梯度下降得到最小MSE
        loss = tf.reduce_mean(tf.square(counterfactual_target_ph - counterfactual))
        epsilon = tf.reduce_mean(tf.square(counterfactual))
        counterfactual_loss = loss + epsilon * 1e-4
        optimizer_exp_var = U.minimize_and_clip(optimizer, counterfactual_loss, counterfactual_vars, grad_norm_clipping)
        # ④创建 counterfactual_target 网络
        target_counterfactual = counterfactual_func(c_inputs, 1, scope="target_counterfactual_func",
                                                    num_units=num_units, activation_fn=None)[:, 0]
        target_counterfactual_vars = U.scope_vars(U.absolute_scope_name("target_counterfactual_func"))
        # ⑤软更新 counterfactual_target网络
        update_target_counterfactual_net = soft_update_exp(counterfactual_vars, target_counterfactual_vars)
        # ⑥输出所需变量及操作
        train = U.function(inputs=obs_ph_n + act_other_ph_n + agent_onehot_ph + [counterfactual_target_ph],
                           outputs=counterfactual_loss, updates=[optimizer_exp_var])
        counterfactual_values = U.function(inputs=obs_ph_n + act_other_ph_n + agent_onehot_ph, outputs=counterfactual)
        target_counterfactual_values = U.function(inputs=obs_ph_n + act_other_ph_n + agent_onehot_ph,
                                                  outputs=target_counterfactual)
        # ⑦返回
        return train, update_target_counterfactual_net, {'counterfactual_values': counterfactual_values,
                                                         'target_counterfactual_values': target_counterfactual_values}


# Actor网络的输出和更新计算过程：
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
        # ①获取动作张量的占位符
        act_pdtpye_n = [make_pdtype(act_shape) for act_shape in act_space_n]
        act_ph_n = [act_pdtpye_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        # ②获取Actor网络输出量
        # warpper_obs_ph = warpper_obs_np
        # 上一时间步的隐状态
        # last_hidden_state = warpper_obs_ph[1]
        # 上一时间步的联合动作- 这个已经包含在每个agent的obs里了

        # 当前时间步其他agent的联合动作
        act_ph_other = [act_pdtpye_n[i].sample_placeholder([None], name="other_action" + str(i)) for i in
                        range(len(act_space_n))]
        # agent的onehot编码
        agent_onehot_ph = [tf.placeholder(tf.float32, [None, len(act_space_n)], name="agent_onehot" + str(i)) for i in
                           range(len(act_space_n))]
        # 加入GRU_cell
        gru_cell = rnn.GRUCell(num_units=num_units,
                               kernel_initializer=tf.random_normal_initializer(-0.1, 0.1, seed=2),
                               bias_initializer=tf.random_normal_initializer(-0.1, 0.1, seed=2),
                               name="policy_gru_cell")
        # ③构建Actor网络

        p_inputs = warpper_obs_ph[:, :obs_shape]
        p, hidden_p = p_func(p_inputs, int(act_pdtpye_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units,
                             rnn_cell=gru_cell, last_h_state=warpper_obs_ph[:, obs_shape:], time_step=rnn_time_step,
                             activation_fn=None)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        # 获取动作
        act_pd = act_pdtpye_n[p_index].pdfromflat(p)
        act_sample = act_pd.sample()  # 未归一化
        # 归一化  -1~1 用tanh
        act_sample = tf.tanh(act_sample)
        # ④复建Critic网络
        q_inputs = tf.concat(obs_ph_n + act_ph_n + agent_onehot_ph, 1)
        if local_func:
            q_inputs = tf.concat(obs_ph_n[p_index] + act_ph_n[p_index] + agent_onehot_ph[p_index], 1)
        q = q_func(q_inputs, 1, scope="q_func", reuse=True, num_units=num_units, activation_fn=None)[:, 0]
        # ⑤建立Actor-loss函数
        # 反事实基线网络
        b_inputs = tf.concat(obs_ph_n + act_ph_other + agent_onehot_ph, 1)  #
        if local_func:
            b_inputs = tf.concat(obs_ph_n[p_index] + act_ph_n[p_index] + agent_onehot_ph[p_index], 1)
        counterfactual = b_func(b_inputs, 1, scope="counterfactual_func", reuse=True, num_units=num_units,
                                activation_fn=None)[:, 0]
        # counterfactual = tf.placeholder(tf.float32, [None], name="counterfactual")
        # 优势函数A =Q - b 梯度下降得到最小偏差
        A_loss = q - counterfactual  # A_loss = [t_size, 1]
        # act_sample = [t_size, act_dim], pg_loss = [1]
        pg_loss = - tf.reduce_mean(A_loss * tf.reduce_prod(act_sample, axis=1))  # [1]

        epsilon = tf.reduce_mean(tf.square(act_pd.flatparam()))  # == tf.reduce_mean(tf.square(p)) [1]
        loss = pg_loss + epsilon * 1e-3  # [1]
        # 更新Actor网络+梯度裁剪
        optimizer_p_vars = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)
        # ⑥构建Actor_target网络
        """
        # 不用构建target_p_network 因为是on-policy不需要目标网络
        """
        target_gru_cell = rnn.GRUCell(num_units=num_units,
                                      kernel_initializer=tf.random_normal_initializer(-0.1, 0.1, seed=2),
                                      bias_initializer=tf.random_normal_initializer(-0.1, 0.1, seed=2),
                                      name="target_policy_gru_cell")
        target_p, target_hidden_p = p_func(p_inputs, int(act_pdtpye_n[p_index].param_shape()[0]), scope="target_p_func",
                                           num_units=num_units,
                                           rnn_cell=target_gru_cell, last_h_state=warpper_obs_ph[:, obs_shape:],
                                           time_step=rnn_time_step, activation_fn=None)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        # 采样动作
        target_act_sample = act_pdtpye_n[p_index].pdfromflat(target_p).sample()
        target_act_sample = tf.tanh(target_act_sample)
        # ⑦软更新Actor_target网络
        target_p_soft_update = soft_update_exp(p_func_vars, target_p_func_vars)
        # ⑧建立function计算图
        act = U.function(inputs=[warpper_obs_ph], outputs=[act_sample, hidden_p])
        p_prob = U.function(inputs=[warpper_obs_ph], outputs=p)
        target_p_prob = U.function(inputs=[warpper_obs_ph], outputs=target_p)
        # hidden_state = U.function(inputs=[warpper_obs_ph], outputs=hidden_p)
        target_act = U.function(inputs=[warpper_obs_ph], outputs=[target_act_sample, target_hidden_p])
        # target_hidden_state = U.function(inputs=[warpper_obs_ph], outputs=target_hidden_p)
        train = U.function(inputs=obs_ph_n + act_ph_n + [warpper_obs_ph] + act_ph_other + agent_onehot_ph, outputs=loss,
                           updates=[optimizer_p_vars])

        return act, train, target_p_soft_update, {'p_prob': p_prob, 'target_p_prob': target_p_prob,
                                                  'target_act': target_act}


# COMA算法核心类
class COMADsicreteAgentTrainer(AgentTrainer):
    # 初始化各种模型变量①智能体数量，②Actor-Critic网络，③记忆库
    # ①智能体参数:COMA变量空间命名为agent的名字，第几个agent的COMA
    # ②Actor-Critic网络输入：状态张量，动作张量，神经网络模型，网络超参数
    # ③记忆库：容量大小，初始化索引
    def __init__(self, name,
                 actor_critic_model,
                 obs_space_n,
                 act_space_n,
                 agent_index,
                 args,
                 local_q_func=False):
        # ①智能体参数
        self.name = name
        self.n = len(obs_space_n)
        self.agent_index = agent_index
        self.args = args

        # self.register(name='ops',values=args.)
        # ②AC网络输入变量：
        # 2.1 状态占位符
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_space_n[i], name="observation" + str(i)).get())
        warpper_obs_tuple_to_list = list(obs_space_n[self.agent_index])
        warpper_obs_ph = U.BatchInput([warpper_obs_tuple_to_list[0] * 4 + args.num_units],
                                      name="warpper_obs_and_last_hidden_state" + str(self.agent_index)).get()
        """先建立counterfactual网络-再Q网络-再P网络"""
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
        # 2.2 建立Q价值网络及计算图
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
        # 2.3 建立策略网络及计算图
        self.act, self.p_train, self.p_target_update, self.p_target_act = p_train(
            scope=self.name,
            obs_ph_n=obs_ph_n,
            warpper_obs_ph=warpper_obs_ph,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=coma_actor_rnn,
            q_func=coma_critic_mlp,
            b_func=coma_critic_mlp,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr * 0.1),
            grad_norm_clipping=0.5,  # 梯度裁剪
            local_func=local_q_func,  # 判断使用Signal-agentRL还是Multi-agentRL
            num_units=args.num_units,
            rnn_time_step=args.rnn_time_step,
            obs_shape=warpper_obs_tuple_to_list[0] * 4)

        # ③记忆库
        self.on_policy_replay_buffer = OnPolicyReplayBuffer(self.args.max_episode_len)  # max step的 容量
        # 一次批量提取的最大长度max_replay_buffer_len
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len  # COMA是存满了再批量更新
        self.replay_sample_index = None
        """权宜之计，在类里加入 hidden_state，以及保存3个历史片段 """
        self.hidden_state = None
        self.history_states = None

    # 选取动作
    def action(self, obs):
        # 取前 3个 step 的 state + 当前 state
        if self.hidden_state is None:  # 类似初始化
            self.hidden_state = np.zeros((self.args.num_units,))
        self.warpper(obs)  # 组成4个历史片段
        actor_inputs_feature = np.concatenate([self.history_states] + [self.hidden_state])
        [act, act_hidden] = self.act(actor_inputs_feature[None])
        self.hidden_state = act_hidden[0]
        return act[0]

    def warpper(self, obs_t):
        if self.history_states is None:  # 即第一次交互时的输入初始化
            self.history_states = np.array([obs_t] * self.args.rnn_time_step).reshape(-1, )
        else:  # 依次保存历史state
            temp_warpper = self.history_states.reshape(self.args.rnn_time_step, -1)
            temp_warpper = temp_warpper[1:]
            self.history_states = np.concatenate([temp_warpper, obs_t.reshape(1, -1)], axis=0).reshape(-1, )  # 在第0维度拼接

    # 存储记忆库
    """on-policy的记忆存储，与off-policy的区别？"""

    # 将经验（S,A,R,S'）存入经验记忆库replay_buffer
    def experience(self, obs, act, rew, obs_next, done, done_all):
        self.on_policy_replay_buffer.add(obs, act, rew, obs_next, 1.0 if done == True or done_all == True else 0.0)

    # 清空记忆库
    # 超参数-重置从记忆库取样的索引（下标）
    def preupdate(self):
        self.replay_sample_index = None

    # numpy->list 部分数组转list
    def obs_act_obs_next_numpy_to_list(self, obs_n_sub, act_n_sub, obs_next_n_sub):
        temp_obs, temp_act, temp_obs_next = [], [], []
        for i in range(self.n):
            temp_obs.append(obs_n_sub[i])
            temp_act.append(act_n_sub[i])
            temp_obs_next.append(obs_next_n_sub[i])
        return temp_obs, temp_act, temp_obs_next

    # 计算更新网络loss
    def update(self, agents, step):
        # ①判定更新间隔
        if len(self.on_policy_replay_buffer) < self.max_replay_buffer_len:
            return

        """更新间隔-一回合"""
        if not step % self.args.max_episode_len == 0:
            return

        # 当前step数
        # current_step = step % self.args.max_episode_len
        # ②提取记忆库数据
        obs_n, act_n, obs_next_n = [], [], []
        # self.replay_sample_index = self.on_policy_replay_buffer.make_sample_index(self.args.batch_size) # 随机采样
        self.replay_sample_index = self.on_policy_replay_buffer.on_policy_make_indexs_sample(
            self.args.max_episode_len)  # 在线采样
        indexs = self.replay_sample_index  # 随机采样编码
        # max_step = self.args.max_episode_len  # step = 200
        # 先获取所有agnet的状态，动作，下一状态 维度是[batch_size*(step - indexs[i]), n]
        for i in range(self.n):
            obs, act, _, obs_next, _ = \
                agents[i].on_policy_replay_buffer.on_poicy_sampe(indexs)
            obs_n.append(obs)
            act_n.append(act)
            obs_next_n.append(obs_next)
        # 再获取当前agent的奖励和done_mask 维度是[batch_size*(step - indexs[i]), 1
        _, _, rew, _, mask = self.on_policy_replay_buffer.on_poicy_sampe(indexs)
        # current_agent_done_mask = agent_done_mask
        # current_agent_rew = agent_rew
        # ③计算更新Actor-Critic网络-批量处理
        # batch_q_loss = []
        # batch_p_loss = []
        num_sample = 1

        # 转ndarray
        # obs_n, act_n, obs_next_n = self.obs_act_obs_next_list_to_numpy(obs_n, act_n, obs_next_n)
        # obs_n, act_n, obs_next_n = np.array(obs_n), np.array(act_n), np.array(obs_next_n)
        # act_n.shape = [N,batch_size,max_step,number_act]
        # agent_rew.shape = [batch_size,max_step,1]
        # for i in range(self.args.batch_size):  # 进行batch_size次片段更新
        #     obs_n_sub = obs_n[:,i,:,:]  # [N,max_step,number_obs]
        #     act_n_sub = act_n[:,i,:,:]  # [N,max_step,number_act]
        #     rew_sub = agent_rew[i,]  # [max_step,1]
        #     obs_next_n_sub = obs_next_n[:,i,:,:]  # [N,max_step,number_obs_next]
        #     done_mask_sub = agent_done_mask[i,] # [max_step,1]
        # 裁剪0元素
        # obs_n_sub, act_n_sub, obs_next_n_sub, rew_sub, done_mask_sub= self.clip_zeros_ndarray(obs_n_sub, act_n_sub, rew_sub, obs_next_n_sub, done_mask_sub)
        # 变成主导的nlist
        # obs_n_sub, act_n_sub, obs_next_n_sub = self.obs_act_obs_next_numpy_to_list(obs_n_sub, act_n_sub, obs_next_n_sub)
        # target_q = 0.0
        # 计算n-step-TD(λ)的所有值
        """
        Gλ(t) = (1-λ)∑n (λ)^(n-1)*G(n)(t)  n = 1,2,...,T , t = 1,2,...,T
        G(n)(t) = ∑n γ^n-1 * rt+n （t=1,2,...,n） + γ^n*Q(st+n,at+n)
        G(n)(t) = [G(1)(t),G(2)(t),...,G(T)(t)] 
        G(n)(t) = [ λ^0[γ^0*rt+1 + γ^1*Q(st+1,at+1)],...,λ^n-1[ ∑γ^n-1 * rt+1  + γ^n*Q(st+n,at+n)] ].sum = [1] t+n=T
        Gλ(t) = [(1-λ)*G(n)(0), (1-λ)*G(n)(1),..., (1-λ)*G(n)(T)]  = [batch_size, 1]
        """
        G_t_lamda = 0.0
        for i in range(num_sample):
            # ①计算 Y 用 TD(λ) 得到序列折扣回报
            # 奇怪的n-step TD(λ)-用随机数代替n，只计算一次，且计算的是整个片段，λ=0.8
            # target_act_next = [N,t_size]
            # 把所有list转为numpy
            target_act_next_n = [agents[i].p_target_act['target_act'](obs_next_n[i]) for i in
                                 range(self.n)]  # =[5, 200, 1]
            # target_q_next = [t_size,1]
            target_q_next = (1.0 - mask) * self.q_target_values['target_q_values'](
                *(obs_next_n + target_act_next_n))  # [200, 1]

            # target_q_next *= np.logspace(0, rew.shape[0]-1,num=rew.shape[0], endpoint=True, base=self.args.tdlambda)  # = [200, 1]
            gamma_seq = np.logspace(1, rew.shape[0], num=rew.shape[0], endpoint=True,
                                    base=self.args.gamma)  # = [200, 1]
            # [tdlambda*gamma, (tdlambda*gamma)^2,...,(tdlambda*gamma)^T] = [batch_size, 1]
            tdlamda_seq = np.logspace(0, rew.shape[0] - 1, num=rew.shape[0], endpoint=True,
                                      base=self.args.tdlambda)  # = [200, 1]
            # 计算target_q_next* tdlamda_seq
            # target_q_next *= gamma_seq * tdlamda_seq
            # """对每step都进行计算 累计 ∑λγQ(st+τ,at+τ) """
            # copy_target_q_next = copy.deepcopy(target_q_next)
            # sum_target_q_next = np.sum(copy_target_q_next)
            # copy_target_q_next = sum_target_q_next - copy_target_q_next
            # copy_target_q_next[0,:] = sum_target_q_next
            # target_q_next = copy_target_q_next
            # tdlamda_sum = [1*(self.args.tdlamda**i) for i in range(0,self.args.max_episode_len)]
            # Gλ(t) = (1-λ)∑(λ)^(n-1)*G(n)(t)  n = 1,2,...,T , t = 1,2,...,T
            # td_discount_reward = [G(n=T)(0),G(n=T-1)(1),...,G(n=1)(T-1)]  = [1,t_size]

            G_t_n = (1 - self.args.tdlambda) * td_lambda_discount_reward(rew, mask, gamma=self.args.gamma,
                                                                         lamda_seq=tdlamda_seq, gamma_seq=gamma_seq,
                                                                         target_q_next=target_q_next)
            G_t = monte_carlo_returns(rew, self.args.gamma, self.args.tdlambda)
            G_t_lamda += G_t_n + G_t

        # target_q = target_q / num_sample
        G_t_lamda = G_t_lamda / num_sample
        # ②更新Critic网络
        q_loss = self.q_train(*(obs_n + act_n + [G_t_lamda]))
        # batch_q_loss.append(q_loss)
        if step % 400 == 0:  # 目标网络更新间隔150
            self.q_targrt_update()
        # ③更新Actor网络 优势函数的基线: A(s,a) = q(s,a) - ∑aiq(s,ai,a-i), ai=[a1,a2,...,a|A|]
        # 在actor网络里重新采样，作为边缘分布的概率，采样一次，可循环采样n次，再取平均即可

        # on-policy
        """
        #########################################
        ########②需要改进的地方：多维动作的边缘概率分布
        ########################################
        """
        target_act_new_n = self.p_target_act['target_act'](obs_next_n[self.agent_index])
        act_actor_n = copy.deepcopy(act_n)  # 为了不影响原本记忆库中数据
        act_actor_n[self.agent_index] = target_act_new_n  # 替换采样数据
        # 计算边缘分布
        q = self.q_target_values['q_values'](*(obs_n + act_actor_n))  # [t_size, 1]
        """
        # on-policy
        target_act_new_n = self.act((obs_next_n[self.agent_index]))
        act_actor_new_n = copy.deepcopy(act_n)
        act_actor_new_n[self.agent_index] = act_new_n
        q = self.q_target_values['q_values'](*(obs_n + act_actor_new_n))
        """
        # 多维动作-
        """
        思考：q是一个批次的值 q = [t_size,1]  那应该是对批次的每个t做求解边缘分布
        targert_act_new_n是1个批次的选择动作 = [t_size, act_dim]
        所以对每个t， 得到的结果应该是  [t_size, 1]             
        """
        # 每个状态下的动作分布-t_size个状态下的 [均值,期望]  [t_size, mean, logstd]
        # p_prob = self.p_target_act['p_prob'](obs_next_n_sub[self.agent_index])
        # p_prob = np.tanh(p_prob)  # 约束到-1~1
        # 从策略分布中采样  # mean = [t_size, act_num], logstd = [t_size, act_num]
        # mean, logstd = np.split(ary=p_prob,indices_or_sections=2,axis=1)
        # 100次蒙特卡洛采样
        # target_act_new_prob = self.mc_sample(mean,logstd,100)
        # target_act_new_prob = np.random.normal(mean, np.exp(logstd), 100)
        # 估计动作均值为期望
        # target_act_new_prob = np.mean(target_act_new_prob, axis=1) # 在t_size上求均值
        advantages_q = q * np.prod(target_act_new_n, axis=1)  # 在t_size上求概率积 [t_size ,1]

        # ave_q = q * target_act_new_n  # [t_size, 1]
        # for i in  range(len(ave_q)): # p_loss = [t_size,1] 相当于进行了t_size次更新
        p_loss = self.p_train(*(obs_n + act_n + [advantages_q]))
        # batch_p_loss.append(p_loss)
        if step % 200 == 0:  # 目标网络更新
            self.p_target_update()  # 目标网络更新间隔50
        # 若是最后一个agent更新完毕，重置所有记忆库
        if self.agent_index == 11:
            for i in range(self.n):
                agents[i].on_policy_replay_buffer.clear()
        # ④输出
        return [q_loss, p_loss, np.mean(G_t), np.mean(rew), np.mean(target_q_next), np.std(G_t)]


# OnPolicyReplayBuffer在线策略的经验回放区
"""
优先回放记忆库？？ 暂放
普通的回放记忆库
"""


class OnPolicyReplayBuffer(object):

    def __init__(self, size):
        self._storage = []
        self._maxsize = int(size)
        self._next_index = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_index = 0

    def add(self, obs_t, action, reward, obs_t1, done_mask):  # 加入replay buffer 长度不超过1个 episode
        data = (obs_t, action, reward, obs_t1, done_mask)
        if self._next_index >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_index] = data
        self._next_index = (self._next_index + 1) % self._maxsize

    ###***************************************###
    ###   on-policy-sample
    ###***************************************###
    # 取最大step步长作为采样索引indexs
    def on_policy_make_indexs_sample(self, step):
        return [step]

    # 普通序列采样数据
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

    # 根据indexs执行采样
    def on_poicy_sampe(self, indexs):
        return self._on_policy_lambda_encode_sampe(indexs)

    ###***************************************###
    ###   off-policy-sample  n-step TD(λ)采样
    ###***************************************###

    def make_sort_sample_index(self, batch_size):
        """ .sort() 函数用于对原列表进行排序，如果指定参数，则使用比较函数指定的比较函数。
        list.sort(cmp=None, key=None, reverse=False)
        reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）。"""
        index_sample = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        list.sort(index_sample)
        return index_sample

    def TD_lambda_sample(self, indexs, step):
        return self._off_policy_lambda_encode_sample(indexs, step)

    def _off_policy_lambda_encode_sample(self, indexs, step):
        obs, act, rew, obs_next, done_mask = [], [], [], [], []
        obs_dim, act_dim, rew_dim, obs_next_dim, done_mask_dim = self._storage[0]
        # 相同维度的0矩阵
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
                if done_mask_t == 1.0:  # 终止状态则停止读取  beyond max_buffer_list
                    break
                if i < len(indexs) - 1:  # 先判断是否超indexs维度  如果是最后1个
                    if indexs[i] + j + 1 >= indexs[i + 1]:  # 再判断相邻索引
                        break
            # 补齐轨迹序列-当做死亡时的状态
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

    ###***************************************###
    ###   off-policy-sample  1-step TD(0)采样
    ###***************************************###
    # 随机采样
    def make_sample_index(self, batch_size):
        """ .sort() 函数用于对原列表进行排序，如果指定参数，则使用比较函数指定的比较函数。
        list.sort(cmp=None, key=None, reverse=False)
        reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）。"""
        index_sample = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        # list.sort(index_sample)
        return index_sample

    # 打乱逆序采样
    def make_last_jam_index(self, batch_size):
        indexs = [(self._next_index - i - 1) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(indexs)
        return indexs

    # 倒序取值
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

    ###***************************************###
    ###   PER 优先级回放（未完成）
    ###***************************************###
    def PER(self, batch_size):  # 优先采样回放
        pass
    # pass
