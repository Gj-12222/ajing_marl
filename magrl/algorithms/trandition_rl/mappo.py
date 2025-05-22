r'''

mappo


'''
import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from Algorithms.Trainer import AgentTrainer
from Algorithms.Network.net import RNNMLPCritic, DiscreteRNNActor
from Algorithms.rl_utils.ReplayBuffer import ReplayBufferTrajectory
from Algorithms.rl_utils.batch import Batch
from Algorithms.rl_utils.valuenorm import ValueNorm


class MAPPODiscreteAgentTrainer(AgentTrainer):
    name = 'mappo'

    def __init__(self, config, algo_config, agent_index, agents):
        super().__init__()
        self.cfg = config
        self.n = len(agents)
        self.agent_cfg = config['agent_config']
        self.algo_cfg = algo_config
        self.agent_index = agent_index
        self.n = config['agent_config']['agent_num']
        self.device = config['device']
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.clip_param = algo_config['clip_param']
        self.ppo_repeat_epoch = algo_config['repeat_epoch']
        self.num_mini_batch = algo_config['mini_batch']
        self.data_chunk_length = algo_config['timestep']
        self.value_loss_coef = algo_config['valueLoss_coef']
        self.entropy_coef = algo_config['entropy_coef']
        self.max_grad_norm = algo_config['max_gradNorm']
        self.huber_delta = algo_config['huber_delta']
        self.hidden_dim = algo_config['hidden_dim']
        self.activate_fn = algo_config['activate_fn']
        self.actor_lr = algo_config['actor_lr']
        self.critic_lr = algo_config['critic_lr']
        self.soft_update_freq = algo_config['soft_update_freq']

        self.use_recurrent_policy = algo_config['use_rnn_policy']
        self.rnn_layer_dim = algo_config['rnn_layer_dim']
        self.use_max_grad_norm = algo_config['use_max_gradNorm']
        self.use_clipped_value_loss = algo_config['use_clip_valueLoss']
        self.use_huber_loss = algo_config['use_huber_loss']
        self.use_popart = algo_config['use_popart']
        self.use_valuenorm = algo_config['use_valueNorm']
        self.use_value_active_masks = algo_config['use_value_active_masks']
        self.use_policy_active_masks = algo_config['use_policy_active_masks']

        self.obs_dim = agents[agent_index].obs_dim
        self.action_dim = agents[agent_index].action_dim

        self.PolicyActor = DiscreteRNNActor(self.obs_dim, self.hidden_dim, sum(self.action_dim), device=self.device,
                                            cfg=algo_config)
        self.PolicyActor_optimizer = torch.optim.Adam(self.PolicyActor.parameters(), lr=self.actor_lr)

        self.v_input_dim = sum([agent.obs_dim for agent in agents])
        self.VNetCritic = RNNMLPCritic(self.v_input_dim, self.hidden_dim, 1, device=self.device, cfg=algo_config)
        self.VNetCritic_optimizer = torch.optim.Adam(self.VNetCritic.parameters(), lr=self.critic_lr)

        # self.V2NetCritic = RNNMLPCritic(self.v_input_dim, self.hidden_dim, 1, device=self.device)
        # self.targetV2NetCritic = RNNMLPCritic(self.v_input_dim, self.hidden_dim, 1, device=self.device)
        # self.V2NetCritic_optimizer = torch.optim.Adam(self.V2NetCritic.parameters(), lr=self.critic_lr)

        assert (self.use_popart and self.use_valuenorm) == False, \
            ("self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self.use_popart:
            self.value_normalizer = self.VNetCritic.value_out
        elif self.use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

        self.replay_buffer = ReplayBufferTrajectory(agents, agent_index, self.algo_cfg)
        # template
        self.action_log_prob = None
        self.actor_rnn_hidden_state = None
        self.critic_rnn_hidden_state = None
        self.critic_values = None

    def action(self, obs_n):
        # actor
        obs = obs_n[self.agent_index]
        if len(obs.shape) < 2:
            obs = torch.tensor(obs[None], device=self.device)

        mask = self.replay_buffer.get_mask()
        if self.use_recurrent_policy:
            _actor_rnn_hidden_state, _critic_rnn_hidden_state = self.replay_buffer.get_rnn_hidden_state()
        else:
            _actor_rnn_hidden_state = None
            _critic_rnn_hidden_state = None

        action_dist, self.actor_rnn_hidden_state = self.PolicyActor(obs, _actor_rnn_hidden_state, mask)
        action = action_dist.sample()  # dim=-1少1维度，是否影响insert replay-buffer
        # critic
        join_obs = torch.tensor(np.concatenate(obs_n, axis=-1), device=self.device).reshape(1, -1)
        self.critic_values, self.critic_rnn_hidden_state = self.VNetCritic(join_obs, _critic_rnn_hidden_state, mask)

        self.action_log_prob = action_dist.log_prob(action).sum(dim=-1)  # dim=-1少1维度，是否影响insert replay-buffer

        return action

    def save_replay_buffer(self, obs, action, reward, next_obs, done, terminal):
        transition, rnn_data = Batch(), Batch()
        transition.obs = obs.reshape(1, -1)
        transition.action = self._t2n(action).reshape(1, -1)
        transition.next_obs = next_obs.reshape(1, -1)
        transition.reward = np.array(reward, dtype=np.float32).reshape(1, -1)
        transition.done = np.array([done or terminal], dtype=np.float32).reshape(1, -1)
        transition.old_value = self._t2n(self.critic_values)
        transition.action_log_prob = self._t2n(self.action_log_prob)
        # 计算 mask
        if done or terminal:
            transition.mask = np.zeros((1, 1), dtype=np.float32)
        else:
            transition.mask = np.ones((1, 1), dtype=np.float32)

        transition.to_numpy()

        if self.use_recurrent_policy:
            # 在replay buffer里处理rnn_hidden_state
            # rnn_hidden_state.shape = [1, rnn_layer_num, rnn_hidden]
            rnn_data.actor_rnn_hidden_state = self._t2n(self.actor_rnn_hidden_state)
            rnn_data.critic_rnn_hidden_state = self._t2n(self.critic_rnn_hidden_state)

            if (transition.done == True).sum() > 0:
                rnn_data.actor_rnn_hidden_state[transition.done.squeeze(axis=0) == True] = \
                    np.zeros(((transition.done == True).sum(), self.algo_cfg['rnn_layer_dim'], self.algo_cfg['rnn_hidden']), dtype=np.float32)
                rnn_data.critic_rnn_hidden_state[transition.done.squeeze(axis=0) == True] = \
                    np.zeros(((transition.done == True).sum(), self.algo_cfg['rnn_layer_dim'], self.algo_cfg['rnn_hidden']), dtype=np.float32)

        else:
            rnn_data = None

        self.replay_buffer.add(transition, rnn_data)

    def buffer_clear(self):
        self.replay_buffer.clear()

    def update(self, agents, train_step, agent_index=None):
        # compute return
        trajectory_data, rnn_trajectory_data, join_obs = self.compute_return(agents)

        # train network
        info = self.train_network(trajectory_data, rnn_trajectory_data, join_obs)

        self.replay_buffer.after_update()

        return info

    def train_network(self, buffer, rnn_buffer, join_obs, update_actor=True):
        # normalize advantage
        if self.use_popart or self.use_valuenorm:
            # denormalize value, 主要考虑了value是经过了normalize
            # 最后一步的next_obs的 advantage， 直接用最后一步的return - next_obs的 value
            advantages = buffer.returns[:-1].cpu().numpy() - self.value_normalizer.denormalize(buffer.old_value[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.old_value[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_mask[:-1].cpu().numpy() == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        advantages = torch.from_numpy(advantages).to(self.device)

        train_info = {'value_loss': 0,
                      'policy_loss': 0,
                      'dist_entropy': 0,
                      'actor_grad_norm': 0,
                      'critic_grad_norm': 0,
                      'ratio': 0,
                      }

        for _ in range(self.ppo_repeat_epoch):
            # data loader
            data_loader = self.train_dataloader(buffer, advantages, join_obs, rnn_buffer=rnn_buffer)

            # train
            for _data in data_loader:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights= \
                    self.ppo_train(_data, update_actor)
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_repeat_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def ppo_train(self, data, update_actor):
        share_obs_batch, obs_batch, actor_rnn_hidden_state, critic_rnn_hidden_state, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = data

        old_action_log_probs_batch = self.check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = self.check(adv_targ).to(**self.tpdv)
        value_preds_batch = self.check(value_preds_batch).to(**self.tpdv)
        return_batch = self.check(return_batch).to(**self.tpdv)
        active_masks_batch = self.check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.evaluate_actions(share_obs_batch,
                                                                       obs_batch,
                                                                       actor_rnn_hidden_state,
                                                                       critic_rnn_hidden_state,
                                                                       actions_batch,
                                                                       masks_batch,
                                                                       available_actions_batch,
                                                                       active_masks_batch,
                                                                       )
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self.use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.PolicyActor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.PolicyActor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = self.get_gard_norm(self.PolicyActor.parameters())

        self.PolicyActor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.VNetCritic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.VNetCritic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = self.get_gard_norm(self.VNetCritic.parameters())

        self.VNetCritic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self.use_popart or self.use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = self.huber_loss(error_clipped, self.huber_delta)
            value_loss_original = self.huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = self.mse_loss(error_clipped)
            value_loss_original = self.mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self.use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def compute_return(self, agents):
        trajectory_data, rnn_trajectory_data, join_obs = None, None, []
        for i in range(self.n):
            _buffer_data, _rnn_buffer_data = agents[i].replay_buffer.get_data(_copy=True)
            join_obs.append(torch.from_numpy(_buffer_data.obs).to(self.device))
            if self.agent_index == i:
                _buffer_data.to_torch(device=self.device)
                _rnn_buffer_data.to_torch(device=self.device)
                trajectory_data = _buffer_data
                rnn_trajectory_data = _rnn_buffer_data

        join_obs = torch.cat(join_obs, dim=-1)
        if self.use_recurrent_policy:
            # 获取最后一步的rnn_hidden_state shape=[rnn_layer_num, rnn_hidden_dim]
            critic_rnn_hidden_state = rnn_trajectory_data.critic_rnn_hidden_state[-1].reshape(self.rnn_layer_dim, -1)
        else:
            critic_rnn_hidden_state = None
        # 计算最后一步的next_obs的next_value，这里不需要action，因为是 V
        next_value, _ = self.VNetCritic(join_obs[-1].unsqueeze(dim=0),
                                        critic_rnn_hidden_state,
                                        trajectory_data.mask[-1].reshape(1, -1))
        next_value = self._t2n(next_value).reshape(1, -1)
        self.replay_buffer.compute_return(next_value, self.value_normalizer)

        return trajectory_data, rnn_trajectory_data, join_obs

    def train_dataloader(self, buffer, advantages, join_obs, mini_batch_size=None, rnn_buffer=None):
        if self.use_recurrent_policy:
            data_loader = self.rnn_forward_generator(buffer,
                                                     rnn_buffer,
                                                     advantages,
                                                     join_obs,
                                                     self.data_chunk_length)
        else:
            data_loader = self.feed_forward_generator(buffer,
                                                      advantages,
                                                      join_obs,
                                                      mini_batch_size=mini_batch_size,
                                                      rnn_buffer=rnn_buffer)

        return data_loader

    def rnn_forward_generator(self, buffer, rnn_buffer, advantages, join_obs, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        episode_length = self.replay_buffer.buffer_len
        batch_size = 1 * episode_length * 1  # 1 * episode_len * 1(非共享,共享为n)
        # 按照data_chunk_length长度切分为data_chunks个序列（向下取整） 1.1取1
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        # num_mini_batch: (int) number of minibatches to split the batch into.
        # 所有序列数data_chunks按照self.num_mini_batch 可以切分mini_batch_size个
        mini_batch_size = data_chunks // self.num_mini_batch

        assert episode_length >= data_chunk_length, (
            "PPO requires the * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(episode_length, data_chunk_length))
        assert data_chunks >= 2, ("need larger batch size")
        # 随机序列index
        rand = torch.randperm(data_chunks).numpy()
        # 按照mini_batch_size计算min_batch的index
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(self.num_mini_batch)]

        if len(join_obs.shape) > 2:
            # [episode_len, obs_shape*agent_num]
            join_obs = join_obs[:-1].transpose(1, 0, 2, 3).reshape(-1, *join_obs.shape[1:])
            obs = buffer.obs[:-1].transpose(1, 0, 2, 3).reshape(-1, *buffer.obs.shape[1:])
        else:
            join_obs = join_obs[:-1]
            obs = buffer.obs[:-1]

        action = buffer.action
        action_log_prob = buffer.action_log_prob
        old_value = buffer.old_value[:-1]
        returns = buffer.returns[:-1]
        mask = buffer.mask[:-1]
        active_mask = buffer.active_mask[:-1]

        actor_rnn_hidden_state = rnn_buffer.actor_rnn_hidden_state[:-1]
        critic_rnn_hidden_state = rnn_buffer.critic_rnn_hidden_state[:-1]

        if buffer.available_actions is not None:
            available_actions = (buffer.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            actor_rnn_hidden_batch = []
            critic_rnn_hidden_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                share_obs_batch.append(join_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(action[ind:ind + data_chunk_length])
                if buffer.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(old_value[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(mask[ind:ind + data_chunk_length])
                active_masks_batch.append(active_mask[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_prob[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                actor_rnn_hidden_batch.append(actor_rnn_hidden_state[ind])
                critic_rnn_hidden_batch.append(critic_rnn_hidden_state[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (N, L, Dim)
            share_obs_batch = torch.stack(share_obs_batch, dim=1)
            obs_batch = torch.stack(obs_batch, dim=1)
            actions_batch = torch.stack(actions_batch, dim=1)

            if buffer.available_actions is not None:
                available_actions_batch = torch.stack(available_actions_batch, dim=1)
            value_preds_batch = torch.stack(value_preds_batch, dim=1)
            return_batch = torch.stack(return_batch, dim=1)
            masks_batch = torch.stack(masks_batch, dim=1)
            active_masks_batch = torch.stack(active_masks_batch, dim=1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, dim=1)
            adv_targ = torch.stack(adv_targ, dim=1)

            # States is just a (N, -1) from_numpy
            actor_rnn_hidden_batch = torch.stack(actor_rnn_hidden_batch, dim=1)
            actor_rnn_hidden_batch = actor_rnn_hidden_batch.reshape(N, *rnn_buffer.actor_rnn_hidden_state.shape[1:])
            critic_rnn_hidden_batch = torch.stack(critic_rnn_hidden_batch, dim=1)
            critic_rnn_hidden_batch = critic_rnn_hidden_batch.reshape(N, *rnn_buffer.actor_rnn_hidden_state.shape[1:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = self._flatten(L, N, share_obs_batch)
            obs_batch = self._flatten(L, N, obs_batch)
            actions_batch = self._flatten(L, N, actions_batch)
            if buffer.available_actions is not None:
                available_actions_batch = self._flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = self._flatten(L, N, value_preds_batch)
            return_batch = self._flatten(L, N, return_batch)
            masks_batch = self._flatten(L, N, masks_batch)
            active_masks_batch = self._flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = self._flatten(L, N, old_action_log_probs_batch)
            adv_targ = self._flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, actor_rnn_hidden_batch, critic_rnn_hidden_batch, actions_batch,\
                value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                adv_targ, available_actions_batch

    def feed_forward_generator(self, buffer, advantages, join_obs, mini_batch_size=None, rnn_buffer=None):
        episode_length = self.replay_buffer.buffer_len
        batch_size = episode_length
        num_mini_batch = self.num_mini_batch
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the"
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(episode_length, episode_length, num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        join_obs = join_obs[:-1].reshape(-1, *join_obs.shape[1:])
        obs = buffer.obs[:-1].reshape(-1, *buffer.obs.shape[1:])
        if self.use_recurrent_policy:
            actor_rnn_shape = buffer.actor_rnn_hidden_state.shape[1:]
            critic_rnn_shape = buffer.critic_rnn_hidden_state.shape[1:]
            actor_rnn_hidden_state = rnn_buffer.actor_rnn_hidden_state[:-1].reshape(-1, *actor_rnn_shape)
            critic_rnn_hidden_state = rnn_buffer.critic_rnn_hidden_state[:-1].reshape(-1,*critic_rnn_shape)

        action = buffer.action.reshape(-1, buffer.action.shape[-1])
        if buffer.available_actions is not None:
            available_actions = buffer.available_actions[:-1].reshape(-1, buffer.available_actions.shape[-1])
        old_value = buffer.old_value[:-1].reshape(-1, 1)
        returns = buffer.returns[:-1].reshape(-1, 1)
        mask = buffer.mask[:-1].reshape(-1, 1)
        active_mask = buffer.active_mask[:-1].reshape(-1, 1)
        action_log_prob = buffer.action_log_prob.reshape(-1, buffer.action_log_prob.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = join_obs[indices]
            obs_batch = obs[indices]
            actions_batch = action[indices]
            if buffer.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = old_value[indices]
            return_batch = returns[indices]
            masks_batch = mask[indices]
            active_masks_batch = active_mask[indices]
            old_action_log_probs_batch = action_log_prob[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            if self.use_recurrent_policy:
                actor_rnn_hidden_state_batch = actor_rnn_hidden_state[indices]
                critic_rnn_hidden_state_batch = critic_rnn_hidden_state[indices]
            else:
                actor_rnn_hidden_state_batch = None
                critic_rnn_hidden_state_batch = None
            yield share_obs_batch, obs_batch, actor_rnn_hidden_state_batch, critic_rnn_hidden_state_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def evaluate_actions(self,
                         cent_obs,
                         obs,
                         actor_rnn_hidden_state,
                         critic_rnn_hidden_state,
                         action,
                         masks,
                         available_actions=None,
                         active_masks=None,
                         ):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        """
        obs = self.check(obs).to(**self.tpdv)
        if not actor_rnn_hidden_state is None:
            actor_rnn_hidden_state = self.check(actor_rnn_hidden_state).to(**self.tpdv)
        batch_action = self.check(action).to(**self.tpdv)
        masks = self.check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = self.check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = self.check(active_masks).to(**self.tpdv)

        action_dist, _ = self.PolicyActor(obs, actor_rnn_hidden_state, masks)
        action_log_probs, dist_entropy = action_dist.log_prob(batch_action).sum(dim=-1), action_dist.entropy()

        if active_masks is not None:
            dist_entropy = (dist_entropy * active_masks.squeeze(-1)).sum() / active_masks.sum()
        else:
            dist_entropy = dist_entropy.mean()

        values, _ = self.VNetCritic(cent_obs, critic_rnn_hidden_state, masks)
        return values, action_log_probs, dist_entropy

    def get_gard_norm(self, it):
        sum_grad = 0
        for x in it:
            if x.grad is None:
                continue
            sum_grad += x.grad.norm() ** 2
        return math.sqrt(sum_grad)

    def update_linear_schedule(self, optimizer, epoch, total_num_epochs, initial_lr):
        """Decreases the learning rate linearly"""
        lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def huber_loss(self, e, d):
        a = (abs(e) <= d).float()
        b = (abs(e) > d).float()
        return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)

    def mse_loss(self, e):
        return e ** 2 / 2

    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def check(self, input):
        if len(input.shape) < 2:
            input = input.reshape(1,-1)
        output = torch.from_numpy(input) if type(input) == np.ndarray else input
        return output

    def train(self):
        self.VNetCritic.train()
        self.PolicyActor.train()

    def eval(self):
        self.VNetCritic.eval()
        self.PolicyActor.eval()

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        save_name = ['PolicyActor', 'PolicyActor_optimizer', 'VNetCritic', 'VNetCritic_optimizer']
        for _save_name in save_name:
            torch.save(getattr(self, _save_name), f"{path}/{_save_name}.pt")

    def load_model(self, path):
        load_name = ['PolicyActor', 'PolicyActor_optimizer', 'VNetCritic', 'VNetCritic_optimizer']
        for _load_name in load_name:
            setattr(self, _load_name, torch.load(f"{path}/{_load_name}.pt", map_location=torch.device(self.device)))

    def _t2n(self, x):
        return x.detach().cpu().numpy()

    def _flatten(self, T, N, x):
        return x.reshape(T * N, *x.shape[2:])

    def _cast(self, x):
        return x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])

    def _soft_update(self, tau, critic_models, target_critic_models):
        # update target critic
        for critic_var, target_critic_var in zip(critic_models.parameters(), target_critic_models.parameters()):
            target_critic_var.data.copy_(critic_var.data * tau + target_critic_var.data * (1.0 - tau))
