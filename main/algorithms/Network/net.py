r'''

network structure for rl algorithm

'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import algorithms.rl_utils.distributions as D
from torch.distributions.categorical import Categorical

class MLPCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activate_function='None', device='cpu', noise_layer=False):
        super().__init__()
        self.device = device

        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.fc3 = nn.Linear(hidden_dim, output_dim, device=device)
        if noise_layer:
            self.noise_layer = NoiseNet()

        if activate_function == 'tanh': self.activate_function = torch.nn.Tanh()
        elif activate_function == 'softmax': self.activate_function = torch.nn.Softmax()
        elif activate_function == 'None': self.activate_function = None

    def forward(self, inputs):
        outputs = torch.relu(self.fc1(inputs))
        outputs = torch.relu(self.fc2(outputs))
        outputs = self.fc3(outputs)

        if self.activate_function is not None:
            outputs = self.activate_fn(outputs)

        return outputs


class RNNMLPCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, activate_function='None', device='cpu', cfg=None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        if cfg['use_featureNorm']:
            self.featureNorm = nn.LayerNorm(input_dim)

        self.mlp = MLPCritic(input_dim,
                             hidden_dim,
                             hidden_dim,
                             activate_function=activate_function,
                             device=device,
                             noise_layer=False)
        if cfg['use_rnn_policy']:
            self.rnn = RNNNet(hidden_dim, hidden_dim, cfg['rnn_layer_dim'], cfg['use_orthogonal'])

        self.value_out = ValueHead(hidden_dim, output_dim, self.device, cfg['use_orthogonal'], cfg['use_popart'])
        self.to(device)

    def forward(self, inputs, last_hidden_state=None, mask=None):

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs).to(self.device)
        if last_hidden_state is not None:
            if not isinstance(last_hidden_state, torch.Tensor):
                last_hidden_state = torch.from_numpy(last_hidden_state).to(self.device)
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).to(self.device)

        if self.cfg['use_featureNorm']:
            inputs = self.featureNorm(inputs)
        outputs = self.mlp(inputs)
        if self.cfg['use_rnn_policy']:
            outputs, last_hidden_state = self.rnn(outputs, last_hidden_state, mask)

        value = self.value_out(outputs)

        return value, last_hidden_state

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activate_function='None', device='cpu'):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.fc3 = nn.Linear(hidden_dim, output_dim, device=device)

        if activate_function == 'tanh': self.activate_function = torch.nn.Tanh()
        elif activate_function == 'softmax': self.activate_function = torch.nn.Softmax()
        elif activate_function == 'None': self.activate_function = None

    def forward(self, inputs):
        outputs = torch.relu(self.fc1(inputs))
        outputs = torch.relu(self.fc2(outputs))
        mean, log_std = self.fc3(outputs)
        if not self.activate_function is None:
            mean = self.activate_fn(mean)

        log_std = torch.clip(log_std, -20, 2)

        actor_dist = torch.distributions.Normal(mean, torch.exp(log_std))

        return actor_dist


class DiscreteRNNActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activate_function='None', device='cpu', cfg=None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        if cfg['use_featureNorm']:
            self.featureNorm = nn.LayerNorm(input_dim, device=device)

        self.mlp = MLPCritic(input_dim,
                             hidden_dim,
                             hidden_dim,
                             activate_function=activate_function,
                             device=device,
                             noise_layer=False)
        if cfg['use_rnn_policy']:
            self.RNN = RNNNet(hidden_dim, hidden_dim, cfg['rnn_layer_dim'], cfg['use_orthogonal'])

        self.actor_layer = ActionHead('discrete',
                                      hidden_dim,
                                      output_dim,
                                      cfg['use_orthogonal'],
                                      self.device,
                                      cfg['last_action_layer_gain'])
        self.to(device)

    def forward(self, inputs, last_hidden_state=None, mask=None):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs).to(self.device)
        if last_hidden_state is not None:
            if not isinstance(last_hidden_state, torch.Tensor):
                last_hidden_state = torch.from_numpy(last_hidden_state).to(self.device)
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).to(self.device)

        if self.cfg['use_featureNorm']:
                inputs = self.featureNorm(inputs)
        outputs = self.mlp(inputs)
        if self.cfg['use_rnn_policy']:
            outputs, last_hidden_state = self.RNN(outputs, last_hidden_state, mask)

        dist = self.actor_layer(outputs)

        return dist, last_hidden_state

class DiscreteActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activate_function='None', device='cpu'):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.fc3 = nn.Linear(hidden_dim, output_dim, device=device)

        if activate_function == 'tanh': self.activate_function = torch.nn.Tanh()
        elif activate_function == 'softmax': self.activate_function = torch.nn.Softmax()
        elif activate_function == 'None': self.activate_function = None

    def forward(self, inputs):
        outputs = torch.relu(self.fc1(inputs))
        outputs = torch.relu(self.fc2(outputs))
        outputs = self.fc3(outputs)
        if not self.activate_function is None:
            outputs = self.activate_fn(outputs)

        dist = Categorical(logits=outputs)

        return dist

class SoftDiscreteActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, distribution_fn='None', device='cpu'):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.fc3 = nn.Linear(hidden_dim, output_dim, device=device)

        if distribution_fn == 'softCategorical': self.distribution_fn = D.SoftCategoricalPd
        elif distribution_fn == 'GumbelSoftCategorical': self.distribution_fn = D.GumbelSoftCategoricalPd
        else: raise NotImplementedError

    def forward(self, inputs):
        outputs = torch.relu(self.fc1(inputs))
        outputs = torch.relu(self.fc2(outputs))
        outputs = self.fc3(outputs)
        dist = self.distribution_fn(logits=outputs, device=self.device)
        return dist


class DistributionCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_atom, v_min, v_max, activate_function='None', device='cpu'):
        super().__init__()
        self.device = device
        self.n_atom = n_atom
        self.output_dim = output_dim
        self.v_min, self.v_max = v_min, v_max

        self.QDistrbution = MLPCritic(input_dim,
                                      hidden_dim,
                                      output_dim * n_atom,
                                      activate_function='None',
                                      device=self.device,
                                      noise_layer=False)

        if activate_function == 'tanh': self.activate_function = torch.nn.Tanh()
        elif activate_function == 'softmax': self.activate_function = torch.nn.Softmax()
        elif activate_function == 'None': self.activate_function = None

    def forward(self, inputs):
        outputs = self.QDistrbution(inputs)

        outputs = outputs.view(*outputs.shape[:-1], self.output_dim, self.n_atom)
        dist = D.SoftCategoricalPd(logits=outputs, device=self.device)
        dist = dist.sample()
        logit = dist * torch.linspace(self.v_min, self.v_max, self.n_atom).to(self.device)
        logit = logit.sum(dim=-1)
        return logit, dist


class RainbowCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_atom, v_min, v_max, activate_function='None', device='cpu'):
        super().__init__()
        self.device = device
        self.n_atom = n_atom
        self.output_dim = output_dim
        self.v_min, self.v_max = v_min, v_max

        self.QValues = MLPCritic(input_dim,
                                 hidden_dim,
                                 n_atom,
                                 activate_function=activate_function,
                                 device=self.device,
                                 noise_layer=False)

        self.ADistrbution = MLPCritic(input_dim,
                                      hidden_dim,
                                      output_dim * n_atom,
                                      activate_function=activate_function,
                                      device=self.device,
                                      noise_layer=False)

    def forward(self, inputs):
        q_value = self.QValues(inputs)
        a_dist = self.ADistrbution(inputs)
        a_dist = a_dist.view(*a_dist.shape[:-1], self.output_dim, self.n_atom)
        q_value = q_value.view(*q_value.shape[:-1], 1, self.n_atom)
        q_dist = q_value + a_dist - a_dist.mean(dim=-2, keepdims=True)
        q_dist = D.SoftCategoricalPd(logits=q_dist, device=self.device)
        q_dist = q_dist.sample()
        logit = q_dist * torch.linspace(self.v_min, self.v_max, self.n_atom).to(self.device)
        logit = logit.sum(dim=-1)

        return {'logit':logit, 'distribution':q_dist}


class NoiseNet(nn.Module):
    def __init__(self):
        super(NoiseNet, self).__init__()
        pass

    def forward(self, input):
        pass


class RNNNet(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_layer_dim=1, use_orthogonal=True, batch_first=False):
        super(RNNNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_layer_dim = rnn_layer_dim
        self.use_orthogonal = use_orthogonal
        self.batch_first = batch_first
        # rnn块
        self.rnn = nn.GRU(input_dim, output_dim, num_layers=rnn_layer_dim, batch_first=batch_first)
        # rnn.init
        for param_name, param_value in self.rnn.named_parameters():
            if 'biase' in param_name:
                nn.init.constant(param_value, 0)
            elif 'weight' in param_name:
                if self.use_orthogonal:
                    nn.init.orthogonal_(param_value)
                else:
                    nn.init.xavier_uniform_(param_value)
        # 层标准化
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs, last_hidden_state, mask_state=None):
        # input.shape = [timestep(T)==seq_len, batch_size(B), feature_dim(N)]
        # last_hidden_state.shape = [batch_size(B), rnn_layer_dim(L) * 1(单向), output_dim(O)]
        if len(last_hidden_state.shape) < 3:
            last_hidden_state = last_hidden_state.unsqueeze(dim=0)
        if inputs.size(0) == last_hidden_state.size(0):
            if not mask_state is None:
                last_hidden_state = (last_hidden_state *
                                mask_state.repeat(1, self.rnn_layer_dim).unsqueeze(dim=-1)).transpose(0, 1).contiguous()
            outputs, last_hidden_state = self.rnn(inputs.unsqueeze(dim=0), last_hidden_state)

            outputs = outputs.squeeze(dim=0)
            last_hidden_state = last_hidden_state.transpose(0, 1)
        else:  # input.shape = [T * B, N], last_hidden_state.shape = [B, L, O]
            batch_size = last_hidden_state.size(0)  # B
            timestep = int(inputs.size(0) / batch_size)  # T

            inputs = inputs.view(timestep, batch_size, inputs.size(-1))

            if not mask_state is None:
                mask_state = mask_state.view(timestep, batch_size)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((mask_state[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())
            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [timestep]

            last_hidden_state = last_hidden_state.transpose(0,1)  # last_hidden_state.shape = [L, B, O]
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_index = has_zeros[i]
                end_index = has_zeros[i + 1]
                temp = (last_hidden_state * mask_state[start_index].view(1, -1, 1).repeat(self.rnn_layer_dim, 1, 1)).contiguous()
                rnn_scores, last_hidden_state = self.rnn(inputs[start_index:end_index], temp)
                outputs.append(rnn_scores)

            # output.shape= [T, B, O]
            outputs = torch.cat(outputs, dim=0)

            # 展开
            outputs = outputs.reshape(timestep * batch_size, -1)
            last_hidden_state = last_hidden_state.transpose(0, 1) # [B, L, O]


        outputs = self.norm(outputs)

        return outputs, last_hidden_state


class ActionHead(nn.Module):
    def __init__(self, action_space, inputs_dim, output_dim, use_orthogonal, device='cpu', gain=0.1):
        super(ActionHead, self).__init__()
        self.device = device
        self.action_space = action_space
        self.inputs_dim = inputs_dim
        self.use_orthogonal = use_orthogonal
        self.gain = gain

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(inputs_dim, output_dim))

    def forward(self, inputs):
        logit = self.linear(inputs)  # 概率分布
        dist = Categorical(logits=logit)

        return dist

class ValueHead(nn.Module):
    def __init__(self, inputs_dim, output_dim, device='cpu', use_orthogonal=True, use_Popart=True):
        super(ValueHead, self).__init__()
        self.device = device
        self.inputs_dim = inputs_dim

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        if use_Popart:
            self.value = init_(PopArt(inputs_dim, output_dim, device=device))
        else:
            self.value = init_(nn.Linear(inputs_dim, output_dim))
        self.to(device)

    def forward(self, inputs):
        value = self.value(inputs)  # 概率密度
        return value


class PopArt(torch.nn.Module):
    def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)

        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    def forward(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        return F.linear(input_vector, self.weight, self.bias)

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        old_mean, old_var = self.debiased_mean_var()
        old_stddev = torch.sqrt(old_var)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        self.stddev = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)

        new_mean, new_var = self.debiased_mean_var()
        new_stddev = torch.sqrt(new_var)

        self.weight = self.weight * old_stddev / new_stddev
        self.bias = (old_stddev * self.bias + old_mean - new_mean) / new_stddev

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]

        return out

    def denormalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.debiased_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]

        out = out.cpu().numpy()

        return out


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module