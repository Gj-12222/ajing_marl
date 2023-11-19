r'''

custom distribution

'''
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.utils import logits_to_probs


class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def logp(self, x):
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError

class GumbelSoftCategoricalPd(Pd):
    def __init__(self, logits, device):
        self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        self.device = device

    def flatparam(self):
        return self.logits
    def prob(self):
        return F.gumbel_softmax(self.logits, dim=-1)

    def cross_entropy(self, x):
        return -F.cross_entropy(self.logits, x)

    def kl(self, other):
        a0 = self.logits - torch.max(self.logits, dim=1, keepdim=True)
        a1 = other.logits - torch.max(other.logits, dim=1, keepdim=True)
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = torch.sum(ea0, dim=1, keepdim=True)
        z1 = torch.sum(ea1, dim=1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (a0 - torch.log(z0) - a1 + torch.log(z1)), dim=1)

    def entropy(self):
        a0 = self.logits - torch.max(self.logits, dim=1, keepdim=True)
        ea0 = torch.exp(a0)
        z0 = torch.sum(ea0, dim=1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (torch.log(z0) - a0), dim=1)

    def sample(self):
        u = torch.rand(self.logits.shape, device=self.device)
        prob = F.gumbel_softmax(self.logits - torch.log(-torch.log(u)), dim=-1)
        return prob

    def rsample(self):
        u = torch.rand(self.logits.shape, device=self.device)
        prob = F.gumbel_softmax(self.logits - torch.log(-torch.log(u)), dim=-1)
        return (torch.argmax(prob) - prob).detach() + prob

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class SoftCategoricalPd(Pd):
    def __init__(self, logits, device):
        self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        self.device = device
        self.probs = self.prob()

    def flatparam(self):
        return self.logits

    def prob(self):
        return F.softmax(self.logits, dim=-1)

    def log_prob(self, value):
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1).view(value.size(0), -1).sum(-1).unsqueeze(-1)

    def cross_entropy(self, x):
        return -F.cross_entropy(self.logits, x)

    def kl(self, other):
        a0 = self.logits - torch.max(self.logits, dim=1, keepdim=True)
        a1 = other.logits - torch.max(other.logits, dim=1, keepdim=True)
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = torch.sum(ea0, dim=1, keepdim=True)
        z1 = torch.sum(ea1, dim=1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (a0 - torch.log(z0) - a1 + torch.log(z1)), dim=1)

    def entropy(self):
        a0 = self.logits - torch.max(self.logits, dim=-1, keepdim=True)[0]
        ea0 = torch.exp(a0)
        z0 = torch.sum(ea0, dim=-1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (torch.log(z0) - a0), dim=-1, keepdim=True)

    def entropy_categorical(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.prob()
        return -p_log_p.sum(dim=-1, keepdim=True)

    def sample(self):
        u = torch.rand(self.logits.shape, device=self.device)  # 0-1
        return F.softmax(self.logits - torch.log(-torch.log(u)), dim=-1)

    # def argsample(self, sample_shape=torch.Size()):
    #     if not isinstance(sample_shape, torch.Size):
    #         sample_shape = torch.Size(sample_shape)
    #     probs_2d = self.probs.reshape(-1, self.logits.shape[-1])
    #     samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
    #     return samples_2d.reshape(self._extended_shape(sample_shape))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class SoftMultiCategoricalPd(Pd):  # doesn't work yet
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = torch.tensor(low, dtype=torch.float32)
        self.categoricals = list(map(SoftCategoricalPd, torch.split(flat, high - low + 1, dim=len(flat.get_shape()) - 1)))
    def flatparam(self):
        return self.flat
    def mode(self):
        x = []
        for i in range(len(self.categoricals)):
            x.append(self.low[i] + self.categoricals[i].mode())
        return torch.cat(x, dim=-1)
    def logp(self, x):
        return torch.add([p.logp(px) for p, px in zip(self.categoricals, torch.unbind(x - self.low, dim=len(x.get_shape()) - 1))])
    def kl(self, other):
        return torch.sum(torch.stack([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)]))
    def entropy(self):
        return torch.sum(torch.stack([p.entropy() for p in self.categoricals]))
    def sample(self):
        x = []
        for i in range(len(self.categoricals)):
            x.append(self.low[i] + self.categoricals[i].sample())
        return torch.cat(x, dim=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat=flat)

class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat  # flat = p = [mu ,logsted]
        mean, logstd = torch.split(flat, split_size_or_sections=2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)  #  (-20,2)
        self.mean = mean
        self.logstd = logstd
        self.std = torch.exp(logstd)
    def flatparam(self):
        return self.flat

    def logp(self, x):
        return - 0.5 * torch.sum(torch.square((x - self.mean) / self.std), dim=1) \
               - 0.5 * np.log(2.0 * np.pi) * float(x.shape[1]) \
               - torch.sum(self.logstd, dim=1)
    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return torch.sum(other.logstd - self.logstd + (torch.square(self.std) + torch.square(self.mean - other.mean)) / (2.0 * torch.square(other.std)) - 0.5, dim=1)
    def entropy(self):
        #  log var + 0.5* log(2*2π*e)
        return torch.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), 1)

    def sample(self):
        return torch.tanh(self.mean + self.std * torch.randn(self.mean.shape))

    def evaluation_sample(self):
        return torch.tanh(self.mean + self.std)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

    def log_gaussian_policy(self, act_resample, act_mu, act_logstd):
        MIN_LIM = 1e-8
        log_normal_sum = -0.5 * (((act_resample - act_mu) / (torch.exp(act_logstd) + MIN_LIM)) ** 2 + 2 * act_logstd + np.log(
            2 * np.pi))
        return torch.mean(log_normal_sum, dim=1)
    # 欧拉变换
    def euler_transformation(self, log_act_resample, act_resample):
        log_act_resample -= torch.mean(2 * (np.log(2) - act_resample - F.softplus(-2 * act_resample)), dim=1)
        return log_act_resample
    # reparameterization
    def reparameterization(self):
        return self.mean + self.std * torch.randn(self.mean.shape)

    def evaluation(self):
        return self.mean

class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self.ps = torch.sigmoid(logits)
    def flatparam(self):
        return self.logits
    def mode(self):
        return torch.round(self.ps)
    def logp(self, x: torch.Tensor):
        return - torch.sum(torch.nn.BCEWithLogitsLoss()(self.logits, x), dim=1)
    def kl(self, other):
        return torch.sum(torch.nn.BCEWithLogitsLoss()(other.logits, self.ps), dim=1) - \
               torch.sum(torch.nn.BCEWithLogitsLoss()(self.logits, self.ps), dim=1)
    def entropy(self):
        return torch.sum(torch.nn.BCEWithLogitsLoss()(self.logits, self.ps), dim=1)
    def sample(self):
        p = torch.sigmoid(self.logits)
        u = torch.rand(p.shape)
        return torch.lt(u, p)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)