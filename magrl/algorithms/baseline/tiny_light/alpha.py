import torch.nn
import torch.nn.functional as F


class Alpha(torch.nn.Module):
    EPS = 1e-9

    def __init__(self, elem_size, config):
        super(Alpha, self).__init__()
        self.elem_size = elem_size
        self.config = config
        self.alpha = torch.nn.Parameter(torch.ones(size=[self.elem_size]))
        self.is_frozen = False
        self.n_alive_idx_after_frozen = None  # only applicable after frozen

    def get_softmax_value(self):
        return F.softmax(self.alpha, dim=0)

    def get_alive_idx(self):
        return self.n_alive_idx_after_frozen

    def get_desc(self):
        return '{}, ent: {}'.format(
            '\t'.join(['{:.3f}'.format(elem) if elem > self.EPS else '-----' for elem in self.get_softmax_value().tolist()]),
            self.get_entropy()
        )

    def get_entropy(self):
        prob = self.get_softmax_value()
        ent = torch.sum(-prob * torch.log(prob))
        return ent

    def hard_threshold_and_freeze_alpha(self, num_alive_elem):
        self.is_frozen = True
        _, self.n_alive_idx_after_frozen = torch.topk(self.alpha, num_alive_elem)  # 返回 列表中最大的n个值
        self.alpha.detach_()
        for idx in range(self.elem_size):
            if idx not in self.n_alive_idx_after_frozen:
                self.alpha[idx] = torch.tensor([self.EPS])
            else:
                self.alpha[idx] = torch.tensor([100.0])
        return self.alpha

    def soft_threshold_freeze_alpha(self, num_alive_elem):
        self.is_frozen = True
        _, self.n_alive_idx_after_frozen = torch.topk(self.alpha, num_alive_elem)  # 返回 列表中最大的n个值
        self.alpha.detach_()
        for idx in range(self.elem_size):
            if idx not in self.n_alive_idx_after_frozen:
                self.alpha[idx] = torch.tensor([self.EPS])
            else:
                self.alpha[idx] = torch.tensor([100.0])
        self.alpha = F.softmax(self.alpha, dim=0)
        return self.alpha

    def freeze_alpha(self):
        self.is_frozen = True
        self.alpha.detach_()
        return self.alpha

    def unfreeze_alpha(self):
        self.is_frozen = False
        self.alpha.requires_grad_()
        return self.alpha