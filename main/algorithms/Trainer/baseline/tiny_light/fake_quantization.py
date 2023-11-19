from torch.autograd import Function


class FakeQuant(Function):
    @staticmethod
    def forward(ctx, x, quan_param):
        x = quan_param.quan_and_dequan(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Shift(Function):
    @staticmethod
    def forward(ctx, x, shift_target):
        x = x + (shift_target - x).detach()
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
