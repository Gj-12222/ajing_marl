import torch 


class QuanParameter(torch.nn.Module):
    def __init__(self, num_bits=8):
        super(QuanParameter, self).__init__()
        self.num_bits = num_bits
        self.scale = None
        self.zero = None
        self.float_min = None
        self.float_max = None
        self.multiplier = None
        self.right_shift = None
        
    def dump(self):
        return {
            'num_bits': self.num_bits,
            'scale': self.scale,
            'zero': self.zero,
            'float_min': self.float_min,
            'float_max': self.float_max,
            'multiplier': self.multiplier,
            'right_shift': self.right_shift,
        }

    @staticmethod
    def load(dic):
        instance = QuanParameter()
        instance.num_bits = dic['num_bits']
        instance.scale = dic['scale']
        instance.zero = dic['zero']
        instance.float_min = dic['float_min']
        instance.float_max = dic['float_max']
        instance.multiplier = dic['multiplier']
        instance.right_shift = dic['right_shift']
        return instance

    def update_parameter(self, float_tensor):
        if self.float_min is None or self.float_min > float_tensor.min():
            self.float_min = float_tensor.min()
        if self.float_min > 0.:  # this is to ensure that float zero can be quantized, which is essential for relu
            self.float_min = 0.

        if self.float_max is None or self.float_max < float_tensor.max():
            self.float_max = float_tensor.max()
        if self.float_max < 0.:
            self.float_max = 0.

        if abs(self.float_max - self.float_min) < 0.01:
            self.float_max = self.float_min + 0.01  # to avoid numeric error

        self.scale, self.zero = self.get_scale_and_zero()

    def get_scale_and_zero(self):
        quan_min = - 2. ** (self.num_bits - 1)
        quan_max = 2. ** (self.num_bits - 1) - 1
        if self.float_min > 0:
            scale = self.float_max / quan_max
        elif self.float_max < 0:
            scale = self.float_min / quan_min
        else:
            scale = max(self.float_max / quan_max, self.float_min / quan_min)
        return scale, 0

    def quantize(self, float_tensor):
        quan_min = - 2. ** (self.num_bits - 1)
        quan_max = 2. ** (self.num_bits - 1) - 1
        quan_tensor = self.zero + float_tensor / self.scale
        quan_tensor.clamp_(quan_min, quan_max).round_()
        return quan_tensor.int()

    def dequantize(self, quan_tensor):
        return self.scale * (quan_tensor.float() - self.zero)

    def quan_and_dequan(self, float_tensor):
        return self.dequantize(self.quantize(float_tensor))
