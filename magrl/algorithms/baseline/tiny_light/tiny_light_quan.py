import random
import numpy as np
from .. import TinyLight
import torch
from ..tiny_light.quan_parameter import QuanParameter
import torch.nn.functional as F
from ..tiny_light.fake_quantization import FakeQuant, Shift
import importlib.util
import platform


class TinyLightQuan(TinyLight):
    def __init__(self, config, env, idx):
        super(TinyLightQuan, self).__init__(config, env, idx)
        if 'mcu_folder' in config.keys():
            if platform.system() == 'Darwin':
                image_path = '{}/flow_{}_agent_{}.cpython-37m-darwin.so'.format(config['mcu_folder'], config['flow_idx'], idx)
            else:
                image_path = '{}/flow_{}_agent_{}.cpython-36m-x86_64-linux-gnu.so'.format(config['mcu_folder'], config['flow_idx'], idx)

            self.mcu = importlib.util.module_from_spec(
                importlib.util.spec_from_file_location('flow_{}_agent_{}'.format(config['flow_idx'], idx), image_path)
            )

    def load(self, model_path):
        super(TinyLightQuan, self).load(model_path)

        assert len(self.n_alpha[0].n_alive_idx_after_frozen) == 2
        assert len(self.n_alpha[1].n_alive_idx_after_frozen) == 1
        assert len(self.n_alpha[2].n_alive_idx_after_frozen) == 1

        # restore alive index
        self.input_alive_idx = self.n_alpha[0].n_alive_idx_after_frozen
        self.layer_1_alive_idx = self.n_alpha[1].n_alive_idx_after_frozen[0]
        self.layer_2_alive_idx = self.n_alpha[2].n_alive_idx_after_frozen[0]

        # quantization params
        self.quan_input_0 = QuanParameter(self.cur_agent['num_bits'])
        self.quan_input_2_first_layer_w_0 = QuanParameter(self.cur_agent['num_bits'])
        self.quan_input_2_first_layer_b_0 = QuanParameter(self.cur_agent['num_bits'])
        self.quan_layer_1_sub0 = QuanParameter(self.cur_agent['num_bits'])

        self.quan_input_1 = QuanParameter(self.cur_agent['num_bits'])
        self.quan_input_2_first_layer_w_1 = QuanParameter(self.cur_agent['num_bits'])
        self.quan_input_2_first_layer_b_1 = QuanParameter(self.cur_agent['num_bits'])
        self.quan_layer_1_sub1 = QuanParameter(self.cur_agent['num_bits'])

        self.quan_layer_1_sum = QuanParameter(self.cur_agent['num_bits'])

        self.quan_first_layer_2_second_layer_w = QuanParameter(self.cur_agent['num_bits'])
        self.quan_first_layer_2_second_layer_b = QuanParameter(self.cur_agent['num_bits'])
        self.quan_layer_2 = QuanParameter(self.cur_agent['num_bits'])

        self.quan_second_layer_2_output_w = QuanParameter(self.cur_agent['num_bits'])
        self.quan_second_layer_2_output_b = QuanParameter(self.cur_agent['num_bits'])
        self.quan_output = QuanParameter(self.cur_agent['num_bits'])

        # parameters to be quantized
        # -- input 2 first layer
        self.input_2_first_layer_w_0 = self.network_local.input_2_first_layer[self.input_alive_idx[0]][self.layer_1_alive_idx][0].weight
        self.input_2_first_layer_b_0 = self.network_local.input_2_first_layer[self.input_alive_idx[0]][self.layer_1_alive_idx][0].bias
        self.input_2_first_layer_w_1 = self.network_local.input_2_first_layer[self.input_alive_idx[1]][self.layer_1_alive_idx][0].weight
        self.input_2_first_layer_b_1 = self.network_local.input_2_first_layer[self.input_alive_idx[1]][self.layer_1_alive_idx][0].bias
        # -- first layer to second layer
        self.first_layer_2_second_layer_w = self.network_local.first_layer_2_second_layer[self.layer_1_alive_idx][self.layer_2_alive_idx][0].weight
        self.first_layer_2_second_layer_b = self.network_local.first_layer_2_second_layer[self.layer_1_alive_idx][self.layer_2_alive_idx][0].bias
        # -- second layer to last layer
        self.second_layer_2_output_w = self.network_local.second_layer_2_last_layer[self.layer_2_alive_idx].weight
        self.second_layer_2_output_b = self.network_local.second_layer_2_last_layer[self.layer_2_alive_idx].bias

        # update quantize network parameters
        self.quan_input_2_first_layer_w_0.update_parameter(self.input_2_first_layer_w_0)
        self.quan_input_2_first_layer_b_0.update_parameter(self.input_2_first_layer_b_0)
        self.quan_input_2_first_layer_w_1.update_parameter(self.input_2_first_layer_w_1)
        self.quan_input_2_first_layer_b_1.update_parameter(self.input_2_first_layer_b_1)
        self.quan_first_layer_2_second_layer_w.update_parameter(self.first_layer_2_second_layer_w)
        self.quan_first_layer_2_second_layer_b.update_parameter(self.first_layer_2_second_layer_b)
        self.quan_second_layer_2_output_w.update_parameter(self.second_layer_2_output_w)
        self.quan_second_layer_2_output_b.update_parameter(self.second_layer_2_output_b)

    def generate(self):
        self.dim_input_0 = self.n_input_feature_dim[self.input_alive_idx[0]]
        self.dim_input_1 = self.n_input_feature_dim[self.input_alive_idx[1]]
        self.dim_layer_1 = self.n_layer_1_dim[self.layer_1_alive_idx]
        self.dim_layer_2 = self.n_layer_2_dim[self.layer_2_alive_idx]
        self.dim_output = self.num_phase

        self.quan_input_2_first_layer_w_0.multiplier, self.quan_input_2_first_layer_w_0.right_shift = self._quantize_multiplier(
            self.quan_input_0.scale * self.quan_input_2_first_layer_w_0.scale / self.quan_layer_1_sub0.scale
        )
        self.quan_input_2_first_layer_b_0.multiplier, self.quan_input_2_first_layer_b_0.right_shift = self._quantize_multiplier(
            self.quan_input_2_first_layer_b_0.scale / self.quan_layer_1_sub0.scale
        )
        self.quan_input_2_first_layer_w_1.multiplier, self.quan_input_2_first_layer_w_1.right_shift = self._quantize_multiplier(
            self.quan_input_1.scale * self.quan_input_2_first_layer_w_1.scale / self.quan_layer_1_sub1.scale
        )
        self.quan_input_2_first_layer_b_1.multiplier, self.quan_input_2_first_layer_b_1.right_shift = self._quantize_multiplier(
            self.quan_input_2_first_layer_b_1.scale / self.quan_layer_1_sub1.scale
        )

        self.quan_layer_1_sub0.multiplier, self.quan_layer_1_sub0.right_shift = self._quantize_multiplier(
            self.quan_layer_1_sub0.scale / self.quan_layer_1_sum.scale
        )
        self.quan_layer_1_sub1.multiplier, self.quan_layer_1_sub1.right_shift = self._quantize_multiplier(
            self.quan_layer_1_sub1.scale / self.quan_layer_1_sum.scale
        )

        self.quan_first_layer_2_second_layer_w.multiplier, self.quan_first_layer_2_second_layer_w.right_shift = self._quantize_multiplier(
            self.quan_layer_1_sum.scale * self.quan_first_layer_2_second_layer_w.scale / self.quan_layer_2.scale
        )
        self.quan_first_layer_2_second_layer_b.multiplier, self.quan_first_layer_2_second_layer_b.right_shift = self._quantize_multiplier(
            self.quan_first_layer_2_second_layer_b.scale / self.quan_layer_2.scale
        )
        self.quan_second_layer_2_output_w.multiplier, self.quan_second_layer_2_output_w.right_shift = self._quantize_multiplier(
            self.quan_layer_2.scale * self.quan_second_layer_2_output_w.scale / self.quan_output.scale
        )
        self.quan_second_layer_2_output_b.multiplier, self.quan_second_layer_2_output_b.right_shift = self._quantize_multiplier(
            self.quan_second_layer_2_output_b.scale / self.quan_output.scale
        )

    def _quantize_multiplier(self, real_multiplier):
        assert real_multiplier > 0.
        right_shift = 31  # there are two sign bits for the result of int_32 * int_32
        if 0. < real_multiplier < 1.:
            while real_multiplier < 0.5:
                real_multiplier *= 2.
                right_shift += 1
            q = int(real_multiplier * (2 ** 31))
            assert q <= 2 ** 31
            if q == 2 ** 31:
                q /= 2
                right_shift -= 1
        else:
            while real_multiplier > 1.:
                real_multiplier /= 2.
                right_shift -= 1
            q = int(real_multiplier * (2 ** 31))
            assert q <= 2 ** 31
            if q == 2 ** 31:
                q /= 2
                right_shift -= 1
        return q, right_shift

    def mcu_pick_action(self, n_obs):
        obs = n_obs[self.idx]
        quan_obs_0 = []
        for obs_0 in obs[self.input_alive_idx[0]].reshape([-1]):
            quan_obs_0.append(self.mcu.quan_feature_0(obs_0))
        quan_obs_1 = []
        for obs_1 in obs[self.input_alive_idx[1]].reshape([-1]):
            quan_obs_1.append(self.mcu.quan_feature_1(obs_1))

        str_obs = ','.join(str(e) for e in quan_obs_0) + ';' + ','.join(str(e) for e in quan_obs_1) + '.'
        action = self.mcu.pick_action(str_obs)

        return action

    def pick_action(self, n_obs, on_training):
        obs = n_obs[self.idx]
        self.quan_input_0.update_parameter(obs[self.input_alive_idx[0]])
        self.quan_input_1.update_parameter(obs[self.input_alive_idx[1]])

        self.network_local.eval()
        with torch.no_grad():
            layer_1_sub0 = F.relu(F.linear(obs[self.input_alive_idx[0]], self.input_2_first_layer_w_0, self.input_2_first_layer_b_0))
            layer_1_sub1 = F.relu(F.linear(obs[self.input_alive_idx[1]], self.input_2_first_layer_w_1, self.input_2_first_layer_b_1))
            self.quan_layer_1_sub0.update_parameter(layer_1_sub0)
            self.quan_layer_1_sub1.update_parameter(layer_1_sub1)

            res_first_layer = layer_1_sub0 + layer_1_sub1
            self.quan_layer_1_sum.update_parameter(res_first_layer)

            res_second_layer = F.relu(F.linear(res_first_layer, self.first_layer_2_second_layer_w, self.first_layer_2_second_layer_b))
            self.quan_layer_2.update_parameter(res_second_layer)

            res_last_layer = F.linear(res_second_layer, self.second_layer_2_output_w, self.second_layer_2_output_b)
            self.quan_output.update_parameter(res_last_layer)

            self.current_phase = torch.argmax(res_last_layer, dim=1).cpu().item()

            if on_training and random.random() < self.cur_agent['epsilon']:
                self.current_phase = random.randint(0, self.num_phase - 1)
        self.network_local.train()
        return self.current_phase

    def _compute_critic_loss(self, obs, act, rew, next_obs, done):
        self.generate()
        # gather q target
        with torch.no_grad():
            layer_1_sub0 = F.relu(F.linear(
                FakeQuant.apply(obs[self.input_alive_idx[0]], self.quan_input_0),
                FakeQuant.apply(self.network_target.input_2_first_layer[self.input_alive_idx[0]][self.layer_1_alive_idx][0].weight, self.quan_input_2_first_layer_w_0),
                FakeQuant.apply(self.network_target.input_2_first_layer[self.input_alive_idx[0]][self.layer_1_alive_idx][0].bias, self.quan_input_2_first_layer_b_0)
            ))
            q_layer_1_sub0 = self.quan_layer_1_sub0.dequantize(
                torch.tensor(self.b_quan_linear_relu(
                    lhs=self.quan_input_0.quantize(obs[self.input_alive_idx[0]]).numpy(),
                    rhs=self.quan_input_2_first_layer_w_0.quantize(self.network_target.input_2_first_layer[self.input_alive_idx[0]][self.layer_1_alive_idx][0].weight.T).numpy(),
                    bias=self.quan_input_2_first_layer_b_0.quantize(self.network_target.input_2_first_layer[self.input_alive_idx[0]][self.layer_1_alive_idx][0].bias).numpy(),
                    quan_multiplier_w=self.quan_input_2_first_layer_w_0.multiplier,
                    right_shift_w=self.quan_input_2_first_layer_w_0.right_shift,
                    quan_multiplier_b=self.quan_input_2_first_layer_b_0.multiplier,
                    right_shift_b=self.quan_input_2_first_layer_b_0.right_shift,
                ))
            )
            self.quan_layer_1_sub0.update_parameter(layer_1_sub0)
            layer_1_sub0 = Shift.apply(layer_1_sub0, q_layer_1_sub0)
            self.quan_layer_1_sub0.update_parameter(layer_1_sub0)

            layer_1_sub1 = F.relu(F.linear(
                FakeQuant.apply(obs[self.input_alive_idx[1]], self.quan_input_1),
                FakeQuant.apply(self.network_target.input_2_first_layer[self.input_alive_idx[1]][self.layer_1_alive_idx][0].weight, self.quan_input_2_first_layer_w_1),
                FakeQuant.apply(self.network_target.input_2_first_layer[self.input_alive_idx[1]][self.layer_1_alive_idx][0].bias, self.quan_input_2_first_layer_b_1)
            ))

            q_layer_1_sub1 = self.quan_layer_1_sub1.dequantize(
                torch.tensor(self.b_quan_linear_relu(
                    lhs=self.quan_input_1.quantize(obs[self.input_alive_idx[1]]).numpy(),
                    rhs=self.quan_input_2_first_layer_w_1.quantize(self.network_target.input_2_first_layer[self.input_alive_idx[1]][self.layer_1_alive_idx][0].weight.T).numpy(),
                    bias=self.quan_input_2_first_layer_b_1.quantize(self.network_target.input_2_first_layer[self.input_alive_idx[1]][self.layer_1_alive_idx][0].bias).numpy(),
                    quan_multiplier_w=self.quan_input_2_first_layer_w_1.multiplier,
                    right_shift_w=self.quan_input_2_first_layer_w_1.right_shift,
                    quan_multiplier_b=self.quan_input_2_first_layer_b_1.multiplier,
                    right_shift_b=self.quan_input_2_first_layer_b_1.right_shift,
                ))
            )
            self.quan_layer_1_sub1.update_parameter(layer_1_sub1)
            layer_1_sub1 = Shift.apply(layer_1_sub1, q_layer_1_sub1)
            self.quan_layer_1_sub1.update_parameter(layer_1_sub1)

            res_first_layer = FakeQuant.apply(layer_1_sub0, self.quan_layer_1_sub0) + FakeQuant.apply(layer_1_sub1, self.quan_layer_1_sub1)
            q_res_first_layer = self.quan_layer_1_sum.dequantize(
                torch.tensor(self.b_quan_add(
                    lhs=self.quan_layer_1_sub0.quantize(layer_1_sub0).numpy(),
                    rhs=self.quan_layer_1_sub1.quantize(layer_1_sub1).numpy(),
                    quan_multiplier_0=self.quan_layer_1_sub0.multiplier,
                    right_shift_0=self.quan_layer_1_sub0.right_shift,
                    quan_multiplier_1=self.quan_layer_1_sub1.multiplier,
                    right_shift_1=self.quan_layer_1_sub1.right_shift,
                ))
            )
            self.quan_layer_1_sum.update_parameter(res_first_layer)
            res_first_layer = Shift.apply(res_first_layer, q_res_first_layer)
            self.quan_layer_1_sum.update_parameter(res_first_layer)

            res_second_layer = F.relu(F.linear(
                FakeQuant.apply(res_first_layer, self.quan_layer_1_sum),
                FakeQuant.apply(self.network_target.first_layer_2_second_layer[self.layer_1_alive_idx][self.layer_2_alive_idx][0].weight, self.quan_first_layer_2_second_layer_w),
                FakeQuant.apply(self.network_target.first_layer_2_second_layer[self.layer_1_alive_idx][self.layer_2_alive_idx][0].bias, self.quan_first_layer_2_second_layer_b)
            ))
            q_res_second_layer = self.quan_layer_2.dequantize(
                torch.tensor(self.b_quan_linear_relu(
                    lhs=self.quan_layer_1_sum.quantize(res_first_layer).numpy(),
                    rhs=self.quan_first_layer_2_second_layer_w.quantize(self.network_target.first_layer_2_second_layer[self.layer_1_alive_idx][self.layer_2_alive_idx][0].weight.T).numpy(),
                    bias=self.quan_first_layer_2_second_layer_b.quantize(self.network_target.first_layer_2_second_layer[self.layer_1_alive_idx][self.layer_2_alive_idx][0].bias).numpy(),
                    quan_multiplier_w=self.quan_first_layer_2_second_layer_w.multiplier,
                    right_shift_w=self.quan_first_layer_2_second_layer_w.right_shift,
                    quan_multiplier_b=self.quan_first_layer_2_second_layer_b.multiplier,
                    right_shift_b=self.quan_first_layer_2_second_layer_b.right_shift,
                ))
            )
            self.quan_layer_2.update_parameter(res_second_layer)
            res_second_layer = Shift.apply(res_second_layer, q_res_second_layer)
            self.quan_layer_2.update_parameter(res_second_layer)

            res_last_layer = F.linear(
                FakeQuant.apply(res_second_layer, self.quan_layer_2),
                FakeQuant.apply(self.network_target.second_layer_2_last_layer[self.layer_2_alive_idx].weight, self.quan_second_layer_2_output_w),
                FakeQuant.apply(self.network_target.second_layer_2_last_layer[self.layer_2_alive_idx].bias, self.quan_second_layer_2_output_b)
            )
            q_res_last_layer = self.quan_output.dequantize(
                torch.tensor(self.b_quan_linear(
                    lhs=self.quan_layer_2.quantize(res_second_layer).numpy(),
                    rhs=self.quan_second_layer_2_output_w.quantize(self.network_target.second_layer_2_last_layer[self.layer_2_alive_idx].weight.T).numpy(),
                    bias=self.quan_second_layer_2_output_b.quantize(self.network_target.second_layer_2_last_layer[self.layer_2_alive_idx].bias).numpy(),
                    quan_multiplier_w=self.quan_second_layer_2_output_w.multiplier,
                    right_shift_w=self.quan_second_layer_2_output_w.right_shift,
                    quan_multiplier_b=self.quan_second_layer_2_output_b.multiplier,
                    right_shift_b=self.quan_second_layer_2_output_b.right_shift,
                ))
            )
            self.quan_output.update_parameter(res_last_layer)
            res_last_layer = Shift.apply(res_last_layer, q_res_last_layer)
            self.quan_output.update_parameter(res_last_layer)

            q_target = rew + self.gamma * torch.max(res_last_layer, dim=1, keepdim=True)[0] * (~done)

        # gather q expected
        layer_1_sub0 = F.relu(F.linear(
            FakeQuant.apply(obs[self.input_alive_idx[0]], self.quan_input_0),
            FakeQuant.apply(self.input_2_first_layer_w_0, self.quan_input_2_first_layer_w_0),
            FakeQuant.apply(self.input_2_first_layer_b_0, self.quan_input_2_first_layer_b_0)
        ))
        q_layer_1_sub0 = self.quan_layer_1_sub0.dequantize(
            torch.tensor(self.b_quan_linear_relu(
                lhs=self.quan_input_0.quantize(obs[self.input_alive_idx[0]]).numpy(),
                rhs=self.quan_input_2_first_layer_w_0.quantize(self.input_2_first_layer_w_0.T).numpy(),
                bias=self.quan_input_2_first_layer_b_0.quantize(self.input_2_first_layer_b_0).numpy(),
                quan_multiplier_w=self.quan_input_2_first_layer_w_0.multiplier,
                right_shift_w=self.quan_input_2_first_layer_w_0.right_shift,
                quan_multiplier_b=self.quan_input_2_first_layer_b_0.multiplier,
                right_shift_b=self.quan_input_2_first_layer_b_0.right_shift,
            ))
        )
        self.quan_layer_1_sub0.update_parameter(layer_1_sub0)
        layer_1_sub0 = Shift.apply(layer_1_sub0, q_layer_1_sub0)
        self.quan_layer_1_sub0.update_parameter(layer_1_sub0)

        layer_1_sub1 = F.relu(F.linear(
            FakeQuant.apply(obs[self.input_alive_idx[1]], self.quan_input_1),
            FakeQuant.apply(self.input_2_first_layer_w_1, self.quan_input_2_first_layer_w_1),
            FakeQuant.apply(self.input_2_first_layer_b_1, self.quan_input_2_first_layer_b_1)
        ))
        q_layer_1_sub1 = self.quan_layer_1_sub1.dequantize(
            torch.tensor(self.b_quan_linear_relu(
                lhs=self.quan_input_1.quantize(obs[self.input_alive_idx[1]]).numpy(),
                rhs=self.quan_input_2_first_layer_w_1.quantize(self.input_2_first_layer_w_1.T).numpy(),
                bias=self.quan_input_2_first_layer_b_1.quantize(self.input_2_first_layer_b_1).numpy(),
                quan_multiplier_w=self.quan_input_2_first_layer_w_1.multiplier,
                right_shift_w=self.quan_input_2_first_layer_w_1.right_shift,
                quan_multiplier_b=self.quan_input_2_first_layer_b_1.multiplier,
                right_shift_b=self.quan_input_2_first_layer_b_1.right_shift,
            ))
        )
        self.quan_layer_1_sub1.update_parameter(layer_1_sub1)
        layer_1_sub1 = Shift.apply(layer_1_sub1, q_layer_1_sub1)
        self.quan_layer_1_sub1.update_parameter(layer_1_sub1)

        res_first_layer = FakeQuant.apply(layer_1_sub0, self.quan_layer_1_sub0) + FakeQuant.apply(layer_1_sub1, self.quan_layer_1_sub1)
        q_res_first_layer = self.quan_layer_1_sum.dequantize(
            torch.tensor(self.b_quan_add(
                lhs=self.quan_layer_1_sub0.quantize(layer_1_sub0).numpy(),
                rhs=self.quan_layer_1_sub1.quantize(layer_1_sub1).numpy(),
                quan_multiplier_0=self.quan_layer_1_sub0.multiplier,
                right_shift_0=self.quan_layer_1_sub0.right_shift,
                quan_multiplier_1=self.quan_layer_1_sub1.multiplier,
                right_shift_1=self.quan_layer_1_sub1.right_shift,
            ))
        )
        self.quan_layer_1_sum.update_parameter(res_first_layer)
        res_first_layer = Shift.apply(res_first_layer, q_res_first_layer)
        self.quan_layer_1_sum.update_parameter(res_first_layer)

        res_second_layer = F.relu(F.linear(
            FakeQuant.apply(res_first_layer, self.quan_layer_1_sum),
            FakeQuant.apply(self.first_layer_2_second_layer_w, self.quan_first_layer_2_second_layer_w),
            FakeQuant.apply(self.first_layer_2_second_layer_b, self.quan_first_layer_2_second_layer_b)
        ))
        q_res_second_layer = self.quan_layer_2.dequantize(
            torch.tensor(self.b_quan_linear_relu(
                lhs=self.quan_layer_1_sum.quantize(res_first_layer).numpy(),
                rhs=self.quan_first_layer_2_second_layer_w.quantize(self.first_layer_2_second_layer_w.T).numpy(),
                bias=self.quan_first_layer_2_second_layer_b.quantize(self.first_layer_2_second_layer_b).numpy(),
                quan_multiplier_w=self.quan_first_layer_2_second_layer_w.multiplier,
                right_shift_w=self.quan_first_layer_2_second_layer_w.right_shift,
                quan_multiplier_b=self.quan_first_layer_2_second_layer_b.multiplier,
                right_shift_b=self.quan_first_layer_2_second_layer_b.right_shift,
            ))
        )
        self.quan_layer_2.update_parameter(res_second_layer)
        res_second_layer = Shift.apply(res_second_layer, q_res_second_layer)
        self.quan_layer_2.update_parameter(res_second_layer)

        res_last_layer = F.linear(
            FakeQuant.apply(res_second_layer, self.quan_output),
            FakeQuant.apply(self.second_layer_2_output_w, self.quan_second_layer_2_output_w),
            FakeQuant.apply(self.second_layer_2_output_b, self.quan_second_layer_2_output_b)
        )
        q_res_last_layer = self.quan_output.dequantize(
            torch.tensor(self.b_quan_linear(
                lhs=self.quan_layer_2.quantize(res_second_layer).numpy(),
                rhs=self.quan_second_layer_2_output_w.quantize(self.second_layer_2_output_w.T).numpy(),
                bias=self.quan_second_layer_2_output_b.quantize(self.second_layer_2_output_b).numpy(),
                quan_multiplier_w=self.quan_second_layer_2_output_w.multiplier,
                right_shift_w=self.quan_second_layer_2_output_w.right_shift,
                quan_multiplier_b=self.quan_second_layer_2_output_b.multiplier,
                right_shift_b=self.quan_second_layer_2_output_b.right_shift,
            ))
        )
        self.quan_output.update_parameter(res_last_layer)
        res_last_layer = Shift.apply(res_last_layer, q_res_last_layer)
        self.quan_output.update_parameter(res_last_layer)

        q_expected = res_last_layer.gather(1, act.long())

        critic_loss = F.mse_loss(q_expected, q_target)
        return critic_loss

    def quan_pick_action(self, n_obs, on_training):
        obs = n_obs[self.idx]
        quan_input_0 = self.quan_input_0.quantize(obs[self.input_alive_idx[0]])

        res_layer_1_0 = np.zeros([self.dim_layer_1])
        self.quan_linear_relu(lhs=quan_input_0.numpy().astype(np.int16).reshape([-1]),
                              rhs=self.quan_input_2_first_layer_w_0.quantize(self.input_2_first_layer_w_0.T).reshape([-1]).numpy().astype(np.int16),
                              bias=self.quan_input_2_first_layer_b_0.quantize(self.input_2_first_layer_b_0).reshape([-1]).numpy().astype(np.int16),
                              result=res_layer_1_0,
                              quan_multiplier_w=self.quan_input_2_first_layer_w_0.multiplier,
                              right_shift_w=self.quan_input_2_first_layer_w_0.right_shift,
                              quan_multiplier_b=self.quan_input_2_first_layer_b_0.multiplier,
                              right_shift_b=self.quan_input_2_first_layer_b_0.right_shift,
                              depth=self.dim_input_0,
                              col=self.dim_layer_1)

        quan_input_1 = self.quan_input_1.quantize(obs[self.input_alive_idx[1]])
        res_layer_1_1 = np.zeros([self.dim_layer_1])
        self.quan_linear_relu(lhs=quan_input_1.numpy().astype(np.int16).reshape([-1]),
                              rhs=self.quan_input_2_first_layer_w_1.quantize(self.input_2_first_layer_w_1.T).reshape([-1]).numpy().astype(np.int16),
                              bias=self.quan_input_2_first_layer_b_1.quantize(self.input_2_first_layer_b_1).reshape([-1]).numpy().astype(np.int16),
                              result=res_layer_1_1,
                              quan_multiplier_w=self.quan_input_2_first_layer_w_1.multiplier,
                              right_shift_w=self.quan_input_2_first_layer_w_1.right_shift,
                              quan_multiplier_b=self.quan_input_2_first_layer_b_1.multiplier,
                              right_shift_b=self.quan_input_2_first_layer_b_1.right_shift,
                              depth=self.dim_input_1,
                              col=self.dim_layer_1)

        res_layer_1_sum = np.zeros([self.dim_layer_1])
        self.quan_add(lhs=res_layer_1_0.astype(np.int16),
                      rhs=res_layer_1_1.astype(np.int16),
                      result=res_layer_1_sum,
                      quan_multiplier_0=self.quan_layer_1_sub0.multiplier,
                      right_shift_0=self.quan_layer_1_sub0.right_shift,
                      quan_multiplier_1=self.quan_layer_1_sub1.multiplier,
                      right_shift_1=self.quan_layer_1_sub1.right_shift,
                      col=self.dim_layer_1
                      )

        res_layer_2 = np.zeros([self.dim_layer_2])
        self.quan_linear_relu(lhs=res_layer_1_sum,
                              rhs=self.quan_first_layer_2_second_layer_w.quantize(self.first_layer_2_second_layer_w.T).reshape([-1]).numpy().astype(np.int16),
                              bias=self.quan_first_layer_2_second_layer_b.quantize(self.first_layer_2_second_layer_b).reshape([-1]).numpy().astype(np.int16),
                              result=res_layer_2,
                              quan_multiplier_w=self.quan_first_layer_2_second_layer_w.multiplier,
                              right_shift_w=self.quan_first_layer_2_second_layer_w.right_shift,
                              quan_multiplier_b=self.quan_first_layer_2_second_layer_b.multiplier,
                              right_shift_b=self.quan_first_layer_2_second_layer_b.right_shift,
                              depth=self.dim_layer_1,
                              col=self.dim_layer_2)

        res_output = np.zeros([self.num_phase])
        self.quan_linear(lhs=res_layer_2,
                         rhs=self.quan_second_layer_2_output_w.quantize(self.second_layer_2_output_w.T).reshape([-1]).numpy().astype(np.int16),
                         bias=self.quan_second_layer_2_output_b.quantize(self.second_layer_2_output_b).reshape([-1]).numpy().astype(np.int16),
                         result=res_output,
                         quan_multiplier_w=self.quan_second_layer_2_output_w.multiplier,
                         right_shift_w=self.quan_second_layer_2_output_w.right_shift,
                         quan_multiplier_b=self.quan_second_layer_2_output_b.multiplier,
                         right_shift_b=self.quan_second_layer_2_output_b.right_shift,
                         depth=self.dim_layer_2,
                         col=self.num_phase)
        action = np.argmax(res_output)
        if on_training and random.random() < self.cur_agent['epsilon']:
            action = random.randint(0, self.num_phase - 1)
        self.current_phase = action
        return action

    def quan_linear_relu(self, lhs, rhs, bias, result,
                         quan_multiplier_w, right_shift_w, quan_multiplier_b, right_shift_b,
                         depth, col):
        for col_idx in range(col):
            rhs_flatten_idx = col_idx
            temp = np.int64(0)
            for depth_idx in range(depth):
                temp += lhs[depth_idx].astype(np.int64) * rhs[rhs_flatten_idx].astype(np.int64)
                rhs_flatten_idx += col
            temp = self._multiply_multiplier(temp, np.int64(quan_multiplier_w), right_shift_w)

            temp2 = self._multiply_multiplier(np.int64(bias[col_idx]), np.int64(quan_multiplier_b), right_shift_b)
            temp += temp2

            temp = max(0, temp)
            result[col_idx] = temp

    def b_quan_linear_relu(self, lhs, rhs, bias,
                           quan_multiplier_w, right_shift_w, quan_multiplier_b, right_shift_b):
        temp1 = np.matmul(lhs, rhs).astype(np.int64)
        temp1 = self._multiply_multiplier(temp1, np.int64(quan_multiplier_w), right_shift_w)
        temp2 = self._multiply_multiplier(bias, np.int64(quan_multiplier_b), right_shift_b)
        temp = temp1 + temp2
        temp[temp < 0] = 0
        return temp

    def quan_linear(self, lhs, rhs, bias, result,
                    quan_multiplier_w, right_shift_w, quan_multiplier_b, right_shift_b,
                    depth, col):
        for col_idx in range(col):
            rhs_flatten_idx = col_idx
            temp = np.int64(0)
            for depth_idx in range(depth):
                temp += lhs[depth_idx].astype(np.int64) * rhs[rhs_flatten_idx].astype(np.int64)
                rhs_flatten_idx += col
            temp = self._multiply_multiplier(temp, np.int64(quan_multiplier_w), right_shift_w)

            temp2 = self._multiply_multiplier(np.int64(bias[col_idx]), np.int64(quan_multiplier_b), right_shift_b)
            temp += temp2
            result[col_idx] = temp

    def b_quan_linear(self, lhs, rhs, bias,
                      quan_multiplier_w, right_shift_w, quan_multiplier_b, right_shift_b):
        temp1 = np.matmul(lhs, rhs).astype(np.int64)
        temp1 = self._multiply_multiplier(temp1, np.int64(quan_multiplier_w), right_shift_w)
        temp2 = self._multiply_multiplier(bias, np.int64(quan_multiplier_b), right_shift_b)
        temp = temp1 + temp2
        return temp

    def quan_add(self, lhs, rhs, result,
                 quan_multiplier_0, right_shift_0, quan_multiplier_1, right_shift_1, col):
        for col_idx in range(col):
            temp0 = lhs[col_idx]
            temp0 = self._multiply_multiplier(temp0, np.int64(quan_multiplier_0), right_shift_0)

            temp1 = rhs[col_idx]
            temp1 = self._multiply_multiplier(temp1, np.int64(quan_multiplier_1), right_shift_1)

            temp = temp0 + temp1
            result[col_idx] = temp

    def b_quan_add(self, lhs, rhs, quan_multiplier_0, right_shift_0, quan_multiplier_1, right_shift_1):
        temp0 = self._multiply_multiplier(lhs, np.int64(quan_multiplier_0), right_shift_0)
        temp1 = self._multiply_multiplier(rhs, np.int64(quan_multiplier_1), right_shift_1)
        temp = temp0 + temp1
        return temp

    def _multiply_multiplier(self, value, quan_multiplier, right_shift):
        return value * quan_multiplier // (2 ** right_shift)

    def save_model_file(self, file_path, platform='pc'):
        with open(file_path, 'w') as fout:
            fout.write(self._mcu_file() if platform == 'mcu' else self._pc_file())

    def _mcu_file(self):
        file_content = '''#include <stdio.h>
#define DIM_INPUT_1 {}
#define DIM_INPUT_2 {}
#define DIM_LAYER_1 {}
#define DIM_LAYER_2 {}
#define DIM_OUTPUT {}
#define int16_max 32767
#define int16_min -32768

namespace tiny_light {{
    // store message from PC
    char str_buffer[300];
    uint8_t str_length;

    // placeholder
    int16_t l1_lhs_1[DIM_INPUT_1];
    int16_t l1_lhs_2[DIM_INPUT_2];
    int16_t l1_sub_1[DIM_LAYER_1];
    int16_t l1_sub_2[DIM_LAYER_1];
    int16_t l1_sum[DIM_LAYER_1];
    int16_t l2_result[DIM_LAYER_2];
    int16_t l3_result[DIM_OUTPUT];

    // layer 1 sub 1
    const int32_t l1_quan_multiplier_w_1 = {}L;
    const uint8_t l1_quan_right_shift_w_1 = {};
    const int32_t l1_quan_multiplier_b_1 = {}L;
    const uint8_t l1_quan_right_shift_b_1 = {};
    const PROGMEM int16_t l1_rhs_1[DIM_INPUT_1 * DIM_LAYER_1] = {{
            {}
    }};
    const int16_t l1_bias_1[DIM_LAYER_1] = {{
            {}
    }};

    // layer 1 sub 2
    const int32_t l1_quan_multiplier_w_2 = {}L;
    const uint8_t l1_quan_right_shift_w_2 = {};
    const int32_t l1_quan_multiplier_b_2 = {}L;
    const uint8_t l1_quan_right_shift_b_2 = {};
    const PROGMEM int16_t l1_rhs_2[DIM_INPUT_2 * DIM_LAYER_1] = {{
            {}
    }};
    const int16_t l1_bias_2[DIM_LAYER_1] = {{
            {}
    }};

    // layer 1 sum
    const int32_t l1_quan_multiplier_sub1 = {}L;
    const uint8_t l1_quan_right_shift_sub1 = {};
    const int32_t l1_quan_multiplier_sub2 = {}L;
    const uint8_t l1_quan_right_shift_sub2 = {};

    // layer 2
    const int32_t l2_quan_multiplier_w = {}L;
    const uint8_t l2_quan_right_shift_w = {};
    const int32_t l2_quan_multiplier_b = {}L;
    const uint8_t l2_quan_right_shift_b = {};
    const PROGMEM int16_t l2_rhs[DIM_LAYER_1 * DIM_LAYER_2] = {{
            {}
    }};
    const int16_t l2_bias[DIM_LAYER_2] = {{
            {}
    }};

    // layer 3
    const int32_t l3_quan_multiplier_w = {}L;
    const uint8_t l3_quan_right_shift_w = {};
    const int32_t l3_quan_multiplier_b = {}L;
    const uint8_t l3_quan_right_shift_b = {};
    const PROGMEM int16_t l3_rhs[DIM_LAYER_2 * DIM_OUTPUT] = {{
            {}
    }};
    const int16_t l3_bias[DIM_OUTPUT] = {{
            {}
    }};
}}
'''.format(
            self.dim_input_0, self.dim_input_1, self.dim_layer_1, self.dim_layer_2, self.dim_output,
            self.quan_input_2_first_layer_w_0.multiplier, self.quan_input_2_first_layer_w_0.right_shift,
            self.quan_input_2_first_layer_b_0.multiplier, self.quan_input_2_first_layer_b_0.right_shift,
            str(self.quan_input_2_first_layer_w_0.quantize(self.input_2_first_layer_w_0.T).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],
            str(self.quan_input_2_first_layer_b_0.quantize(self.input_2_first_layer_b_0).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],

            self.quan_input_2_first_layer_w_1.multiplier, self.quan_input_2_first_layer_w_1.right_shift,
            self.quan_input_2_first_layer_b_1.multiplier, self.quan_input_2_first_layer_b_1.right_shift,
            str(self.quan_input_2_first_layer_w_1.quantize(self.input_2_first_layer_w_1.T).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],
            str(self.quan_input_2_first_layer_b_1.quantize(self.input_2_first_layer_b_1).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],

            self.quan_layer_1_sub0.multiplier, self.quan_layer_1_sub0.right_shift,
            self.quan_layer_1_sub1.multiplier, self.quan_layer_1_sub1.right_shift,

            self.quan_first_layer_2_second_layer_w.multiplier, self.quan_first_layer_2_second_layer_w.right_shift,
            self.quan_first_layer_2_second_layer_b.multiplier, self.quan_first_layer_2_second_layer_b.right_shift,
            str(self.quan_first_layer_2_second_layer_w.quantize(self.first_layer_2_second_layer_w.T).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],
            str(self.quan_first_layer_2_second_layer_b.quantize(self.first_layer_2_second_layer_b).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],

            self.quan_second_layer_2_output_w.multiplier, self.quan_second_layer_2_output_w.right_shift,
            self.quan_second_layer_2_output_b.multiplier, self.quan_second_layer_2_output_b.right_shift,
            str(self.quan_second_layer_2_output_w.quantize(self.second_layer_2_output_w.T).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],
            str(self.quan_second_layer_2_output_b.quantize(self.second_layer_2_output_b).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],
        )
        return file_content

    def _pc_file(self):
        return """#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdio.h>
#define DIM_INPUT_1 {}
#define DIM_INPUT_2 {}
#define DIM_LAYER_1 {}
#define DIM_LAYER_2 {}
#define DIM_OUTPUT {}
#define int16_max 32767
#define int16_min -32768

namespace tiny_light {{
    // store message from PC
    char* str_buffer;
    uint8_t str_length;

    // placeholder
    int16_t l1_lhs_1[DIM_INPUT_1];
    int16_t l1_lhs_2[DIM_INPUT_2];
    int16_t l1_sub_1[DIM_LAYER_1];
    int16_t l1_sub_2[DIM_LAYER_1];
    int16_t l1_sum[DIM_LAYER_1];
    int16_t l2_result[DIM_LAYER_2];
    int16_t l3_result[DIM_OUTPUT];

    // layer 1 sub 1
    const int32_t l1_quan_multiplier_w_1 = {}L;
    const uint8_t l1_quan_right_shift_w_1 = {};
    const int32_t l1_quan_multiplier_b_1 = {}L;
    const uint8_t l1_quan_right_shift_b_1 = {};
    const int16_t l1_rhs_1[DIM_INPUT_1 * DIM_LAYER_1] = {{
            {}
    }};
    const int16_t l1_bias_1[DIM_LAYER_1] = {{
            {}
    }};

    // layer 1 sub 2
    const int32_t l1_quan_multiplier_w_2 = {}L;
    const uint8_t l1_quan_right_shift_w_2 = {};
    const int32_t l1_quan_multiplier_b_2 = {}L;
    const uint8_t l1_quan_right_shift_b_2 = {};
    const int16_t l1_rhs_2[DIM_INPUT_2 * DIM_LAYER_1] = {{
            {}
    }};
    const int16_t l1_bias_2[DIM_LAYER_1] = {{
            {}
    }};

    // layer 1 sum
    const int32_t l1_quan_multiplier_sub1 = {}L;
    const uint8_t l1_quan_right_shift_sub1 = {};
    const int32_t l1_quan_multiplier_sub2 = {}L;
    const uint8_t l1_quan_right_shift_sub2 = {};

    // layer 2
    const int32_t l2_quan_multiplier_w = {}L;
    const uint8_t l2_quan_right_shift_w = {};
    const int32_t l2_quan_multiplier_b = {}L;
    const uint8_t l2_quan_right_shift_b = {};
    const int16_t l2_rhs[DIM_LAYER_1 * DIM_LAYER_2] = {{
            {}
    }};
    const int16_t l2_bias[DIM_LAYER_2] = {{
            {}
    }};

    // layer 3
    const int32_t l3_quan_multiplier_w = {}L;
    const uint8_t l3_quan_right_shift_w = {};
    const int32_t l3_quan_multiplier_b = {}L;
    const uint8_t l3_quan_right_shift_b = {};
    const int16_t l3_rhs[DIM_LAYER_2 * DIM_OUTPUT] = {{
            {}
    }};
    const int16_t l3_bias[DIM_OUTPUT] = {{
            {}
    }};
}}
""".format(
            self.dim_input_0, self.dim_input_1, self.dim_layer_1, self.dim_layer_2, self.dim_output,
            self.quan_input_2_first_layer_w_0.multiplier, self.quan_input_2_first_layer_w_0.right_shift,
            self.quan_input_2_first_layer_b_0.multiplier, self.quan_input_2_first_layer_b_0.right_shift,
            str(self.quan_input_2_first_layer_w_0.quantize(self.input_2_first_layer_w_0.T).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],
            str(self.quan_input_2_first_layer_b_0.quantize(self.input_2_first_layer_b_0).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],

            self.quan_input_2_first_layer_w_1.multiplier, self.quan_input_2_first_layer_w_1.right_shift,
            self.quan_input_2_first_layer_b_1.multiplier, self.quan_input_2_first_layer_b_1.right_shift,
            str(self.quan_input_2_first_layer_w_1.quantize(self.input_2_first_layer_w_1.T).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],
            str(self.quan_input_2_first_layer_b_1.quantize(self.input_2_first_layer_b_1).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],

            self.quan_layer_1_sub0.multiplier, self.quan_layer_1_sub0.right_shift,
            self.quan_layer_1_sub1.multiplier, self.quan_layer_1_sub1.right_shift,

            self.quan_first_layer_2_second_layer_w.multiplier, self.quan_first_layer_2_second_layer_w.right_shift,
            self.quan_first_layer_2_second_layer_b.multiplier, self.quan_first_layer_2_second_layer_b.right_shift,
            str(self.quan_first_layer_2_second_layer_w.quantize(self.first_layer_2_second_layer_w.T).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],
            str(self.quan_first_layer_2_second_layer_b.quantize(self.first_layer_2_second_layer_b).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],

            self.quan_second_layer_2_output_w.multiplier, self.quan_second_layer_2_output_w.right_shift,
            self.quan_second_layer_2_output_b.multiplier, self.quan_second_layer_2_output_b.right_shift,
            str(self.quan_second_layer_2_output_w.quantize(self.second_layer_2_output_w.T).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],
            str(self.quan_second_layer_2_output_b.quantize(self.second_layer_2_output_b).reshape([-1]).numpy().astype(np.int16).tolist())[1:-1],
        ) + """

void read_feature(const char *s, int16_t *feature_1, int16_t *feature_2, uint8_t dim_feature_1, uint8_t dim_feature_2) {
    memset(feature_1, 0, sizeof(int16_t) * dim_feature_1);
    memset(feature_2, 0, sizeof(int16_t) * dim_feature_2);
    int sign = 1;
    int16_t *feature = feature_1;

    for (uint8_t s_idx = 0; ; ++s_idx) {
        if (s[s_idx] == '-') {
            sign = -1;
        }
        else if (s[s_idx] >= '0' && s[s_idx] <= '9') {
            *feature = (*feature) * 10 + sign * (s[s_idx] - '0');
        }
        else if (s[s_idx] == ',') {
            ++feature;
            sign = 1;
        }
        else if (s[s_idx] == ';') {
            feature = feature_2;
            sign = 1;
        }
        else {
            break;
        }
    }
}

int64_t quan_multiply(int64_t val, int32_t quan_multiplier, uint8_t right_shift) {
    val = val * static_cast<int64_t>(quan_multiplier);
    return val >> right_shift;
}

void linear_relu(
        const int16_t *lhs,
        const int16_t *rhs,
        const int16_t *bias,
        int16_t *result,
        const int32_t quan_multiplier_w,
        const uint8_t right_shift_w,
        const int32_t quan_multiplier_b,
        const uint8_t right_shift_b,
        const uint16_t depth,
        const uint16_t col
) {
    for (uint16_t col_idx = 0; col_idx < col; ++col_idx) {
        uint16_t rhs_flatten_idx = col_idx;
        int64_t quan_matmul = 0;
        for (uint16_t depth_idx = 0; depth_idx < depth; ++depth_idx) {
            quan_matmul += static_cast<int64_t>(lhs[depth_idx]) * static_cast<int64_t>(static_cast<int16_t>(rhs[rhs_flatten_idx]));
            rhs_flatten_idx += col;
        }
        quan_matmul = quan_multiply(quan_matmul, quan_multiplier_w, right_shift_w);

        int64_t quan_bias = quan_multiply(static_cast<int64_t>(bias[col_idx]), quan_multiplier_b, right_shift_b);
        quan_matmul = quan_matmul + quan_bias;
        quan_matmul = quan_matmul > 0 ? quan_matmul : 0;
        quan_matmul = quan_matmul < int16_max ? quan_matmul : int16_max;
        result[col_idx] = static_cast<int16_t>(quan_matmul);
    }
}

void linear(
        const int16_t *lhs,
        const int16_t *rhs,
        const int16_t *bias,
        int16_t *result,
        const int32_t quan_multiplier_w,
        const uint8_t right_shift_w,
        const int32_t quan_multiplier_b,
        const uint8_t right_shift_b,
        const uint16_t depth,
        const uint16_t col
) {
    for (uint16_t col_idx = 0; col_idx < col; ++col_idx) {
        uint16_t rhs_flatten_idx = col_idx;
        int64_t quan_matmul = 0;
        for (uint16_t depth_idx = 0; depth_idx < depth; ++depth_idx) {
            quan_matmul += static_cast<int64_t>(lhs[depth_idx]) * static_cast<int64_t>(static_cast<int16_t>(rhs[rhs_flatten_idx]));
            rhs_flatten_idx += col;
        }
        quan_matmul = quan_multiply(quan_matmul, quan_multiplier_w, right_shift_w);

        int64_t quan_bias = quan_multiply(static_cast<int64_t>(bias[col_idx]), quan_multiplier_b, right_shift_b);
        quan_matmul = quan_matmul + quan_bias;
        quan_matmul = quan_matmul > int16_min ? quan_matmul : int16_min;
        quan_matmul = quan_matmul < int16_max ? quan_matmul : int16_max;
        result[col_idx] = static_cast<int16_t>(quan_matmul);
    }
}

void add(
        const int16_t *lhs,
        const int16_t *rhs,
        int16_t *result,
        const int32_t quan_multiplier_lhs,
        const uint8_t right_shift_lhs,
        const int32_t quan_multiplier_rhs,
        const uint8_t right_shift_rhs,
        const uint16_t col
) {
    for (uint16_t col_idx = 0; col_idx < col; ++col_idx) {
        int64_t res_lhs = quan_multiply(lhs[col_idx], quan_multiplier_lhs, right_shift_lhs);
        int64_t res_rhs = quan_multiply(rhs[col_idx], quan_multiplier_rhs, right_shift_rhs);
        int64_t res = res_lhs + res_rhs;
        res = res > int16_min ? res : int16_min;
        res = res < int16_max ? res : int16_max;
        result[col_idx] = static_cast<int16_t>(res);
    }
}

int _pick_action()
{
    read_feature(tiny_light::str_buffer, tiny_light::l1_lhs_1, tiny_light::l1_lhs_2, DIM_INPUT_1, DIM_INPUT_2);
    linear_relu(
            tiny_light::l1_lhs_1,
            tiny_light::l1_rhs_1,
            tiny_light::l1_bias_1,
            tiny_light::l1_sub_1,
            tiny_light::l1_quan_multiplier_w_1,
            tiny_light::l1_quan_right_shift_w_1,
            tiny_light::l1_quan_multiplier_b_1,
            tiny_light::l1_quan_right_shift_b_1,
            DIM_INPUT_1,
            DIM_LAYER_1
            );
    linear_relu(
            tiny_light::l1_lhs_2,
            tiny_light::l1_rhs_2,
            tiny_light::l1_bias_2,
            tiny_light::l1_sub_2,
            tiny_light::l1_quan_multiplier_w_2,
            tiny_light::l1_quan_right_shift_w_2,
            tiny_light::l1_quan_multiplier_b_2,
            tiny_light::l1_quan_right_shift_b_2,
            DIM_INPUT_2,
            DIM_LAYER_1
            );
    add(
            tiny_light::l1_sub_1,
            tiny_light::l1_sub_2,
            tiny_light::l1_sum,
            tiny_light::l1_quan_multiplier_sub1,
            tiny_light::l1_quan_right_shift_sub1,
            tiny_light::l1_quan_multiplier_sub2,
            tiny_light::l1_quan_right_shift_sub2,
            DIM_LAYER_1
            );
    linear_relu(
            tiny_light::l1_sum,
            tiny_light::l2_rhs,
            tiny_light::l2_bias,
            tiny_light::l2_result,
            tiny_light::l2_quan_multiplier_w,
            tiny_light::l2_quan_right_shift_w,
            tiny_light::l2_quan_multiplier_b,
            tiny_light::l2_quan_right_shift_b,
            DIM_LAYER_1,
            DIM_LAYER_2
            );
    linear(
            tiny_light::l2_result,
            tiny_light::l3_rhs,
            tiny_light::l3_bias,
            tiny_light::l3_result,
            tiny_light::l3_quan_multiplier_w,
            tiny_light::l3_quan_right_shift_w,
            tiny_light::l3_quan_multiplier_b,
            tiny_light::l3_quan_right_shift_b,
            DIM_LAYER_2,
            DIM_OUTPUT
            );

    int action = 0;
    for (int idx = 1; idx < DIM_OUTPUT; ++idx) {
        if (tiny_light::l3_result[idx] > tiny_light::l3_result[action]) {
            action = idx;
        }
    }
    return action;
}

static PyObject *
quan_feature_0(PyObject *self, PyObject *args) {
    float f;
    int16_t i;
    if (!PyArg_ParseTuple(args, "f", &f))
        return NULL;
    i = static_cast<int32_t>(round(f / """ + str(self.quan_input_0.scale.item()) + """));
    i = i > int16_max ? int16_max : i;
    i = i < int16_min ? int16_min : i;
    return Py_BuildValue("i", i);
}

static PyObject *
quan_feature_1(PyObject *self, PyObject *args) {
    float f;
    int16_t i;
    if (!PyArg_ParseTuple(args, "f", &f))
        return NULL;
    i = static_cast<int32_t>(round(f / """ + str(self.quan_input_1.scale.item()) + """));
    i = i > int16_max ? int16_max : i;
    i = i < int16_min ? int16_min : i;
    return Py_BuildValue("i", i);
}

static PyObject *
pick_action(PyObject *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "s", &tiny_light::str_buffer))
        return NULL;
    return Py_BuildValue("i", _pick_action());
}

static PyMethodDef Methods[] = {
    {"pick_action", pick_action, METH_VARARGS, "pick action"},
    {"quan_feature_0", quan_feature_0, METH_VARARGS, "quantize feature 0"},
    {"quan_feature_1", quan_feature_1, METH_VARARGS, "quantize feature 1"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "mcu_agent",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC
PyInit_flow_""" + str(self.config["flow_idx"]) +  """_agent_""" + str(self.idx) + """(void)
{
    return PyModule_Create(&module);
}
"""
