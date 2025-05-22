from .. import BaseAgent
import torch


class SOTL(BaseAgent):
    """
    Seung-Bae Cools, Carlos Gershenson, and Bart D’Hooghe. 2013. Self-organizing traffic lights: A realistic simulation.
        In Advances in applied self-organizing systems. Springer, 45–55.
    """
    def __init__(self, config, env, idx):
        super(SOTL, self).__init__(config, env, idx)

        # the minimum duration of time of one phase
        self.t_min = 10
        # some thresholds to deal with phase requests
        self.min_green_vehicle = 20
        self.max_red_vehicle = 0
        # phase 2 passable lane
        self.phase_2_passable_lane = torch.tensor(self.inter.phase_2_passable_lane_idx)

    def reset(self):
        pass

    def pick_action(self, n_obs, on_training):
        obs = n_obs[self.idx]
        assert obs[0].shape[0] == 1, 'should be mini-batch with size 1'

        action = self.inter.current_phase
        if self.inter.current_phase_time >= self.t_min:
            # num of waiting vehicles on lanes w.r.t. current phase
            num_green_vehicle = torch.sum(obs[0] * self.phase_2_passable_lane[self.current_phase:(self.current_phase+1), :])
            # num of waiting vehicles on other lanes
            num_red_vehicle = torch.sum(obs[0] * (1 - self.phase_2_passable_lane[self.current_phase:(self.current_phase+1), :]))

            if num_green_vehicle <= self.min_green_vehicle and num_red_vehicle > self.max_red_vehicle:
                action = (action + 1) % self.num_phase

        self.current_phase = action
        return self.current_phase
