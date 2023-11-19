from .. import BaseAgent
import math


class MaxPressure(BaseAgent):
    """
    Varaiya, P.: The max-pressure controller for arbitrary networks of signalized intersections.
        In: Advances in Dynamic Network Modeling in Complex Transportation Systems, pp. 27–66. Springer (2013)
    """
    def __init__(self, config, env, idx):
        super(MaxPressure, self).__init__(config, env, idx)
        self.t_min = 20  # the minimum duration of one phase

    def reset(self):
        pass

    def pick_action(self, n_obs, on_training):
        obs = n_obs[self.idx]
        if self.inter.current_phase_time < self.t_min:
            return self.current_phase

        max_pressure = -math.inf
        for phase_id in range(self.num_phase):
            pressure = self._get_pressure_for_phase(obs, phase_id)
            if pressure > max_pressure:
                max_pressure = pressure
                self.current_phase = phase_id
        return self.current_phase

    def _get_pressure_for_phase(self, obs, phase_id):
        pressure = 0
        n_available_lane_link = self.inter.n_phase[phase_id].n_available_lanelink_id
        for lane_link in n_available_lane_link:
            start_lane_idx, end_lane_idx = lane_link[0], lane_link[1]
            pressure += obs[0][0, self.inter.n_lane_id.index(start_lane_idx)].item()
            pressure -= obs[0][0, self.inter.n_lane_id.index(end_lane_idx)].item()
        return pressure
