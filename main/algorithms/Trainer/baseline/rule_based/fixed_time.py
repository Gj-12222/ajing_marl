from .. import BaseAgent


class FixedTime(BaseAgent):
    """
    Alan J Miller. 1963. Settings for fixed-cycle traffic signals. Journal of the Operational Research Society 14,
        4 (1963), 373–386.
    """
    def __init__(self, config, env, idx):
        super(FixedTime, self).__init__(config, env, idx)
        self.fixed_time_interval = self.cur_agent['fixed_time_interval']
        self.current_phase = -1

    def reset(self):
        self.current_phase = -1  # will be reset to 0 when pick_action is called for the first time

    def pick_action(self, n_obs, on_training):
        if self.config['current_episode_step_idx'] % self.fixed_time_interval == 0:
            self.current_phase = (self.current_phase + 1) % self.num_phase
        return self.current_phase
