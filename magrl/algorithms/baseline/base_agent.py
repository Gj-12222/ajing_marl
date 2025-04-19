from Env.Agent.agent import TSCAgent
from Env.CityFlowEnv import CityFlowWorld


class BaseAgent:
    def __init__(self, config, algo_config, idx, agents: list):
        #  config, algo_config, agent_index, agents):
        self.config = config
        self.algo_cfg = algo_config
        self.env = agents[idx].env  # type: CityFlowWorld
        self.idx = idx
        self.cur_agent = self.config['algo_config']  # type: dict

        self.inter = agents[idx]  # type: TSCAgent
        self.action_space = self.env.n_action_space[idx]
        self.obs_shape = agents[idx].obs_dim
        self.num_phase = self.action_space.n
        self.current_phase = 0
        self.device = config['device']

    def reset(self):
        raise NotImplementedError

    def action(self, n_obs):
        raise NotImplementedError

    def update(self, agents, agent_index=None, train_step=None):
        pass

    def save_replay_buffer(self, obs, action, reward, next_obs, done, terminal):
        pass
