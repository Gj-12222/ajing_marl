class AgentTrainer(object):
    def __init__(self, name, obs_shape_n, act_space_n, agent_index, args, local_q_func):
        raise NotImplemented()

    def action(self, obs):
        raise NotImplemented()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def update(self, agents, t):
        raise NotImplemented()