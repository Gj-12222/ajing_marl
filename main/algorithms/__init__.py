class AgentTrainer(object):
    def action(self, obs):
        raise NotImplemented()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def update(self, agents, t):
        raise NotImplemented()