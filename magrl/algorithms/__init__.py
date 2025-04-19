class AgentTrainer(object):
    def action(self, obs):
        raise NotImplemented()

    def update(self, trainers, train_step, agent_index=None):
        raise NotImplemented()


