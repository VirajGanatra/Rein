class Agent:
    def __init__(self, policy):
        self.policy = policy

    def act(self, state):
        return self.policy.select_action(state)

    def update(self, transition):
        raise NotImplementedError