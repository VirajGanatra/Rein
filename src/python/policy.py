import numpy as np

class Policy:
    def select_action(self, state):
        raise NotImplementedError

class RandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        return np.random.choice(self.action_space)