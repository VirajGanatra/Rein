import gymnasium as gym

class Environment:
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

class CartPoleEnv(Environment):
    def __init__(self):
        self.env = gym.make('CartPole-v1')

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


    def render(self):
        self.env.render()