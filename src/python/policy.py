import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy:
    def select_action(self, state):
        raise NotImplementedError

class RandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        return np.random.choice(self.action_space)

class NNPolicy(nn.Module, Policy):
    def __init__(self, state_dim, action_dim):
        super(NNPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


    def select_action(self, state):
        state_array = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state_array).float().unsqueeze(0)
        probs = self.forward(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def log_probs(self, states, actions):
        states_tensor = states.clone().detach().float()
        probs = self.forward(states_tensor)
        dist = Categorical(probs)
        return dist.log_prob(actions.clone().detach())