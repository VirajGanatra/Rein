import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class REINFORCE(nn.Module):



    def __init__(self, policy, learning_rate=1e-2, gamma=0.99):
        super(REINFORCE, self).__init__()
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma


    def update(self, states, actions, rewards):
        returns = self._compute_returns(rewards)
        loss = self._compute_loss(states, actions, returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _compute_returns(self, rewards):
        returns = deque()
        R = 0 #return of state (after discounting)
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + math.ulp(0.0))
        return returns

    def _compute_loss(self, states, actions, returns):
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions)
        log_probs = self.policy.log_probs(states, actions)
        return -(log_probs * returns).sum()
