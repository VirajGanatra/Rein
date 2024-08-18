import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import replay_buffer # TODO: Implement replay_buffer.cpp

class DQN(nn.Module):
    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=1e-3, capacity=10000, epsilon=0.1,
                 epsilon_decay=0.99, epsilon_min=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.memory = replay_buffer.ReplayBuffer(capacity)

        # These govern the chance of selecting a random action - it decays over time as the agent learns
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=learning_rate)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict()) # Copy the weights from the Q network to the target network

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute Q(s_t, a) - the model computes Q(s_t), select columns of actions taken
        current_q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            expected_q_values = rewards + ((1 - dones) * self.gamma * next_q_values)

        loss = nn.SmoothL1Loss()(current_q_values, expected_q_values.unsqueeze(1))
        return loss

    def update(self, batch_size):
        if self.memory.size() < batch_size:
            return

        batch = self.memory.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

        return loss.item()


