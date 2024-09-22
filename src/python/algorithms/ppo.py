import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay_buffer import ReplayBuffer
import ppo_core

class PPO(nn.Module):
    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=1e-3, capacity=10000, epsilon=0.1,
                 epsilon_decay=0.99, epsilon_min=0.01, clip_param=0.2, value_coeff=0.5, entropy_coeff=0.01,
                 max_grad_norm=0.5, gae_lambda=0.95, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.memory = ReplayBuffer(capacity)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.clip_param = clip_param
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda

        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.optimizer = optim.AdamW(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate)

    def _build_actor(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
            nn.Softmax(dim=-1)
        )
        return model

    def _build_critic(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        return model

    def compute_advantages(self, states, next_states, rewards, dones):
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        rewards = rewards.detach().numpy()
        dones = dones.detach().numpy()

        advantages = ppo_core.compute_advantages(values.numpy(), next_values.numpy(), rewards, dones, self.gamma, self.gae_lambda)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def compute_loss(self, states, actions, old_action_log_probs, advantages, returns):
        action_probs = self.actor(states)
        action_log_probs = torch.log(action_probs.gather(1, actions))

        actor_loss = ppo_core.compute_actor_loss(action_log_probs, old_action_log_probs, advantages, self.clip_param)
        critic_loss = ppo_core.compute_critic_loss(self.critic(states), returns)
        entropy_loss = ppo_core.compute_entropy_loss(action_probs)

        total_loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy_loss
        return total_loss, actor_loss, critic_loss, entropy_loss

    def update(self, batch_size, num_epochs=10):
        if self.memory.size() < batch_size:
            return

        batch = self.memory.random_batch(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        advantages, returns = self.compute_advantages(states, next_states, rewards, dones)

        with torch.no_grad():
            old_action_probs = self.actor(states)
            old_action_log_probs = torch.log(old_action_probs.gather(1, actions))

        for _ in range(num_epochs):
            total_loss, actor_loss, critic_loss, entropy_loss = self.compute_loss(
                states, actions, old_action_log_probs, advantages, returns
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return total_loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)

        if np.random.rand() < self.epsilon:
            action = torch.randint(0, self.action_size, (1,)).item()
        else:
            action = torch.multinomial(action_probs, 1).item()

        return action