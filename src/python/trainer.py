import numpy as np
from collections import deque
import plotly.graph_objects as go

class Trainer:
    def __init__(self, env, agent, algorithm, episodes=1000, max_steps=1000):
        self.env = env
        self.agent = agent
        self.algorithm = algorithm
        self.episodes = episodes
        self.max_steps = max_steps
        self.rewards_history = deque(maxlen=100)

    def train(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            episode_reward = 0
            states, actions, rewards = [], [], []

            for step in range(self.max_steps):
                action, log_prob = self.agent.policy.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                episode_reward += reward

                if done:
                    break

            self.rewards_history.append(episode_reward)
            loss = self.algorithm.update(states, actions, rewards)

            if episode % 10 == 0:
                avg_reward = np.mean(self.rewards_history)
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")

            if episode % 100 == 0:
                self.plot_rewards()

    def plot_rewards(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=list(self.rewards_history), mode='lines', name='Average Reward'))
        fig.update_layout(title='Average Reward over Last 100 Episodes',
                          xaxis_title='Episode',
                          yaxis_title='Average Reward')
        fig.write_image(f"rewards_plot_episode_{len(self.rewards_history)}.png")