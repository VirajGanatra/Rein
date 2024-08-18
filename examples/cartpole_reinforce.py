from src.python.algorithms.reinforce import REINFORCE
from src.python.environment import CartPoleEnv
from src.python.policy import NNPolicy
from src.python.trainer import Trainer
from src.python.agent import Agent


def main():
    print("Initializing environment...")
    env = CartPoleEnv()
    print("Initializing policy...")
    policy = NNPolicy(4, 2)
    print("Initializing agent...")
    agent = Agent(policy)
    print("Initializing REINFORCE algorithm...")
    reinforce = REINFORCE(policy)
    print("Initializing trainer...")
    trainer = Trainer(env, agent, reinforce, episodes=1000)

    print("Starting training...")
    trainer.train()
    print("Training completed.")

if __name__ == '__main__':
    main()
