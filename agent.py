# TODO
# Implement DQN to Gym Environment
import random
import retro
import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from dqn import DQN
from exp_replay import ReplayMemory
from preprocessing import preprocess

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Action Space = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
# env.action_space.sample()

class Agent:
    def __init__(self, hyperparameter_set):
        retro.data.Integrations.add_custom_path(
            os.path.join(SCRIPT_DIR, 'custom_integrations')
        )
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparam_sets = yaml.safe_load(file)
            hyperparam = all_hyperparam_sets[hyperparameter_set]

        self.replay_memory_size = hyperparam['replay_memory_size']
        self.batch_size = hyperparam['batch_size']
        self.epsilon_init = hyperparam['epsilon_init']
        self.epsilon_decay = hyperparam['epsilon_decay']
        self.epsilon_min = hyperparam['epsilon_min']

    def run(self, is_training=True, render=False):
        env = retro.make("Tetris-Nes", inttype=retro.data.Integrations.ALL)

        #num_obs = env.observation_space.shape
        num_actions = env.action_space.n

        rewards_per_episode = []
        epsilon_history = []

        policy_net = DQN(input_shape=(1, 84, 84), actions_dim=num_actions).to(device)

        if is_training:
            replay_memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

        for episode in range(1000):
            obs = env.reset()

            done = False
            episode_reward = 0.0

            state = preprocess(obs, (1, -1, -1, 1), 84)

            while not done:
                env.render()
                state = torch.from_numpy(state).unsqueeze(0).to(device)
                policy_net.eval()
                with torch.no_grad():
                    action_values = policy_net(state)
                policy_net.train()

                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = np.argmax(action_values.cpu().data.numpy())
                    action = [action==0, action==1, action==2, action==3, action==4, action==5, action==6, action==7, action==8]
                else:
                    action = env.action_space.sample()

                next_obs, rew, done, info = env.step(action)

                episode_reward += rew

                state = preprocess(next_obs, (1, -1, -1, 1), 84)
                rew = torch.tensor(rew, dtype=torch.float, device=device)

                if is_training:
                    replay_memory.append((obs, action, next_obs, rew, done))

            rewards_per_episode.append(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

        env.close()

if __name__ == "__main__":
    agent = Agent("tetris")
    agent.run(is_training=True, render=True)