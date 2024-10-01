# TODO
# Implement DQN to Gym Environment
import random
import retro
import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
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

        self.replay_memory_size = hyperparam['replay_memory_size'] # Size of replay memory
        self.batch_size = hyperparam['batch_size']                 # Size of training data set to be sampled from replay memory
        self.epsilon_init = hyperparam['epsilon_init']             # 1 - 100% random actions
        self.epsilon_decay = hyperparam['epsilon_decay']           # epsilon decay rate
        self.epsilon_min = hyperparam['epsilon_min']               # minimum epsilon value
        self.network_sync_rate = hyperparam['network_sync_rate']   # Target step count to sync the policy and target nets
        self.learning_rate = hyperparam['learning_rate']
        self.discount_factor = hyperparam['discount_factor']
        self.epoch = hyperparam['epoch']

        self.image_crop = (35, 204, 85, 170)
        self.image_resize = 64
        self.actions = {
            # B
            0: [1, 0, 0, 0, 0, 0, 0, 0, 0],
            # No Operation
            1: [0, 1, 0, 0, 0, 0, 0, 0, 0],
            # SELECT
            2: [0, 0, 1, 0, 0, 0, 0, 0, 0],
            # START
            3: [0, 0, 0, 1, 0, 0, 0, 0, 0],
            # UP
            4: [0, 0, 0, 0, 1, 0, 0, 0, 0],
            # DOWN
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0],
            # LEFT
            6: [0, 0, 0, 0, 0, 0, 1, 0, 0],
            # RIGHT
            7: [0, 0, 0, 0, 0, 0, 0, 1, 0],
            # A
            8: [0, 0, 0, 0, 0, 0, 0, 0, 1]
        }

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def optimize(self, batch, policy_net, target_net):
        for state, action, new_state, reward, terminated in batch:
            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.discount_factor * target_net(new_state).max()

            current_q = policy_net(state)

            loss = self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def run(self, is_training=True, render=False):
        env = retro.make("Tetris-Nes", inttype=retro.data.Integrations.ALL)

        #num_obs = env.observation_space.shape
        num_actions = env.action_space.n

        rewards_per_episode = []
        epsilon_history = []

        policy_net = DQN(input_shape=(1, self.image_crop, self.image_crop), actions_dim=num_actions).to(device)

        if is_training:
            replay_memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

            target_net = DQN(input_shape=(1, self.image_crop, self.image_crop), actions_dim=num_actions).to(device)
            target_net.load_state_dict(policy_net.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.learning_rate)

        for episode in range(self.epoch):
            obs = env.reset()

            done = False
            episode_reward = 0.0

            state = preprocess(obs, self.image_crop, self.image_resize)

            while not done:
                env.render()

                state = torch.from_numpy(state).unsqueeze(0).to(device)
                policy_net.eval()
                with torch.no_grad():
                    action_values = policy_net(state)
                policy_net.train()

                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = np.unravel_index(torch.argmax(action_values), action_values.shape)[1]
                    action = self.actions[action]
                else:
                    action = env.action_space.sample()

                next_obs, rew, done, info = env.step(action)

                episode_reward += rew

                state = preprocess(next_obs, self.image_crop, self.image_resize)

                print(episode_reward)

                if is_training:
                    replay_memory.append((obs, action, next_obs, rew, done))

                    step_count += 1

            rewards_per_episode.append(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            if (len(replay_memory)) > self.batch_size:
                batch = replay_memory.sample(self.batch_size)

                self.optimize(batch, policy_net, target_net)

                if step_count > self.network_sync_rate:
                    target_net.load_state_dict(policy_net.state_dict())
                    step_count = 0

        env.close()

if __name__ == "__main__":
    agent = Agent("tetris")
    agent.run(is_training=True, render=True)