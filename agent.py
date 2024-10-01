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

# Action Space = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

class Agent:
    def __init__(self, input_shape, n_actions, hyperparameter_set, training=False):
        # Get hyperparameters from yaml file in current directory
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparam_sets = yaml.safe_load(file)
            hyperparam = all_hyperparam_sets[hyperparameter_set]

        self.hidden_layer_num = hyperparam['hidden_layers']        # Number of hidden layers to use for linear nn layers
        self.replay_memory_size = hyperparam['replay_memory_size'] # Size of replay memory
        self.batch_size = hyperparam['batch_size']                 # Size of training data set to be sampled from replay memory
        self.epsilon = hyperparam['epsilon_init']                  # 1 - 100% random actions
        self.epsilon_decay = hyperparam['epsilon_decay']           # epsilon decay rate
        self.epsilon_min = hyperparam['epsilon_min']               # minimum epsilon value
        self.network_sync_rate = hyperparam['network_sync_rate']   # Target step count to sync the policy and target nets
        self.learning_rate = hyperparam['learning_rate']           # Learning rate for training
        self.discount_factor = hyperparam['discount_factor']       # Discount factor for DQN algorithm
        self.epoch = hyperparam['epoch']                           # Amount of games to train for

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.image_crop = input_shape[1]
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
        self.policy_net = DQN(self.input_shape, self.n_actions, self.hidden_layer_num).to(device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        if training:
            self.replay_memory = ReplayMemory(self.replay_memory_size)

            self.target_net = DQN(input_shape=self.input_shape, actions_dim=self.n_actions).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def optimize(self, batch, policy_net, target_net):
        for curr_state, act, new_state, reward, terminated in batch:
            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.discount_factor * target_net(new_state).max()

            current_q = policy_net(curr_state)

            loss = self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def act(self, curr_state, action_space):
        observation = torch.from_numpy(curr_state).unsqueeze(0).to(device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(observation)
        self.policy_net.train()

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return self.actions[int(np.unravel_index(torch.argmax(action_values), action_values.shape)[1])]
        else:
            return action_space.sample()

if __name__ == "__main__":
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set path for custom ROMs
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, 'custom_integrations')
    )

    env = retro.make("Tetris-Nes", inttype=retro.data.Integrations.ALL)
    num_actions = env.action_space.n

    # TODO
    # Rename this variable
    INPUT_SHAPE = (35, 204, 85, 170)

    agent = Agent((1, INPUT_SHAPE, INPUT_SHAPE), num_actions,"tetris", training=True)
    is_training = True
    rewards_per_episode = []
    epsilon_history = []

    if is_training:
        step_count = 0

    for episode in range(agent.epoch):
        obs = env.reset()

        done = False
        episode_reward = 0.0

        state = preprocess(obs, agent.image_crop, agent.image_resize)

        while not done:
            env.render()
            action = agent.act(state, env.action_space)
            next_obs, rew, done, info = env.step(action)
            episode_reward += rew
            state = preprocess(next_obs, agent.image_crop, agent.image_resize)

            print(episode_reward)

            if is_training:
                agent.replay_memory.append((obs, action, next_obs, rew, done))

                step_count += 1

        rewards_per_episode.append(episode_reward)

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        epsilon_history.append(agent.epsilon)

        if (len(agent.replay_memory)) > agent.batch_size:
            batch = agent.replay_memory.sample(agent.batch_size)

            agent.optimize(batch, agent.policy_net, agent.target_net)

            if step_count > agent.network_sync_rate:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                step_count = 0

    env.close()