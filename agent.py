# TODO
# Implement DQN to Gym Environment
import random

import matplotlib
import torch
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from collections import namedtuple

from dqn import DQN
from exp_replay import ReplayMemory

# Action Space = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set path for logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class Agent:
    def __init__(self, input_shape, n_actions, hyperparameter_set, training=False):
        # Get hyperparameters from yaml file in current directory
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparam_sets = yaml.safe_load(file)
            hyperparam = all_hyperparam_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.hidden_layer_num = hyperparam['hidden_layers']        # Number of hidden layers to use for linear nn layers
        self.replay_memory_size = hyperparam['replay_memory_size'] # Size of replay memory
        self.batch_size = hyperparam['batch_size']                 # Size of training data set to be sampled from replay memory
        self.epsilon = hyperparam['epsilon_init']                  # 1 - 100% random actions
        self.epsilon_decay = hyperparam['epsilon_decay']           # epsilon decay rate
        self.epsilon_min = hyperparam['epsilon_min']               # minimum epsilon value
        self.network_sync_rate = hyperparam['network_sync_rate']   # Target step count to sync the policy and target nets
        self.learning_rate = hyperparam['learning_rate']           # Learning rate for training
        self.gamma = hyperparam['gamma']                           # Discount factor gamma for DQN algorithm
        self.epoch = hyperparam['epoch']                           # Amount of games to train for

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.image_resize = input_shape[1]
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

        self.LOG_FILE = os.path.join(LOG_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(LOG_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(LOG_DIR, f'{self.hyperparameter_set}.png')


        self.replay_memory = ReplayMemory(self.replay_memory_size, self.batch_size, device)
        self.target_net = DQN(self.input_shape, self.n_actions, self.hidden_layer_num).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

        self.TAU = 0.005

    # Calculate the Q targets for the current states and run the selected optimizer
    def optimize(self):
        transitions = self.replay_memory.sample()
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)

        next_states = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_states[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_states * self.gamma) + rewards

        # # Expected Q values using the policy network
        current_q = self.policy_net(states).gather(1, actions)

        # with torch.no_grad():
        #     next_state_q = self.target_net(next_states).max(1)[0]
        #
        # # Compute Q targets
        # target_q = rewards + (self.gamma * next_state_q)

        # Compute loss
        loss = self.loss_fn(current_q, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.soft_update()

    # Run the input through the policy network
    def act(self, curr_state):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(curr_state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.choice(np.arange(self.n_actions))]], device=device, dtype=torch.long)

    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)