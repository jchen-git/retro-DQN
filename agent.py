# TODO
# Implement DQN to Gym Environment
import random
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from dqn import DQN
from exp_replay import ReplayMemory

# Action Space = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            self.replay_memory = ReplayMemory(self.replay_memory_size, self.batch_size, device)

            self.target_net = DQN(self.input_shape, self.n_actions, self.hidden_layer_num).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    # Calculate the Q targets for the current states and run the selected optimizer
    def optimize(self, mini_batch):
        states, actions, rewards, next_states, dones = mini_batch

        # Expected Q values using the policy network
        q_expected_current = self.policy_net(states)
        q_expected = q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Max predicted Q values for future states using the target network
        q_targets_next = self.target_net(next_states).detach().max(1)[0]

        # Compute Q targets
        q_targets = rewards + (self.discount_factor * q_targets_next * (1 - dones))

        # Compute loss
        loss = self.loss_fn(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Run the input through the policy network
    def act(self, curr_state):
        observation = torch.from_numpy(curr_state).unsqueeze(0).to(device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(observation)
        self.policy_net.train()

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return np.unravel_index(torch.argmax(action_values), action_values.shape)[1]
        else:
            return random.choice(np.arange(self.n_actions))