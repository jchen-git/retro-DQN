import random
import torch
import yaml
import os
import numpy as np
from dqn import DQN
from exp_replay import ReplayMemory

# Action Space = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set path for logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class Agent:
    def __init__(self, input_shape, hyperparameter_set, training=False):
        # Get hyperparameters from yaml file in current directory
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparam_sets = yaml.safe_load(file)
            hyperparam = all_hyperparam_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.hidden_layer_num = hyperparam['hidden_layers']        # Number of hidden layers to use for linear nn layers
        self.replay_memory_size = hyperparam['replay_memory_size'] # Size of replay memory
        self.mini_batch_size = hyperparam['mini_batch_size']                 # Size of training data set to be sampled from replay memory
        self.epsilon = hyperparam['epsilon_init']                  # 1 - 100% random actions
        self.epsilon_decay = hyperparam['epsilon_decay']           # epsilon decay rate
        self.epsilon_min = hyperparam['epsilon_min']               # minimum epsilon value
        self.update_rate = hyperparam['update_rate']               # Target step count to run the optimize function
        self.learning_rate = hyperparam['learning_rate']           # Learning rate for training
        self.GAMMA = hyperparam['GAMMA']                           # Discount factor gamma for DQN algorithm
        self.epoch = hyperparam['epoch']                           # Amount of games to train for
        self.TAU = hyperparam['TAU']
        self.FRAME_SKIPS = hyperparam['frame_skips']               # Amount of frames to skip during training

        self.input_shape = input_shape
        self.actions = {
            # B
            0: [1, 0, 0, 0, 0, 0, 0, 0, 0],
            # No Operation
            1: [0, 1, 0, 0, 0, 0, 0, 0, 0],
            # LEFT
            2: [0, 0, 0, 0, 0, 0, 1, 0, 0],
            # RIGHT
            3: [0, 0, 0, 0, 0, 0, 0, 1, 0]
        }
        self.policy_net = DQN(self.input_shape, len(self.actions), self.hidden_layer_num).to(device)

        self.loss_fn = torch.nn.MSELoss()

        self.replay_memory = ReplayMemory(self.replay_memory_size, self.mini_batch_size, device)
        self.target_net = DQN(self.input_shape, len(self.actions), self.hidden_layer_num).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

        self.LOG_FILE = os.path.join(LOG_DIR, f'{self.hyperparameter_set}.log')
        self.DATA_FILE = os.path.join(LOG_DIR, f'{self.hyperparameter_set}.dat')
        self.MODEL_FILE = os.path.join(LOG_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = LOG_DIR

    # Calculate the Q targets for the current states and run the selected optimizer
    def optimize(self):
        transitions = self.replay_memory.sample()
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, device=device, dtype=torch.float)

        ratings = self.target_net(next_states).max(dim=1)[0]

        # Compute the expected Q values
        expected_q = (1 - dones) * ratings * self.GAMMA + rewards

        # Expected Q values using the policy network
        current_q = self.policy_net(states).squeeze(1)

        # Compute loss
        loss = self.loss_fn(current_q, expected_q)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()

    def act(self, states):
        if random.random() > self.epsilon:
            max_rating = None
            best_state = None
            with torch.no_grad():
                ratings = self.policy_net(torch.tensor([state for i, (action, state) in enumerate(states)], device=device, dtype=torch.float))

                for i, (action, state) in enumerate(states):
                    rating = ratings[i]
                    if not max_rating or rating > max_rating:
                        max_rating = rating
                        best_state = (action, state)
            return best_state
        else:
            return random.choice(states)

    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)