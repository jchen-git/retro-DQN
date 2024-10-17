import random
import torch
import yaml
import os
import numpy as np
from dqnCNN import DQN
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
        self.image_resize = input_shape[1]
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
        self.GRAPH_FILE = os.path.join(LOG_DIR, f'{self.hyperparameter_set}.png')
        self.GRAPH_SCORE_FILE = os.path.join(LOG_DIR, f'{self.hyperparameter_set}_score.png')

    # Calculate the Q targets for the current states and run the selected optimizer
    def optimize(self):
        transitions = self.replay_memory.sample()
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, device=device, dtype=torch.float)

        # Compute the expected Q values
        expected_q = (1 - dones) * self.target_net(next_states).max(1).values * self.GAMMA + rewards

        # Expected Q values using the policy network
        current_q = self.policy_net(states).gather(1, actions)

        # Compute loss
        loss = self.loss_fn(current_q, expected_q.unsqueeze(1))

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.soft_update()

    # Run the input through the policy network
    def act(self, curr_state):
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(curr_state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.choice(np.arange(len(self.actions)))]], device=device, dtype=torch.long)

    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)