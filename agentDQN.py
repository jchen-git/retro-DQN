import random
import torch
import yaml
import os
from dqn import DQN
from exp_replay import ReplayMemory

# Action Space for gym retro emulation
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set path for logging
LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class Agent:
    def __init__(self, input_shape, hyperparameter_set, is_training=False):
        # Get hyperparameters from yaml file in current directory
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparam_sets = yaml.safe_load(file)
            hyperparam = all_hyperparam_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.hidden_layer_num = hyperparam['hidden_layers']  # Number of hidden layers to use for linear nn layers
        self.replay_memory_size = hyperparam['replay_memory_size'] # Size of replay memory
        self.mini_batch_size = hyperparam['mini_batch_size']       # Size of training data set to be sampled from replay memory
        self.learning_rate = hyperparam['learning_rate']           # Learning rate for training
        self.GAMMA = hyperparam['GAMMA']                           # Discount factor gamma for DQN algorithm
        self.epoch = hyperparam['epoch']                           # Amount of games to train for
        self.TAU = hyperparam['TAU']                               # TAU value for soft updating the networks
        self.bumpiness_weight = hyperparam['bumpiness_weight']
        self.agg_height_weight = hyperparam['agg_height_weight']
        self.hole_weight = hyperparam['hole_weight']
        self.line_clear_weight = hyperparam['line_clear_weight']
        self.model_dir = hyperparam['model_dir']
        self.log_dir = hyperparam['log_dir']

        if is_training:
            self.epsilon = hyperparam['epsilon_init']                  # 1 - 100% random actions
            self.epsilon_decay = hyperparam['epsilon_decay']           # epsilon decay rate
            self.epsilon_min = hyperparam['epsilon_min']               # minimum epsilon value
            self.update_rate = hyperparam['update_rate']               # Target step count to run the optimize function
        else:
            self.epsilon = 0
            self.epsilon_min = 0

        self.input_shape = input_shape
        self.policy_net = DQN(self.input_shape, self.hidden_layer_num).to(device)
        self.replay_memory = ReplayMemory(self.replay_memory_size, self.mini_batch_size, device)
        self.target_net = DQN(self.input_shape, self.hidden_layer_num).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

        self.LOG_FILE = os.path.join(self.log_dir, f'{self.hyperparameter_set}.log')
        self.DATA_FILE = os.path.join(self.log_dir, f'{self.hyperparameter_set}.dat')
        self.MODEL_FILE = os.path.join(self.model_dir, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = self.log_dir

    # Calculate the Q targets for the current states and run the selected optimizer
    # Input:  None
    # Output: None
    def optimize(self):
        # Randomly pick out a mini batch from agent's replay memory
        transitions = self.replay_memory.sample()
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.cat(states)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, device=device, dtype=torch.float)

        # Compute the target Q values of the next state
        target_q = (1 - dones) * self.target_net(next_states).max(dim=1)[0] * self.GAMMA + rewards

        # Compute the Q values of the previous state using the policy network
        current_q = self.policy_net(states).squeeze(1)

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()

    # Chooses an action using Epsilon-Greedy algorithm
    # Input:  List of (actions, state) pairs that are the possible end states at the current state of the game
    # Output: Returns the (actions, state) pair with the highest Q-value based on the policy network's prediction
    #         Returns a random (actions, state) pair if random.random() is less than epsilon
    def act(self, states):
        if random.random() > self.epsilon:
            with torch.no_grad():
                ratings = self.policy_net(torch.tensor([state for i, (action, state) in enumerate(states)], device=device, dtype=torch.float))
            return states[ratings.argmax()]
        else:
            return random.choice(states)

    # Updates the target network to match the policy network and applies the TAU update rate to the updated values
    # Input:  None
    # Output: None
    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)