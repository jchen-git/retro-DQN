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
from preprocessing import preprocess, stack_frame

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
            self.replay_memory = ReplayMemory(self.replay_memory_size, self.batch_size, device)

            self.target_net = DQN(self.input_shape, self.n_actions, self.hidden_layer_num).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def optimize(self, mini_batch):
        states, actions, rewards, next_states, dones = mini_batch

        # Get expected Q values from policy model
        q_expected_current = self.policy_net(states)
        q_expected = q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.target_net(next_states).detach().max(1)[0]

        # Compute Q targets for current states
        q_targets = rewards + (self.discount_factor * q_targets_next * (1 - dones))

        # Compute loss
        loss = self.loss_fn(q_expected, q_targets)

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
            return np.unravel_index(torch.argmax(action_values), action_values.shape)[1]
        else:
            return random.choice(np.arange(self.n_actions))

def stack_frames(frames, curr_state, image_crop, image_size, is_new=False):
    frame = preprocess(curr_state, image_crop, image_size)
    frames = stack_frame(frames, frame, is_new)

    return frames

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
    #INPUT_SHAPE = (35, 204, 85, 170)
    INPUT_SHAPE = (1, 64, 64)

    agent = Agent(INPUT_SHAPE, num_actions,"tetris", training=True)
    is_training = True
    rewards_per_episode = []
    epsilon_history = []

    if is_training:
        step_count = 0

    for episode in range(agent.epoch):
        obs = env.reset()

        done = False
        episode_reward = 0.0

        state = stack_frames(None, obs, agent.image_crop, agent.image_resize, True)

        while not done:
            env.render()
            action = agent.act(state, env.action_space)
            next_obs, rew, done, info = env.step(agent.actions[int(action)])
            episode_reward += rew
            next_state = stack_frames(state, next_obs, agent.image_crop, agent.image_resize, False)

            print(episode_reward)

            if is_training:
                agent.replay_memory.append(state, action, rew, next_state, done)

                step_count += 1

            state = next_state

        rewards_per_episode.append(episode_reward)

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        epsilon_history.append(agent.epsilon)

        if (len(agent.replay_memory)) > agent.batch_size:
            batch = agent.replay_memory.sample()

            agent.optimize(batch)

            if step_count > agent.network_sync_rate:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                step_count = 0

    env.close()