import random
import torch
import numpy as np
from collections import namedtuple, deque

#Transition = namedtuple('Transition',
#                       ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity, batch_size, device, seed=None):
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple("Transition", field_names=["state", "action", "reward", "next_state", "done"])
        self.batch_size = batch_size
        self.device = device

        if seed is not None:
            random.seed(seed)

    def append(self, state, action, reward, next_state, done):
        e = self.transition(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        transitions = random.sample(self.memory, k=self.batch_size)

        states = torch.tensor(np.array([t.state for t in transitions if t is not None])).float().to(self.device)
        actions = torch.tensor(np.array([t.action for t in transitions if t is not None])).long().to(self.device)
        rewards = torch.tensor(np.array([t.reward for t in transitions if t is not None])).float().to(self.device)
        next_states = torch.tensor(np.array([t.next_state for t in transitions if t is not None])).float().to(
            self.device)
        dones = torch.tensor(np.array(np.array([t.done for t in transitions if t is not None])).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)