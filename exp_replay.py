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
        """Save a transition"""
        e = self.transition(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)