import random
from collections import namedtuple, deque

#Transition = namedtuple('Transition',
 #                       ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity, seed=None):
        self.memory = deque([], maxlen=capacity)

        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)