import random
from collections import namedtuple, deque

class ReplayMemory(object):
    def __init__(self, capacity, batch_size, device, seed=None):
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple("transition", ("state", "action", "reward", "next_state", "done"))
        self.batch_size = batch_size
        self.device = device

        if seed is not None:
            random.seed(seed)

    def append(self, state, action, reward, next_state, done):
        e = self.transition(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)