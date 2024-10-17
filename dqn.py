import torch
import torch.nn as nn
import torch.autograd as autograd

class DQN(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    # Determine next action using the ReLU activation function
    def forward(self, x):
        x = self.fc(x)
        return x