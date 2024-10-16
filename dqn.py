import torch
import torch.nn as nn
import torch.autograd as autograd

class DQN(nn.Module):
    def __init__(self, input_shape, actions_dim, hidden_dim):
        super(DQN, self).__init__()
<<<<<<< Updated upstream
        self.input_shape = input_shape
        self.num_actions = actions_dim

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
=======
>>>>>>> Stashed changes

        self.fc = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
<<<<<<< Updated upstream
            nn.Linear(hidden_dim, actions_dim)
=======
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
>>>>>>> Stashed changes
        )

    # Determine next action using the ReLU activation function
    def forward(self, x):
        x = self.fc(x)
        return x