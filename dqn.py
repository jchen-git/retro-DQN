import torch
import torch.nn as nn
import torch.autograd as autograd

class DQN(nn.Module):
    def __init__(self, input_shape, actions_dim, hidden_dim):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = actions_dim

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, actions_dim)
        )

    # Determine next action using the ReLU activation function
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)