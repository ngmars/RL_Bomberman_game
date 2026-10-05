import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from gymnasium import spaces

class BombermanCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_planes = observation_space.shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(n_planes, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.conv(obs))

if __name__ == "__main__":
    space = spaces.Box(0.0, 1.0, (9, 17, 17), np.float32)
    net = BombermanCNN(space)
    print(net(torch.zeros(4, 9, 17, 17)).shape)
    print("parameters:", sum(p.numel() for p in net.parameters()))