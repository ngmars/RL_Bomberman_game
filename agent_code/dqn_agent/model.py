import torch
import torch.nn as nn

import settings as s
from .features import N_PLANES
import time
N_ACTIONS = 6


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_PLANES, 32, kernel_size=3, padding= 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * s.COLS * s.ROWS, 256), nn.ReLU(),
            nn.Linear(256, N_ACTIONS),
        )
    def forward(self, x):
        return self.head(self.conv(x))

if __name__ == "__main__":
    torch.set_num_threads(1)
    net = QNetwork().eval()
    print("parameters:", sum(p.numel() for p in net.parameters()))

    with torch.no_grad():
        batch = torch.zeros(32, N_PLANES, s.COLS, s.ROWS)
        print("output shape:", net(batch).shape)

        one = torch.zeros(1, N_PLANES, s.COLS, s.ROWS)
        start = time.perf_counter()
        for _ in range(100):
            net(one)
    print("ms per forward:", (time.perf_counter() - start) * 10)