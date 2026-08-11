"""Shared MultiTaskNet architecture — imported by 03_multitask_nn.py and 04_screening.py."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class MultiTaskNet(nn.Module):
    def __init__(self, n_feat, hidden=512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            ResBlock(hidden), ResBlock(hidden), ResBlock(hidden),
        )
        self.head_bg     = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        self.head_dhf    = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        self.head_ehull  = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        self.head_stable = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                         nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.shared(x)
        return self.head_bg(h), self.head_dhf(h), self.head_ehull(h), self.head_stable(h)
