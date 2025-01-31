import torch
import torch.nn as nn


# CNN Branch Structure
class CNN(nn.Module):
    def __init__(self, out_channels, input_channels=1):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.slice = nn.Linear(out_channels, 32)

    def forward(self, x):
        x = self.features(x)
        x = self.slice(x)
        x = nn.Dropout(0.1)(x)
        return x


class DB_Net(nn.Module):
    def __init__(self):
        super(DB_Net, self).__init__()
        self.branch1 = CNN(124)
        self.branch2 = CNN(110)
        self.linear = nn.Sequential(
            nn.Linear(1024, 32),
            nn.Linear(32, 1)
        )

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out = torch.cat((out1, out2), dim=1)
        out = out.transpose(1, 2).flatten(start_dim=1)
        out = self.linear(out)
        return out
