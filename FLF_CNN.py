import torch
import torch.nn as nn


# Do PCA first
class FLF_Net(nn.Module):
    def __init__(self, input_channels=1):
        super(FLF_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(4 * (17 + 3), 1024),
            nn.Linear(1024, 32),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=2)
        x = self.features(x)
        x = x.transpose(1, 2).flatten(start_dim=1)
        x = self.regressor(x)
        return x
