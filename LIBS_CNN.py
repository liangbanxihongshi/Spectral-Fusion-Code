import torch
import torch.nn as nn


# The CNN branch of LIBS,
# the decision fusion takes this result and the NIRS branch result into Multiple Linear Regression (MLR) for fusion
class LIBS_Net(nn.Module):
    def __init__(self, input_channels=1):
        super(LIBS_Net, self).__init__()
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
            nn.Linear(16 * 124, 1024),
            nn.Linear(1024, 32),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.regressor(x)
        return x
