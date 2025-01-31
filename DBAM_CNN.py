import torch
import torch.nn as nn


# CNN Branch Structure
class CNN(nn.Module):
    def __init__(self, out_channels, input_channels=1):
        super(CNN, self).__init__()
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

        self.spatial = SpatialAttention()
        self.slice = nn.Linear(out_channels, 32)

    def forward(self, x):
        x = self.features(x)
        x = self.spatial(x)
        x = self.slice(x)
        x = nn.Dropout(0.1)(x)
        return x


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.conv(combined)
        attention_map = self.sigmoid(attention_map)
        return x * attention_map.expand_as(x)


# Channel Attention Module
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a, b, c = x.size()
        avg_out = self.avg_pool(x).squeeze(dim=-1)
        max_out = self.max_pool(x).squeeze(dim=-1)
        max_out = self.fc(max_out).unsqueeze(2)
        avg_out = self.fc(avg_out).unsqueeze(2)
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = out.expand_as(x)
        return x * out


class DBAM_Net(nn.Module):
    def __init__(self):
        super(DBAM_Net, self).__init__()
        self.branch1 = CNN(124)
        self.branch2 = CNN(110)
        self.se = SEBlock(32)
        self.linear = nn.Sequential(
            nn.Linear(1024, 32),
            nn.Linear(32, 1)
        )

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out = torch.cat((out1.unsqueeze(0), out2.unsqueeze(0)), dim=2).squeeze(0)
        out = self.se(out)
        out = out.transpose(1, 2).flatten(start_dim=1)
        out = self.linear(out)
        return out
