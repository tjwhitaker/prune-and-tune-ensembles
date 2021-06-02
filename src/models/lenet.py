import torch
import torch.nn as nn
import torch.nn.functional as F


# Blueprint for small/medium/large networks
# Modified CNN based on LeNet-5 architecture.
class LeNet(nn.Module):
    def __init__(self, size="small", n_outputs=10):
        super(LeNet, self).__init__()
        # 5 epochs of training baseline
        # small = 65,058 params ~62% accuracy
        # medium = 253,290 params ~68% accuracy
        # large = 1,007,306 params ~72% accuracy

        params = {
            "small": {
                "conv": [8, 16],
                "fc": [128, 64]
            },
            "medium": {
                "conv": [16, 32],
                "fc": [256, 128]
            },
            "large": {
                "conv": [32, 64],
                "fc": [512, 256]
            },
        }

        self.conv1 = nn.Conv2d(3, params[size]["conv"][0], 5)
        self.conv2 = nn.Conv2d(
            params[size]["conv"][0], params[size]["conv"][1], 5)

        self.pool = nn.MaxPool2d(kernel_size=2)

        cnn_size = params[size]["conv"][1] * 5 * 5

        self.fc1 = nn.Linear(cnn_size, params[size]["fc"][0])
        self.fc2 = nn.Linear(params[size]["fc"][0], params[size]["fc"][1])
        self.fc3 = nn.Linear(params[size]["fc"][1], n_outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
