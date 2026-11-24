"""Models of Lab 7: the oversized baseline/teacher CNN and the ~100k-parameter student."""
import math

import torch
from torch import nn


class BigCNN(nn.Module):
    """Oversized baseline (teacher), about 1.0 M parameters with the default sizes.

    conv1 1->32 (28x28) -> pool -> conv2 32->64 (14x14) -> pool -> conv3 64->128 (7x7)
    -> flatten 128*7*7 = 6272 -> fc1 144 -> dropout -> fc2 10

    The channel counts are arguments so that a structurally pruned copy (fewer filters)
    can be rebuilt as a genuinely smaller dense network.
    """

    def __init__(self, channels=(32, 64, 128), hidden=144, dropout=0.3):
        super().__init__()
        c1, c2, c3 = channels
        self.channels, self.hidden = tuple(channels), hidden
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(c3 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))      # c1 x 14 x 14
        x = self.pool(self.act(self.conv2(x)))      # c2 x 7 x 7
        x = self.act(self.conv3(x))                 # c3 x 7 x 7
        x = self.drop(self.act(self.fc1(x.flatten(1))))
        return self.fc2(x)


class SmallCNN(nn.Module):
    """Student, about 104 k parameters: same layout as the teacher, much narrower."""

    def __init__(self, channels=(16, 32, 48), hidden=36):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c3 * 7 * 7, hidden), nn.ReLU(),
                                        nn.Linear(hidden, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def prunable_layers(model):
    """(name, module) of every Conv2d / Linear layer; their weights are the pruning targets."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


# ------------------------------------------------------------------ structured pruning
def filter_l2_norms(conv):
    """L2 norm of every output filter of a conv layer (over in_channels x k x k)."""
    return conv.weight.detach().flatten(1).norm(p=2, dim=1)


def structured_prune(model: BigCNN, fraction):
    """Remove `fraction` of the filters of conv1, conv2 and conv3 (smallest L2 norm first)
    and return a new, physically smaller BigCNN with the surviving weights copied in.

    Removing filter j of a conv layer also removes input channel j of the next layer; for
    conv3 it removes the 7*7 = 49 input features of fc1 that came from channel j.
    """
    convs = [model.conv1, model.conv2, model.conv3]
    keep = []
    for conv in convs:
        n_keep = max(1, math.ceil((1 - fraction) * conv.out_channels))
        idx = torch.argsort(filter_l2_norms(conv), descending=True)[:n_keep]
        keep.append(torch.sort(idx).values)

    small = BigCNN(channels=[len(k) for k in keep], hidden=model.hidden)
    with torch.no_grad():
        prev = torch.arange(1)                              # the single input channel
        for new, old, k in zip([small.conv1, small.conv2, small.conv3], convs, keep):
            new.weight.copy_(old.weight[k][:, prev])
            new.bias.copy_(old.bias[k])
            prev = k
        cols = (keep[2][:, None] * 49 + torch.arange(49)[None, :]).flatten()   # (C, H, W) flatten order
        small.fc1.weight.copy_(model.fc1.weight[:, cols])
        small.fc1.bias.copy_(model.fc1.bias)
        small.fc2.load_state_dict(model.fc2.state_dict())
    return small, [len(k) for k in keep]
