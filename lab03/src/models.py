"""The three architectures compared in Lab 3."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 x 14 x 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 x 7 x 7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class PatchTransformer(nn.Module):
    """28x28 image -> 16 patches of 7x7 -> Linear(49, 64) -> 2 encoder layers -> mean pool -> Linear(64, 10).

    A learned position embedding (16 x 64) is added after the projection; without it
    the encoder could not tell where in the image a patch came from.
    """

    def __init__(self, patch=7, d_model=64, nhead=4, dim_ff=128, num_layers=2):
        super().__init__()
        self.patch = patch
        self.n_patches = (28 // patch) ** 2
        self.proj = nn.Linear(patch * patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def patchify(self, x):
        b = x.shape[0]
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)          # (B, 1, 4, 4, 7, 7)
        return x.reshape(b, self.n_patches, p * p)      # (B, 16, 49)

    def forward(self, x):
        x = self.proj(self.patchify(x)) + self.pos
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


MODELS = {"MLP": MLP, "CNN": CNN, "Attention": PatchTransformer}


def build(name):
    return MODELS[name]()
