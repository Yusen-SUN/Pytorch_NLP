import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, name, hidden_dim):
        super(Discriminator, self).__init__()
        self.name = name
        self.main = nn.Sequential(
            # input: batch, hidden
            nn.Linear(in_features=hidden_dim, out_features=64)
            nn.Linear(in_features=64, out_features=16)
            nn.Linear(in_features=16, out_features=1)
            nn.Sigmoid()
        )

    def forward(self, input):
        # output: batch, 1
        return self.main(input)