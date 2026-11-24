"""Separate body-progress readout. Does not replace extent or presence heads."""
import torch
from torch import nn

FAMILIES = [2, 3, 4, 5, 7]

class BodyProgressHead(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32).clone())
        self.head = nn.Sequential(nn.Linear(150, 96), nn.GELU(), nn.Linear(96, 5))

    def forward(self, x):
        return self.head((x - self.mean) / self.std)

    @classmethod
    def from_boundary(cls, parent):
        net = cls(parent['mean'], parent['std'])
        indices = torch.tensor([2, 5, 8, 11, 14])
        assert parent['head.2.weight'].shape == (15, 96)
        with torch.no_grad():
            for name in ['weight', 'bias']:
                getattr(net.head[0], name).copy_(parent['head.0.' + name])
                getattr(net.head[2], name).copy_(parent['head.2.' + name][indices])
        assert torch.equal(net.mean, parent['mean']) and torch.equal(net.std, parent['std'])
        for name in ['weight', 'bias']:
            assert torch.equal(getattr(net.head[0], name), parent['head.0.' + name])
            assert torch.equal(getattr(net.head[2], name), parent['head.2.' + name][indices])
        return net
