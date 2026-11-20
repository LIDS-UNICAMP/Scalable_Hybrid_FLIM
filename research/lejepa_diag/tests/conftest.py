import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def make_model(seed=0, chans=(3, 8, 12, 16, 20)):
    """Encoder nn.Sequential de 4 blocos (L1..L4) + EmbeddingHead oficial (gap, 1280)."""
    from ga_search.modehb.pooling_head import EmbeddingHead
    torch.manual_seed(seed)
    blocks = [nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2) if i < 2 else nn.Identity())
              for i, (a, b) in enumerate(zip(chans[:-1], chans[1:]))]
    enc = nn.Sequential(*blocks)
    head = EmbeddingHead(chans[-1], 1280, "gap")
    return enc.eval(), head.eval()


@pytest.fixture
def model():
    return make_model(0)


@pytest.fixture
def data():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(20, 3, 24, 24, generator=g)
    y = torch.arange(20) % 3
    Xv = torch.randn(10, 3, 24, 24, generator=g)
    yv = torch.arange(10) % 3
    return (X, y), (Xv, yv)
