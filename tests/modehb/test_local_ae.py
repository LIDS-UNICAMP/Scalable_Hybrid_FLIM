import gc
import weakref

import pytest
import torch
from torch import nn

from ga_search.modehb import local_ae
from ga_search.modehb.local_ae import effective_rank, train_local_ae


class Rec:
    def __init__(self):
        self.logs, self.alerts = [], []

    def log(self, d, step=None):
        self.logs.append((step, dict(d)))

    def alert(self, title, text, level):
        self.alerts.append((title, text, level))


def _setup(seed=0):
    torch.manual_seed(seed)
    prefix = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2))
    block = nn.Sequential(nn.Conv2d(8, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
    data = [torch.rand(4, 3, 32, 32) for _ in range(3)]
    loader = [(x, torch.zeros(4)) for x in data]
    return prefix, block, loader


@pytest.mark.parametrize("target", ["pixel", "feature", "masked"])
def test_targets_train_and_prefix_frozen(target):
    prefix, block, loader = _setup()
    pre_state = {k: v.clone() for k, v in prefix.state_dict().items()}
    blk_before = [p.detach().clone() for p in block.parameters()]
    rec = Rec()
    m = train_local_ae(prefix, block, loader, target=target, epochs=2, lr=1e-2, device="cpu", logger=rec)

    for k, v in prefix.state_dict().items():  # incl. BN running stats
        assert torch.equal(v, pre_state[k]), k
    assert all(not p.requires_grad and p.grad is None for p in prefix.parameters())
    assert not prefix.training
    assert any(not torch.equal(a, b.detach()) for a, b in zip(blk_before, block.parameters()))

    for key in ("final_recon_loss", "initial_recon_loss", "stability", "time_s", "effective_rank"):
        assert key in m and torch.isfinite(torch.tensor(m[key]))
    assert 0 < m["effective_rank"] <= 12
    assert m["steps"] == 6
    keys = set().union(*(d.keys() for _, d in rec.logs))
    assert {"ae/recon_loss", "ae/lr", "ae/epoch_time_s", "ae/grad_norm/new_block.0.weight"} <= keys
    assert not any("decoder" in k for k in m)
    assert not any(isinstance(v, nn.Module) for v in m.values())


def test_decoder_not_leaked(monkeypatch):
    refs = []
    orig = local_ae._LocalDecoder.__init__

    def spy(self, *a, **k):
        orig(self, *a, **k)
        refs.append(weakref.ref(self))

    monkeypatch.setattr(local_ae._LocalDecoder, "__init__", spy)
    prefix, block, loader = _setup()
    train_local_ae(prefix, block, loader, target="pixel", epochs=1, device="cpu")
    gc.collect()
    assert len(refs) == 1 and refs[0]() is None
    assert not any(isinstance(m, local_ae._LocalDecoder) for m in block.modules())


def test_nan_steps_skipped_then_raise_over_limit():
    prefix, block, loader = _setup()
    loader = [(x * float("nan"), y) for x, y in loader]
    rec = Rec()   # default: non-finite steps are skipped (alerted), the candidate survives
    out = train_local_ae(prefix, block, loader, target="pixel", epochs=1, device="cpu", logger=rec)
    assert out["skipped_steps"] > 0 and rec.alerts
    rec = Rec()
    with pytest.raises(FloatingPointError):
        train_local_ae(prefix, block, loader, target="pixel", epochs=1, device="cpu", logger=rec,
                       max_skip_frac=0.5)
    assert rec.alerts and rec.alerts[-1][2] == "ERROR"


def test_logger_none_and_invalid_target():
    prefix, block, loader = _setup()
    train_local_ae(prefix, block, loader, target="feature", epochs=1, device="cpu", logger=None)
    with pytest.raises(ValueError):
        train_local_ae(prefix, block, loader, target="bogus", epochs=1)


def test_effective_rank_bounds():
    z = torch.randn(512, 6)
    assert 5.0 < effective_rank(z) <= 6.0
    one = torch.randn(512, 1) @ torch.ones(1, 6)
    assert effective_rank(one) == pytest.approx(1.0, abs=1e-3)
