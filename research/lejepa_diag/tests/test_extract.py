import torch

from research.lejepa_diag.extract import extract, official_embedding


def test_shapes_and_official(model, data):
    enc, head = model
    (X, y), _ = data
    ex = extract(enc, head, (X, y), batch_size=6, max_map_images=5)
    assert ex.embedding.shape == (20, 1280) and ex.embedding.dtype == torch.float32
    assert list(ex.layers) == ["L1", "L2", "L3", "L4"]
    assert [ex.layers[k].shape for k in ex.layers] == [(20, 8), (20, 12), (20, 16), (20, 20)]
    assert ex.maps["L1"].shape == (5, 8, 12, 12) and ex.maps["L4"].shape == (5, 20, 6, 6)
    with torch.no_grad():
        ref = head(enc(X)).float()
    assert torch.allclose(ex.embedding, ref, atol=1e-5)
    assert ex.labels.tolist() == y.tolist() and ex.time_s > 0 and ex.peak_mem_bytes >= 0
    assert ex.mem_kind == "cpu_rss_peak_delta" and ex.tracemalloc_peak > 0


def test_gap_layer_matches_last_map(model, data):
    enc, head = model
    (X, y), _ = data
    ex = extract(enc, None, (X, y), batch_size=7)
    assert torch.allclose(ex.embedding, ex.layers["L4"], atol=1e-6)   # sem head: GAP do ultimo mapa


def test_deterministic_and_batch_invariant(model, data):
    enc, head = model
    (X, y), _ = data
    a = extract(enc, head, (X, y), batch_size=4)
    b = extract(enc, head, (X, y), batch_size=4)
    c = extract(enc, head, (X, y), batch_size=20)
    assert torch.equal(a.embedding, b.embedding)
    assert all(torch.equal(a.layers[k], b.layers[k]) for k in a.layers)
    assert torch.allclose(a.embedding, c.embedding, atol=1e-5)


def test_dataloader_input_and_hooks_removed(model, data):
    enc, head = model
    (X, y), _ = data
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=9)
    ex = extract(enc, head, dl, batch_size=4)
    assert ex.embedding.shape[0] == 20
    assert all(len(m._forward_hooks) == 0 for m in enc.children())
    assert torch.allclose(ex.embedding, official_embedding(enc, head, X), atol=1e-5)
