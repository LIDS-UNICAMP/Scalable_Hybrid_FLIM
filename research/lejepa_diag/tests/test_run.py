import json

import pandas as pd
import torch

from research.lejepa_diag import run
from research.lejepa_diag.tests.conftest import make_model


def test_runner_all_stages_idempotent(tmp_path):
    for i, name in enumerate(("p0.pt", "p1.pt")):
        enc, head = make_model(i)
        torch.save({"encoder": enc, "head": head}, tmp_path / name)
    man = tmp_path / "checkpoint_manifest.json"
    man.write_text(json.dumps({"pairs": [{"pair_id": "syn", "p0_path": str(tmp_path / "p0.pt"),
                                          "p1_path": str(tmp_path / "p1.pt"), "loader": "synthetic",
                                          "dataset": "synthetic", "split": 1, "image_size": 16,
                                          "n_train": 18, "n_val": 9}]}))
    out = tmp_path / "data"
    argv = ["--manifest", str(man), "--out-dir", str(out), "--stage", "matrix", "--stage", "layers",
            "--stage", "texture", "--sigmas", "0", "1", "--batch-size", "4", "--layer-k", "6"]
    run.main(argv)
    m = pd.read_csv(out / "metrics.csv")
    assert len(m) == 6 and set(m.readout) == {"GAP", "COV", "COV+scaler"}
    assert set(m.model) == {"P0", "P1"} and m.loc[m.model == "P1", "ctrl_cka"].notna().all()
    assert m.loc[m.readout == "COV+scaler", "svm_protocol"].str.contains("scaler=standard").all()
    assert m.loc[m.readout == "COV", "svm_protocol"].str.contains("scaler=none").all()
    lm = pd.read_csv(out / "layer_metrics.csv")
    assert len(lm) == 2 * 4 * 2 and set(lm.layer) == {"L1", "L2", "L3", "L4"}
    tm = pd.read_csv(out / "texture_metrics.csv")
    assert len(tm) == 2 * 3 * 2
    assert (tm.loc[tm.sigma == 0, "cos_mean"] > 0.999999).all()
    ts = m.timestamp.tolist()
    run.main(argv)                                                  # idempotente: nada recalculado
    assert pd.read_csv(out / "metrics.csv").timestamp.tolist() == ts
    run.main(argv + ["--force"])
    assert len(pd.read_csv(out / "metrics.csv")) == 6
