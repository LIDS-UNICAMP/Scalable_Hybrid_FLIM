"""Valida os YAMLs de ga_search/config/modehb: carregam e têm as chaves obrigatórias."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

CFG = Path(__file__).resolve().parents[2] / "ga_search" / "config" / "modehb"
DATASETS = ["larvae", "remote_sensing", "flowers", "ham10000", "aid", "eggs",
            "coconut_trees", "protozoan", "food101"]
N_CLASSES = {"remote_sensing": 14, "flowers": 5, "aid": 30, "coconut_trees": 2, "ham10000": 7,
             "food101": 101, "eggs": 9, "larvae": 2, "protozoan": 7}
ABLATIONS = ["init", "softhebb", "whitening_stat", "whitening_dbn", "batchnorm", "activation",
             "pooling", "local_ae", "ssl", "simclr_sigreg", "sigreg_vs_weak", "tta",
             "progressive_vs_direct"]


def load(p):
    with open(p) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("name", DATASETS)
def test_dataset(name):
    c = load(CFG / "datasets" / f"{name}.yaml")
    for k in ("name", "layout", "root", "images_dir", "split_json", "train_key", "eval",
              "split_sha256", "expected_counts", "image_size", "classes", "n_classes"):
        assert k in c, k
    assert c["layout"] == "organized" and c["image_size"] == 224
    assert c["eval"]["source"] == "validation" and c["eval"]["search_uses_labels"] is False
    assert c["n_classes"] == N_CLASSES[name] == len(c["classes"])
    split = Path(c["split_json"].format(root=c["root"], split=1))
    if split.exists():  # verificação de integridade quando o dado está montado
        assert hashlib.sha256(split.read_bytes()).hexdigest() == c["split_sha256"]
        j = json.loads(split.read_text())
        assert {k: len(j[k]) for k in ("train", "validation", "test")} == c["expected_counts"]


def test_search():
    s = load(CFG / "search.yaml")
    assert s["project"] == "progressive_mo_dehb" and s["entity"] == "ophira-ai"
    assert s["datasets"] == [d for d in DATASETS if d != "food101"]  # food101 fora do benchmark (tempo)
    assert s["fidelities"] == [5, 15, 45]
    assert (s["fidelity_split"]["ae"], s["fidelity_split"]["ssl"]) == (1, 2)
    p = s["population"]
    assert p["initial_size"] >= 16 and p["spifil_offspring_share"] == 0.75
    assert p["random_twin_at_max_fidelity"] is True and p["exploration_epsilon"] == 0.1
    assert set(p["initial_grid"]["filters"]) == {16, 32, 64, 128}
    assert set(p["initial_grid"]["init"]) == {"spifil", "random"}
    assert {"stage1", "stage2"} <= set(s["stages"])
    assert [o["name"] for o in s["objectives"]] == ["quality", "flops"]
    assert "reference_point" in s["hypervolume"]
    assert s["fitness"]["knn_view_consistency"] == 0.5 == s["fitness"]["clustering_stability"]
    assert s["tta"]["search"] == 1
    t = s["training"]
    assert {"batch_size", "lr_new", "lr_old"} <= set(t)
    assert t["auto_adjust_batch_size"] is False and t["auto_adjust_lr"] is False
    assert "collapse" in s and "spectral_every" in s and s["seeds"]
    r = s["resources"]
    assert r["gpu_memory_target_gb"] == {"min": 88, "max": 91}
    assert isinstance(r["num_workers"], int) and isinstance(r["ram_cache_gb"], (int, float))  # medidos no Q1
    assert "MEASURED_IN_Q1" not in json.dumps(s)
    assert all(v.startswith("artifacts/modehb/{dataset}/") for v in s["paths"].values())


@pytest.mark.parametrize("name", ABLATIONS)
def test_ablation(name):
    a = load(CFG / "ablations" / f"{name}.yaml")
    for k in ("name", "factor", "levels", "fixed", "seeds", "budget", "base"):
        assert k in a, k
    assert len(a["levels"]) >= 2
    assert a["factor"] not in a["fixed"]  # único fator livre
    assert a["seeds"] == load(CFG / "search.yaml")["seeds"]


def test_ablations_share_budget():
    budgets = {name: load(CFG / "ablations" / f"{name}.yaml")["budget"] for name in ABLATIONS}
    assert len({json.dumps(b, sort_keys=True) for b in budgets.values()}) == 1


def test_search_v3_spifil_only_space():
    s = load(CFG / "search_v3.yaml")
    p = s["population"]
    assert p["random_baseline"] is False and p["initial_grid"]["init"] == ["spifil"]
    assert p["initial_size"] >= 16 and p["random_twin_at_max_fidelity"] is False
    assert s["operators_from_start"] is True and s["search_space_from_yaml"] is True
    sp = s["stages"]["stage2"]["space"]
    assert sp["operator"] == ["conv", "dwsep", "inception"] and sp["softhebb"] == [True, False]
    assert sp["norm"] == ["none", "batch", "zca", "dbn", "batch_zca", "batch_dbn"]
    assert sp["pool"] == ["none", "max", "avg"] and sp["residual"] == [True, False]
    assert sp["pooling"] == ["gap", "gem", "moments"] and sp["kernel"] == [3, 5, 7]
    assert s["fitness_mode"] == "svm_kappa" and s["spifil_ops"] is True and s["max_brackets"] == 10
    assert s["fidelities"] == [5, 15, 45, 100, 200]
