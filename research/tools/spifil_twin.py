"""spifil_twin.py -- gêmeo "SPiFiL sem backpropagation" de um candidato modehb: a MESMA arquitetura (genome),
cada bloco inicializado pelo SPiFiL (mesmo código e mesmo pool de imagens da busca), SEM autoencoder local e
SEM SSL (Evaluator._grow com ae_off=True; nada herdado de pai). Cabeça = a do original reinicializada
(não é usada pela leitura de 2ª ordem). Grava <out>/encoder.pt + config_resolved.yaml (para o eval achar os dados).
Uso: python spifil_twin.py <encoder.pt original> <out_dir> [--device cuda]"""
import sys, copy, shutil, yaml, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ga_search.modehb.run_search import Evaluator, _NullRun
src, out = Path(sys.argv[1]), Path(sys.argv[2]); dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
cfg_p = next(p / "config_resolved.yaml" for p in src.parents if (p / "config_resolved.yaml").is_file())
cfg = yaml.safe_load(cfg_p.read_text()); cfg["device"] = dev
d = torch.load(src, map_location="cpu", weights_only=False)
out.mkdir(parents=True, exist_ok=True)
ev = Evaluator(cfg, out / "_eval_tmp")
g = dict(d["genome"])
with torch.no_grad():
    pass
enc, specs, logs = ev._grow(g, {}, 0, int(g.get("seed", cfg.get("seed", 42))), _NullRun(), inherit=None, ae_off=True)
head = copy.deepcopy(d["head"])
for m in head.modules():
    if hasattr(m, "reset_parameters"): m.reset_parameters()
# o alocador do SPiFiL pode dar MENOS canais que o pedido: ajusta a 1ª conv da cabeça ao encoder real
with torch.no_grad():
    c_out = enc.cpu()(torch.zeros(1, 3, 224, 224)).shape[1]
for name, m in list(head.named_modules()):
    if isinstance(m, torch.nn.Conv2d):
        if m.in_channels != c_out:
            new_c = torch.nn.Conv2d(c_out, m.out_channels, m.kernel_size, m.stride, m.padding, bias=m.bias is not None)
            parent = head
            *path, last = name.split(".")
            for pp in path: parent = getattr(parent, pp)
            setattr(parent, last, new_c)
            print(f"[twin] cabeça {name}: in {m.in_channels} -> {c_out} (largura efetiva do SPiFiL)")
        break
torch.save({"encoder": enc.cpu().eval(), "head": head.cpu().eval(), "genome": g, "twin_of": str(src),
            "procedure": "SPiFiL por bloco, sem AE local, sem SSL (ae_off=True); cabeça reinicializada",
            "effective_specs": specs, "fidelity": 0.0, "cand_id": d.get("cand_id")}, out / "encoder.pt")
shutil.copy(cfg_p, out / "config_resolved.yaml")
shutil.rmtree(out / "_eval_tmp", ignore_errors=True)
print("ok", out / "encoder.pt", sum(p.numel() for p in enc.parameters()), "params encoder")
