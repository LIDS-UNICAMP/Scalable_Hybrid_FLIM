# wrapper: reescreve o caminho do athena (/exps/moliveira/encoders/ssl_pequeno/<ds>/fps/<ds>_split1_pct100) para o L0 local
import sys, runpy, builtins, os, numpy as np
R='/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/external_ssl_pequeno'
_open=builtins.open
def fix(f):
    if isinstance(f,(str,os.PathLike)) and os.fspath(f).startswith('/exps/moliveira/encoders/ssl_pequeno/'):
        f=os.fspath(f); import re as _re; m=_re.search(r'/([a-z0-9_]+_split\d+_pct\d+)/(.*)$',f); return f"{R}/{m.group(1)}/L0/{m.group(2)}"
    return f
def op(f,*a,**k):
    return _open(fix(f),*a,**k)
    if isinstance(f,str) and f.startswith('/exps/moliveira/encoders/ssl_pequeno/'):
        tail=f.split('/fps/')[1]; ds_dir,rest=tail.split('/',1); f=f"{R}/{ds_dir}/L0/{rest}"
    return _open(f,*a,**k)
builtins.open=op
_npl=np.load
np.load=lambda f,*a,**k:_npl(fix(f),*a,**k)
import torch
_lsd=torch.nn.Module.load_state_dict
def lsd(self,sd,strict=True,**k):
    own=set(self.state_dict().keys())
    drop=[x for x in sd if x not in own and not x.startswith(('encoder.','conv'))]
    if drop: sd={x:v for x,v in sd.items() if x not in drop}
    return _lsd(self,sd,strict=strict,**k)
torch.nn.Module.load_state_dict=lsd
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
