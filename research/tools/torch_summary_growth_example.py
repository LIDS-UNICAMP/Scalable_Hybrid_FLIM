import sys, torch, torch.nn as nn
sys.path.insert(0, '/dados/home/moliveira/exps/growth/code')
from ga_search.hmodehb.growth_chain import backbone_from_production
from ga_search.backbones import morphisms as MO
torch.set_grad_enabled(False)
P='/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/modehb_spifil/remote_sensing/state/seed42/candidates/c00306/f100/encoder.pt'
b0,_=backbone_from_production(P,'remote_sensing',224)
r1=MO.deepen_residual_zero(MO.start(b0),2,act='silu',kernel=3); b1=r1.child
r2=MO.deepen_residual_zero(MO.start(b1),2,act='silu',kernel=3); b2=r2.child
def summary(model,title):
    rows=[]; hooks=[]
    def mk(name,m):
        def h(mod,inp,out):
            x=inp[0] if isinstance(inp,tuple) and inp else inp
            o=out[0] if isinstance(out,(tuple,list)) else out
            ish=tuple(x.shape) if torch.is_tensor(x) else None; osh=tuple(o.shape) if torch.is_tensor(o) else None
            np_=sum(p.numel() for p in mod.parameters(recurse=False))
            rows.append((name,type(mod).__name__,ish,osh,np_))
        return h
    for name,m in model.named_modules():
        if name=='' : continue
        leaf=len(list(m.children()))==0
        if leaf or type(m).__name__ in ('SpifilOpsBlock','ConvBlock','InceptionConv','DecorrelatedBatchNorm'):
            hooks.append(m.register_forward_hook(mk(name,m)))
    model.eval(); model(torch.zeros(1,3,224,224))
    for h in hooks: h.remove()
    print('='*100); print(title); print('='*100)
    print(f"{'Layer (nome)':42s} {'Tipo':24s} {'Saída':22s} {'Param #':>10s}  Operação")
    print('-'*100)
    for name,t,ish,osh,np_ in rows:
        op=''
        if ish and osh and ish==osh: op='← não muda shape: '+t
        print(f"{name:42s} {t:24s} {str(list(osh)) if osh else '-':22s} {np_:>10,}  {op}")
    tot=sum(p.numel() for p in model.parameters()); print('-'*100)
    enc=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('head'))
    print(f"Total params: {tot:,} | encoder {enc:,} | cabeça {tot-enc:,}")
for m,t in ((b0,'B0 = seed01 (SPiFiL c00306)'),(b1,'B1 = B1g0c00 (deepen_residual_zero em 2)'),(b2,'B2 = B2g1c02 (deepen_residual_zero em 2 sobre o B1)')):
    summary(m,t)
