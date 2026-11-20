import sys, torch, torch.nn as nn
sys.path.insert(0,"/dados/home/moliveira/Scalable_Hybrid_FLIM")
torch.set_num_threads(2)
R="/dados/home/moliveira/Scalable_Hybrid_FLIM/"
for ck,exp in [("artifacts/modehb_hybrid/remote_sensing/state/seed42/candidates/c00009/f5/encoder.pt",875445760),
               ("artifacts/modehb_lemonade/eggs/state/seed42/candidates/c00009/f5/encoder.pt",490688352)]:
    c=torch.load(R+ck,map_location="cpu",weights_only=False)
    m=nn.Sequential(c["encoder"].float().eval(),c["head"].float().eval())
    macs=0
    def h(mod,i,o):
        global macs
        if isinstance(mod,nn.Conv2d): macs+=o.numel()*mod.kernel_size[0]*mod.kernel_size[1]*mod.in_channels//mod.groups
        elif isinstance(mod,nn.Linear): macs+=o.numel()*mod.in_features
    hs=[x.register_forward_hook(h) for x in m.modules() if isinstance(x,(nn.Conv2d,nn.Linear))]
    with torch.no_grad(): out=m(torch.zeros(1,3,224,224))
    p=sum(x.numel() for x in m.parameters())
    print(ck.split('/')[-5],ck.split('/')[-3],"MAC",macs,"registry",exp,"ratio",round(macs/exp,4),"params",p,"out",tuple(out.shape))
