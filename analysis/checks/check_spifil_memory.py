import json,sys,warnings
from types import SimpleNamespace
import torch
from spifil.metrics import Euclidean,Mahalanobis
from spifil.scoring import FisherScorer,per_class_ranks
from ga_search.progressive_spifil_memory import StreamingFisherScorer
warnings.filterwarnings('ignore',category=UserWarning)
torch.set_num_threads(1)
records=[]
for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
 for metric_type in [Euclidean,Mahalanobis]:
  for n,chunk,zero in [(1,4096,False),(17,4096,False),(26,4096,False),(97,32,False),(4103,4096,False),(41,16,True)]:
   torch.manual_seed(42)
   feats=torch.randn(n,7,device=device)
   if zero:feats.zero_()
   patches=SimpleNamespace(feats=feats,labels=torch.arange(n,device=device)%3)
   old=FisherScorer()(patches,metric_type(chunk_size=chunk))
   new=StreamingFisherScorer()(patches,metric_type(chunk_size=chunk))
   equal=torch.equal(old,new)
   rank_equal=torch.equal(per_class_ranks(old,patches.labels),per_class_ranks(new,patches.labels))
   rec=dict(device=device,metric=metric_type.__name__,n=n,chunk=chunk,zero=zero,exact=equal,ranks_exact=rank_equal,max_abs=float((old-new).abs().max()))
   records.append(rec);print(json.dumps(rec),flush=True)
   assert torch.allclose(old,new,atol=1e-6,rtol=1e-6) and rank_equal,rec
print(json.dumps(dict(status='passed',cases=len(records),cuda_available=torch.cuda.is_available())))
