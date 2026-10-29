"""Curvas parciais reais do GA; sem imputar dados ou misturar teste/validação."""
from pathlib import Path
import csv,json,math
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/nas_multires_1000'
OUT=ROOT/'reports/ga_partial_curves'
PCTS=[1,5,25,50,75,100]
DATASETS=[('helminth-eggs','Eggs'),('helminth-larvae','Larvae'),('protozoan-cysts','Protozoan')]
COLORS={1:'#0072B2',2:'#D55E00',3:'#009E73'}

def load():
 rows=[];candidates=[]
 for ds,label in DATASETS:
  for root in sorted((BASE/ds).glob('r*/*split*_pct*')):
   axes=dict(dataset=ds,label=label,resolution=int(root.parent.name[1:]),split=int(root.name.split('_split')[1].split('_')[0]),percentage=int(root.name.rsplit('pct',1)[1]))
   done=(root/'result.json').exists()
   path=root/'result.json' if done else root/'search/search_state.json'
   if not path.exists():continue
   record=json.loads(path.read_text());best=(record.get('best') or {}).get('result') or {}
   row={**axes,'completed':done,'status':best.get('status'),'J':best.get('J'),'kappa':best.get('svm_kappa'),'coverage':best.get('coverage'),'depth':best.get('depth'),'individual_id':best.get('individual_id'),'svm_fit_status':best.get('svm_fit_status'),'inference_path':best.get('inference_path'),'cost':best.get('cost',{})}
   local=[]
   for p in root.glob('individuals/*/result.json'):
    result=json.loads(p.read_text())
    if result.get('status')=='ok' and isinstance(result.get('svm_kappa'),(int,float)):
     candidate={**axes,'completed':done,**{k:result.get(k) for k in ['individual_id','J','svm_kappa','svm_fit_status','coverage','depth','inference_path','effective_genome','cost']}}
     local.append(candidate);candidates.append(candidate)
   row['max_candidate_kappa']=max((x['svm_kappa'] for x in local),default=None)
   rows.append(row)
 return rows,candidates

def plot(rows,resolution,diagnostic=False):
 fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
 for ax,(ds,label) in zip(axes,DATASETS):
  data=[r for r in rows if r['dataset']==ds and r['resolution']==resolution]
  for split in [1,2,3]:
   subset={r['percentage']:r for r in data if r['split']==split}
   y=[subset[p].get('kappa') if p in subset and subset[p].get('status')=='ok' else np.nan for p in PCTS]
   y=[v if isinstance(v,(int,float)) and math.isfinite(v) else np.nan for v in y]
   if not any(np.isfinite(v) for v in y):continue
   color=COLORS[split];ax.plot(range(6),y,color=color,linewidth=1.7,alpha=.8)
   for i,p in enumerate(PCTS):
    r=subset.get(p)
    if not r or not np.isfinite(y[i]):continue
    ax.plot(i,y[i],marker='o' if r['completed'] else 'D',markerfacecolor=color if r['completed'] else 'white',markeredgecolor=color,markersize=7,linestyle='none')
   if diagnostic:
    other=[subset[p].get('max_candidate_kappa') if p in subset else None for p in PCTS]
    other=[v if isinstance(v,(int,float)) else np.nan for v in other]
    ax.plot(range(6),other,color=color,linestyle=':',linewidth=1.6,marker='x',markersize=5)
  ax.set_title(label,fontsize=17,fontweight='bold');ax.set_xticks(range(6),[f'{p}%' for p in PCTS]);ax.set_xlabel('Porcentagem do pool de treino');ax.set_ylim(-.12,1.02);ax.axhline(0,color='gray',linewidth=.6);ax.grid(alpha=.2)
 axes[0].set_ylabel('Kappa da SVM na validação')
 handles=[Line2D([0],[0],color=COLORS[s],marker='o',label=f'Split {s}') for s in [1,2,3] if any(r['split']==s and r.get('kappa') is not None for r in rows if r['resolution']==resolution)]
 handles.append(Line2D([0],[0],color='black',marker='D',markerfacecolor='white',linestyle='none',label='Busca ainda em andamento'))
 if diagnostic:handles.append(Line2D([0],[0],color='gray',linestyle=':',marker='x',label='Maior kappa entre candidatos (exploratório)'))
 fig.suptitle(f'GA parcial — {resolution}×{resolution} — modelos selecionados por menor J',fontsize=15)
 fig.legend(handles=handles,loc='lower center',ncol=3 if diagnostic else 4,bbox_to_anchor=(.5,.045),fontsize=10)
 fig.text(.5,.012,'Somente medições disponíveis; linhas apenas conectam pontos. Sem extrapolação, teste ou média de splits incompletos.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.19,1,.92))
 name=f'kappa_validation_r{resolution}'+('_candidate_comparison' if diagnostic else '')
 fig.savefig(OUT/f'{name}.png',dpi=160);fig.savefig(OUT/f'{name}.svg');plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);rows,candidates=load()
 for size in sorted({r['resolution'] for r in rows}):plot(rows,size);plot(rows,size,True)
 (OUT/'snapshot.json').write_text(json.dumps(dict(time=datetime.now(ZoneInfo('America/Sao_Paulo')).isoformat(),runs=rows,candidates=candidates),indent=2))
 fields=['dataset','resolution','split','percentage','completed','status','J','kappa','coverage','depth','svm_fit_status','max_candidate_kappa','individual_id','inference_path']
 with (OUT/'validation_curves.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 winners={}
 for ds,label in DATASETS:
  valid=[r for r in rows if r['dataset']==ds and r.get('status')=='ok'];pool=[r for r in candidates if r['dataset']==ds]
  winners[ds]=dict(lowest_J=min(valid,key=lambda x:x['J']) if valid else None,highest_candidate_kappa=max(pool,key=lambda x:x['svm_kappa']) if pool else None,completed=sum(r['completed'] for r in rows if r['dataset']==ds))
 (OUT/'winners.json').write_text(json.dumps(winners,indent=2))
 print(json.dumps(winners,indent=2))
if __name__=='__main__':main()
