import pandas as pd, numpy as np, json, os, hashlib, sys
R='/dados/home/moliveira/Scalable_Hybrid_FLIM'
P=R+'/research/spifil_b0b4/agent1/parts/'
d=pd.read_csv(P+'A1a_data.csv',low_memory=False)
d['checkpoint']=d.checkpoint.astype(str).str.replace(R+'/','',regex=False)
ag6=pd.read_csv(R+'/research/spifil_b0b4/agent6/checkpoints_E0_E8_congelado_20261117_0159.csv')
md5map=dict(zip(ag6.checkpoint,ag6.md5))
BANDS=[('B0',0,300e3),('B1',300e3,800e3),('B2',800e3,1.5e6),('B3',1.5e6,3e6),('B4',3e6,5.5e6)]
DS=['remote_sensing','ham10000','flowers','eggs','larvae','protozoan','eggs_raw','larvae_raw','protozoan_raw','coconut_trees','aid']
PRI={'remote_sensing':1,'ham10000':1,'flowers':1,'eggs':2,'larvae':2,'protozoan':2,'eggs_raw':2,'larvae_raw':2,'protozoan_raw':2,'coconut_trees':3,'aid':3}
def md5(path):
    if not isinstance(path,str) or not path: return '',''
    if path in md5map: return md5map[path],'agent6/checkpoints_E0_E8_congelado_20261117_0159.csv'
    f=os.path.join(R,path)
    if os.path.isfile(f):
        h=hashlib.md5()
        with open(f,'rb') as fh:
            for ch in iter(lambda: fh.read(1<<20), b''): h.update(ch)
        return h.hexdigest(),'calculado_agora_jaci (17/11; se o job veio do athena, conferir com o original)'
    return '','ausente_na_jaci'
rows=[]
def add(ds,B,regua,r,origem,fonte_val,obs,lacuna=False,kv=None,status=None):
    ck=r.get('checkpoint','') if r is not None else ''
    m,ms=md5(ck) if r is not None else ('','')
    fl=r.get('flops',np.nan) if r is not None else np.nan
    rows.append(dict(dataset=ds,prioridade=PRI[ds],faixa=B,regua=regua,
        lacuna='sim' if lacuna else 'nao',
        host=(r.get('host','') if r is not None else ''),run=(r.get('run','') if r is not None else ''),
        busca=(r.get('kind','') if r is not None else ''),cand=(r.get('cand_id','') if r is not None else ''),
        fid=(('f%d'%int(r['fidelity'])) if r is not None and pd.notna(r.get('fidelity')) else ''),
        kappa_val=(round(float(kv),4) if kv is not None and pd.notna(kv) else ''),
        params=(int(r['params']) if r is not None and pd.notna(r.get('params')) else ''),
        flops_MAC=(int(fl) if pd.notna(fl) else ''),flops_2MAC=(int(2*fl) if pd.notna(fl) else ''),
        depth=(int(r['depth']) if r is not None and pd.notna(r.get('depth')) else ''),
        widths=(r.get('widths','') if r is not None and isinstance(r.get('widths',''),str) else ''),
        head_pooling=(r.get('head_pooling','') if r is not None and isinstance(r.get('head_pooling',''),str) else ''),
        wise_alpha=(r.get('wise_alpha','') if r is not None else ''),
        protect=(r.get('protect','') if r is not None and isinstance(r.get('protect',''),str) else ''),
        teste_historico_existe=('sim' if r is not None and pd.notna(r.get('test_kappa_orig',np.nan)) else 'nao'),
        status_search_state=(status or (r.get('status','') if r is not None else '')),
        checkpoint=ck,md5=m,md5_fonte=ms,fonte_kappa_val=fonte_val,origem=origem,obs=obs))
# ---------- COV ruler
cov=d[(d.fitness_sig.str.startswith('C100_kw1_cov'))&(((d.status=='ok')&(d.role=='search'))|(d.status=='ok(athena_test_only)'))].copy()
for ds in DS:
    x=cov[cov.dataset==ds]
    for B,lo,hi in BANDS:
        y=x[(x.params>lo)&(x.params<=hi)]
        if len(y):
            y=y.sort_values(['val_kappa','fidelity','flops'],ascending=[False,True,True])
            r=y.iloc[0].to_dict()
            nt=(y.val_kappa>=r['val_kappa']-0.01).sum()
            ob=f'n_jobs_cand_x_fid_na_faixa={len(y)}; candidatos a <=0,01 do melhor={nt} (ruido de val ~0,01-0,04: empate pratico)'
            if r['status']=='ok(athena_test_only)': ob+='; search_state do athena fora do bundle: populacao incompleta, so linhas testadas (vies de selecao)'
            add(ds,B,'COV_val (cov_head+StandardScaler, C=100, mt4000, alfa WiSE)',r,'busca GA atual C100_kw1',f"A1a_data.csv (search_state: {r['source']})",ob,kv=r['val_kappa'])
        else:
            z=x[x.params<=hi].sort_values(['val_kappa'],ascending=False)
            r=z.iloc[0].to_dict() if len(z) else None
            add(ds,B,'COV_val (cov_head+StandardScaler, C=100, mt4000, alfa WiSE)',r,'LACUNA: nenhum candidato na faixa; fallback = melhor de faixa menor' if r is not None else 'LACUNA total','A1a_data.csv' ,f'0 candidatos com params em ({int(lo)},{int(hi)}]',lacuna=True,kv=(r['val_kappa'] if r is not None else None))
# ---------- ORIG ruler sources
orig=[]
leg=d[(d.fitness_sig=='C100_kw0.75_head_none_mt4000')&(d.status=='ok')&(d.role=='search')]
for _,r in leg.iterrows():
    rr=r.to_dict(); rr['_k']=r.val_kappa; rr['_src']=f"A1a_data.csv (search_state legado {r['source']}; fitness C100_kw0.75_head_none_mt4000 = head sem scaler, coluna val_kappa)"; rr['_orig']='busca legada (modehb_spifil/strat/v3; pipeline antiga sem AE/protect atuais)'; rr['_regua']='ORIG_sub4000_legado'
    orig.append(rr)
ed=pd.read_csv(P+'A1b_embed_diag.csv')
ed=ed[ed.variant=='saved']
for _,e in ed.iterrows():
    m=d[(d.checkpoint==e.checkpoint)&(d.status=='ok')]
    rr=(m.iloc[0].to_dict() if len(m) else {'checkpoint':e.checkpoint,'params':e.params,'cand_id':e.cand_id,'fidelity':e.fidelity,'dataset':e.dataset})
    full = e.dataset in ('remote_sensing','larvae','larvae_raw')
    rr['_k']=e.k_orig_noscaler; rr['_src']=f"A1b_embed_diag.csv (variant=saved, k_orig_noscaler; n_train={e.n_train}, n_val={e.n_val})"
    rr['_orig']='busca GA atual C100_kw1 (re-pontuada em CPU pelo A1b)'; rr['_regua']='ORIG_full' if full else 'ORIG_subamostra(treino<=2000; val 1500 em protozoan)'
    orig.append(rr)
for line in open(R+'/research/spifil_b0b4/agent6/results/E0_val_jaci_cpu.jsonl'):
    j=json.loads(line)
    if j.get('exp')!='E0' or 'cost' not in j: continue
    m=d[(d.checkpoint==j['checkpoint'])&(d.status=='ok')]
    rr=m.iloc[0].to_dict() if len(m) else {'checkpoint':j['checkpoint']}
    rr['_k']=j['kappa']; rr['_src']='agent6/results/E0_val_jaci_cpu.jsonl (E0, ORIG_sub4000, embed_dim=%d)'%j['embed_dim']
    rr['_orig']='busca GA atual C100_kw1 (E0 do Agente 6)'; rr['_regua']='ORIG_sub4000'
    orig.append(rr)
O=pd.DataFrame(orig)
for ds in DS:
    x=O[O.dataset==ds] if len(O) else O
    for B,lo,hi in BANDS:
        y=x[(x.params>lo)&(x.params<=hi)] if len(x) else x
        if len(y):
            y=y.sort_values('_k',ascending=False); r=y.iloc[0].to_dict()
            ob=f"medidos na faixa={len(y)} ({', '.join(sorted(set(y._regua)))}); NAO comparavel entre sub-reguas; re-pontuar a faixa inteira com ORIG_full antes de congelar"
            add(ds,B,'ORIG_val ['+r['_regua']+'] (head>=1280, core.svm.fit_svm C=100, sem scaler)',r,r['_orig'],r['_src'],ob,kv=r['_k'])
        else:
            add(ds,B,'ORIG_val (head>=1280, core.svm.fit_svm C=100, sem scaler)',None,'LACUNA: ORIG-val nao medido nesta faixa','-','re-pontuar os candidatos COV da faixa (e o top-10 COV) com ORIG_full em val (svm_on_demand / E0 Agente 6); nunca usar teste',lacuna=True)
POST={('ham10000','B2'):'POS-SNAPSHOT (CHECK C3): search_state atual tem ham hybrid c00043 COV 0,612 nesta faixa; re-extrair antes de congelar',
('larvae_raw','B1'):'POS-SNAPSHOT (CHECK C3): search_state atual tem larvae_raw hybrid c00026 COV 0,915; re-extrair antes de congelar',
('flowers','B3'):'POS-SNAPSHOT (CHECK C3): estrato novo flowers hybrid B3 apareceu depois do snapshot',
('coconut_trees','B4'):'CHECK C4: coconut hybrid c00012 f5 (3,54M) esta failed no search_state (params > cap 3,5M); excluido'}
for r in rows:
    k=(r['dataset'],r['faixa'])
    if k in POST and r['regua'].startswith('COV'): r['obs']=(r['obs']+'; ' if r['obs'] else '')+POST[k]
df=pd.DataFrame(rows)
df.to_csv(sys.argv[1],index=False)
print(df.groupby(['regua','lacuna']).size())
print(df[['dataset','faixa','regua','cand','fid','kappa_val','params','md5_fonte']].to_string())
