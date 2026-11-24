"""Frozen post-selection input crossover. Saved equal-input scores are reused."""
from pathlib import Path
import sys,json,hashlib,time,datetime,importlib.util
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;B=U.parent/'word100_body_boundary_20261017'
SEEDS=(20261013,20261014,20261015);VIEWS=('masked','sensor')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(a.tobytes()).hexdigest()
def write(p,d):
    with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def now():return datetime.datetime.now().astimezone().isoformat()
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def pin():
    p=U/'CROSS_PROTOCOL.json'
    write(p,dict(at=now(),code_sha256=sha(__file__),plan_sha256=sha(U/'PLAN.json'),branches=['C_on_W_input','W_on_C_input'],
        selection_influence=False,unchanged='Reuse original saved selected-epoch logits for byte-identical input; never rescore them.',
        changed='Only changed model/input pairs are inferred once after W selection. Original32-row group and maximum sequence padding length retained. Batch may have fewer rows; eval normalization is independent of batch mates. No augmentation or model update.',
        scope='DEV300 already exposed. 3seeds both views, all300 retained per branch. Diagnostic only, not blind generalization.'))
    print('CROSS PROTOCOL frozen before training; no inference')
def run():
    protocol=read(U/'CROSS_PROTOCOL.json');assert protocol['code_sha256']==sha(__file__) and protocol['plan_sha256']==sha(U/'PLAN.json')
    assert read(U/'VERIFIED.json')['status']=='PASS_SAVED_SCORES_AND_ACTUAL_BATCH_TRACE'
    assert not (U/'RUNNING.json').exists()
    sys.path.insert(0,str(B));import training_driver as dr
    import training_core as co
    import torch
    dr.configure_torch()
    c=read(U/'CONTEXT.json');o=read(B/'training_CONTEXT.json')
    for p,h in {**c['pins'],**read(U/'PLAN.json')['protected']}.items():assert sha(p)==h,p
    rows=[r for r in c['ordered_rows'] if r['role']=='development_only'];oldrows={r['clip_id']:r for r in o['ordered_rows'] if r['role']=='development_only'}
    arrays={};identical={}
    for r in rows:
        cid=r['clip_id'];arrays[cid]={};identical[cid]={}
        for v in VIEWS:
            with np.load(r['views'][v]['channel_file'],allow_pickle=False) as z:new=z['x_whole'].copy()
            with np.load(oldrows[cid]['views'][v]['channel_file'],allow_pickle=False) as z:old=z['x_whole'].copy()
            assert new.shape==old.shape
            arrays[cid][v]={'old':old,'new':new};identical[cid][v]=same(old,new)
    for seed in SEEDS:
        for branch,unit,key in [('C_on_W_input',B,'new'),('W_on_C_input',U,'old')]:
            path=U/f'CROSS_{branch}_{seed}.json';start=U/f'CROSS_{branch}_{seed}_START.json'
            ckpath=unit/f'training_C_{seed}_best.pt';savedpath=unit/f'training_C_{seed}_RESULT.json'
            if path.exists():
                prior=read(path);assert prior['checkpoint_sha256']==sha(ckpath) and prior['protocol_sha256']==sha(U/'CROSS_PROTOCOL.json');continue
            assert not start.exists(),'Preserve incomplete crossover; do not silently re-infer.'
            ck=torch.load(ckpath,map_location='cpu',weights_only=True);m=dr.MaskedSignTCN(**co.MODEL);m.load_state_dict(ck['state_dict']);m.eval()
            state=dr.state_hash(m.state_dict());saved=read(savedpath)
            write(start,dict(at=now(),checkpoint_sha256=sha(ckpath),context_sha256=sha(U/'CONTEXT.json'),protocol_sha256=sha(U/'CROSS_PROTOCOL.json')))
            predictions={};metrics={};audit={};started=time.perf_counter()
            with torch.no_grad():
                for view in VIEWS:
                    scores={p['clip_id']:p for p in saved['predictions'][view] if identical[p['clip_id']][view]};reused=sorted(scores);inferred=[]
                    for at in range(0,len(rows),32):
                        group=rows[at:at+32];changed=[r for r in group if not identical[r['clip_id']][view]]
                        if not changed:continue
                        n=max(len(arrays[r['clip_id']][view][key]) for r in group)
                        x=np.zeros((len(changed),n,280),np.float32);pad=np.zeros((len(changed),n),np.float32)
                        for i,r in enumerate(changed):
                            a=arrays[r['clip_id']][view][key];x[i,:len(a)]=a;pad[i,:len(a)]=1
                        logits=m(torch.from_numpy(x),torch.from_numpy(pad)).numpy()
                        assert logits.shape==(len(changed),100) and np.isfinite(logits).all()
                        for r,a in zip(changed,logits):
                            cid=r['clip_id'];assert cid not in scores;scores[cid]=co.prediction(r,a);inferred.append(dict(clip_id=cid,input_sha256=ah(arrays[cid][view][key])))
                    assert set(scores)==set(oldrows) and len(scores)==300
                    predictions[view]=[scores[k] for k in sorted(scores)];metrics[view]=co.detailed_metrics(predictions[view],c['missingness_groups'][view]);audit[view]=dict(reused_ids=reused,new_inferences=inferred)
            assert dr.state_hash(m.state_dict())==state
            write(path,dict(at=now(),branch=branch,seed=seed,checkpoint_sha256=sha(ckpath),source_scores_sha256=sha(savedpath),protocol_sha256=sha(U/'CROSS_PROTOCOL.json'),predictions=predictions,metrics=metrics,audit=audit,seconds=time.perf_counter()-started,model_state_unchanged=True,selected_model_changed=False))
            print(branch,seed,{v:metrics[v]['top1_correct'] for v in VIEWS},flush=True)
    # Independent rank/CE arithmetic; does not call the training metric implementation.
    spec=importlib.util.spec_from_file_location('cross_score_audit',B/'verify_training_scores.py');sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
    evidence={};pins={}
    for seed in SEEDS:
        for branch in ('C_on_W_input','W_on_C_input'):
            p=U/f'CROSS_{branch}_{seed}.json';r=read(p);pins[str(p)]=sha(p)
            for v in VIEWS:
                counted=sc.recount(r['predictions'][v],oldrows,{});mm=sc.detailed(counted,c['missingness_groups'][v]);sc.same(r['metrics'][v],mm)
                expected_reuse={cid for cid in oldrows if identical[cid][v]};a=r['audit'][v]
                assert set(a['reused_ids'])==expected_reuse and {q['clip_id'] for q in a['new_inferences']}==set(oldrows)-expected_reuse
                source=read((B if branch=='C_on_W_input' else U)/f'training_C_{seed}_RESULT.json');saved={q['clip_id']:q for q in source['predictions'][v]}
                for q in r['predictions'][v]:
                    if q['clip_id'] in expected_reuse:assert q==saved[q['clip_id']]
                evidence[f'{branch}_{seed}_{v}']=dict(metrics=mm,reused=len(expected_reuse),newly_inferred=300-len(expected_reuse))
    write(U/'CROSS_VERIFIED.json',dict(at=now(),status='PASS_SAVED_CROSSOVER_SCORES',evidence=evidence,pins=pins,protocol_sha256=sha(U/'CROSS_PROTOCOL.json'),no_model_selection=True,not_blind_evaluation=True))
if __name__=='__main__':pin() if sys.argv[1]=='pin' else run()
