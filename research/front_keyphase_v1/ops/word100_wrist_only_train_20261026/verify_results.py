"""Saved-score, checkpoint and actual input-batch audit. No model forward/training."""
from pathlib import Path
import sys,json,hashlib,importlib.util,datetime,math
from collections import Counter
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;B=U.parent/'word100_body_boundary_20261017'
SEEDS=(20261013,20261014,20261015);VIEWS=('masked','sensor')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def ah(a):return hashlib.sha256(a.tobytes()).hexdigest()
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
sc=module('wrist_saved_score_checker',B/'verify_training_scores.py')
tr=module('wrist_saved_trace_checker',B/'verify_training_trace.py')
def schedule(cid,epoch):
    a=int.from_bytes(hashlib.sha256(cid.encode()).digest(),'big')&1
    b=int.from_bytes(hashlib.sha256(('sensor_view:'+cid).encode()).digest(),'big')&1
    return ['sensor' if (((epoch-1)//2+b)&1) else 'masked','x_context' if ((epoch+a)&1) else 'x_whole']
def pack(xs,labels):
    n=max(map(len,xs));x=np.zeros((len(xs),n,280),np.float32);p=np.zeros((len(xs),n),np.float32)
    for i,a in enumerate(xs):x[i,:len(a)]=a;p[i,:len(a)]=1
    return x,p,np.asarray(labels,np.int64)
def actual_batches(rows,records,epochs,seed,arrays):
    assert len(records)==29*epochs
    k=0
    for epoch in range(1,epochs+1):
        order=np.random.default_rng(seed+epoch).permutation(899)
        allids=[]
        for at in range(0,899,32):
            batch=[rows[i] for i in order[at:at+32]];r=records[k];k+=1
            ids=[v['clip_id'] for v in batch];choices=[schedule(c,epoch) for c in ids]
            assert r['seed']==seed and r['epoch']==epoch and r['clip_ids']==ids and r['choices']==choices
            xs=[arrays[c][s][t] for c,(s,t) in zip(ids,choices)]
            x,p,y=pack(xs,[v['label_index'] for v in batch])
            assert [r['x_sha256'],r['pad_sha256'],r['labels_sha256']]==[ah(x),ah(p),ah(y)]
            allids+=ids
        assert Counter(allids)==Counter(v['clip_id'] for v in rows)
    return dict(actual_batches_checked=k,actual_video_exposures=epochs*899,actual_tensor_hashes_checked=k*3,all_899_once_per_epoch=True)
def verify():
    import torch
    assert not (U/'RUNNING.json').exists() and (U/'TRAINING_DONE.json').exists()
    c=read(U/'CONTEXT.json');plan=read(U/'PLAN.json');original=read(B/'training_CONTEXT.json');pins={}
    for p,h in {**c['pins'],**plan['protected']}.items():assert digest(p)==h,p
    assert c['seeds']==list(SEEDS) and c['training']==original['training'] and c['model']==original['model']
    assert [(r['clip_id'],r['role'],r['label_index'],r['signer_id']) for r in c['ordered_rows']]==[(r['clip_id'],r['role'],r['label_index'],r['signer_id']) for r in original['ordered_rows']]
    train=[r for r in c['ordered_rows'] if r['role']=='train_candidate'];dev={r['clip_id']:r for r in c['ordered_rows'] if r['role']=='development_only'}
    assert len(train)==899 and len(dev)==300 and c['missingness_groups']==original['missingness_groups']
    arrays={}
    for row in train:
        arrays[row['clip_id']]={}
        for view in VIEWS:
            with np.load(row['views'][view]['channel_file'],allow_pickle=False) as z:arrays[row['clip_id']][view]={k:z[k].copy() for k in ('x_whole','x_context')}
    results={};oldresults={};traces={};pairs={}
    for seed in SEEDS:
        prefix=f'training_C_{seed}';rp=U/(prefix+'_RESULT.json');hist=U/(prefix+'_history.json');best=U/(prefix+'_best.pt');last=U/(prefix+'_last.pt');bp=U/f'BATCH_TRACE_{seed}.jsonl'
        r=read(rp);o=read(B/(prefix+'_RESULT.json'));h=read(hist)
        assert r['status']=='COMPLETE_DEVELOPMENT_ONLY' and r['context_sha256']==digest(U/'CONTEXT.json')
        assert r['checkpoint_sha256']==digest(best) and r['initial_state_sha256']==o['initial_state_sha256']
        assert r['selected_scores_reinferred'] is False and r['user_data_used'] is False and r['production_model_replaced'] is False
        results[str(seed)]={};oldresults[str(seed)]={};pairs[str(seed)]={}
        for view in VIEWS:
            rr=sc.recount(r['predictions'][view],dev,{});oo=sc.recount(o['predictions'][view],dev,{})
            mm=sc.detailed(rr,c['missingness_groups'][view]);om=sc.detailed(oo,c['missingness_groups'][view]);sc.same(r['metrics'][view],mm);sc.same(o['metrics'][view],om)
            results[str(seed)][view]=mm;oldresults[str(seed)][view]=om;pairs[str(seed)][view]=sc.paired(oo,rr)
        hh=tr.inspect_history(h,[r['clip_id'] for r in train],'C',r)
        ck=torch.load(best,map_location='cpu',weights_only=True);lk=torch.load(last,map_location='cpu',weights_only=True)
        assert ck['candidate_id']=='W' and ck['feature_schema']['context_sha256']==digest(U/'CONTEXT.json')
        ch=tr.inspect_checkpoint_dicts(torch,lk,ck,r,h,c,digest(U/'CONTEXT.json'),'C',seed)
        batches=[json.loads(l) for l in bp.read_text('utf8').splitlines() if l.strip()]
        bh=actual_batches(train,batches,len(h),seed,arrays)
        traces[str(seed)]=dict(history=hh,checkpoint=ch,actual_batch_trace=bh)
        for p in (rp,hist,best,last,bp):pins[str(p)]=digest(p)
    summary={name:{v:sc.three_seed_summary([rr[str(s)][v] for s in SEEDS]) for v in VIEWS} for name,rr in [('C',oldresults),('W',results)]}
    differences={v:[results[str(s)][v]['top1_correct']-oldresults[str(s)][v]['top1_correct'] for s in SEEDS] for v in VIEWS}
    gate=dict(sensor_sum_improves=sum(differences['sensor'])>0,sensor_nonregression_at_least_two=sum(d>=0 for d in differences['sensor'])>=2,sensor_improved_seed=sum(d>0 for d in differences['sensor'])>=1,masked_mean_nonregression=sum(differences['masked'])>=0,no_input_exclusion=all(results[str(s)][v]['input_failures']==0 for s in SEEDS for v in VIEWS))
    for p in (U/'CONTEXT.json',U/'PLAN.json',Path(__file__),B/'verify_training_scores.py',B/'verify_training_trace.py'):pins[str(p)]=digest(p)
    out=dict(at=datetime.datetime.now().astimezone().isoformat(),status='PASS_SAVED_SCORES_AND_ACTUAL_BATCH_TRACE',results=results,baseline=oldresults,summary=summary,paired=pairs,traces=traces,gate=dict(checks=gate,passed=all(gate.values())),pins=pins,independence='Same Codex separate arithmetic plus saved tensor/actual batch trace audit; no independent human truth or blind test.',production_changed=False)
    with (U/'VERIFIED.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=out['status'],differences=differences,gate=gate)),flush=True)
def selftest():
    # Unequal lengths and an all-zero real frame must retain pad=1.
    a=np.ones((2,280),np.float32);b=np.zeros((1,280),np.float32);x,p,y=pack([a,b],[3,4])
    assert x.shape==(2,2,280) and p.tolist()==[[1,1],[1,0]] and y.tolist()==[3,4]
    assert not x[1].any() and p[1,0]==1
    for cid in ['first','last','crossing']:
        counts=Counter(tuple(schedule(cid,e)) for e in range(1,5));assert len(counts)==4 and set(counts.values())=={1}
    print('6 checks PASS: padding, zero real frame, labels, 3 four-epoch schedules; no model run')
if __name__=='__main__':selftest() if sys.argv[1]=='selftest' else verify()
