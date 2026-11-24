"""Isolated, masked guide-episode membership continuation. Public sources only."""
from pathlib import Path
import sys,json,hashlib,time,traceback,collections
import numpy as np,torch
from torch import nn
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
from prepare_linker import load
from occurrence_binding import union_indices
FAMILIES=[2,3,4,5,7]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Membership(nn.Module):
    def __init__(self,seed):
        super().__init__();self.role=nn.Sequential(nn.Linear(684,96),nn.GELU(),nn.Linear(96,11))
        ck=torch.load(O/'word100_candidate_role_learning_20261111/runs'/str(seed)/'model.pt',map_location='cpu',weights_only=True)
        self.load_state_dict({k:v for k,v in ck.items() if k.startswith('role.')})
    def forward(self,x):
        logits=self.role(x)
        return logits[:,FAMILIES]-logits[:,0,None]

def feature(z,a,b,condition):
    if condition=='mean':return z[a:b+1].mean(0).astype(np.float32)
    indices=np.arange(a,b+1);bins=np.array_split(indices,3)
    if any(len(g)==0 for g in bins):bins=[indices[:1],indices,indices[-1:]]
    v=z[:,228:456]
    return np.concatenate([v[g].mean(0) for g in bins]).astype(np.float32)

def freeze():
    assert not (W/'FROZEN.json').exists();meta=read(W/'INPUTS_COMPLETE.json');checks=[]
    for p,h in meta['pins'].items():assert sha(p)==h
    train=set();probe=set();counts=collections.Counter()
    for r in meta['rows']:
        assert sha(r['path'])==r['sha256'] and sha(r['cache'])==r['cache_sha256']
        a,b=r['span'];cache=np.load(r['cache'])['z'];v=np.load(r['path'])
        for condition,key in [('mean','mean'),('ordered','local')]:assert np.allclose(v[key],feature(cache,a,b,condition),atol=1e-6)
        assert np.isfinite(v['mean']).all() and np.isfinite(v['local']).all()
        expected=np.zeros(5);mask=np.zeros(5)
        for t in r['targets']:
            j=FAMILIES.index(t['family']);expected[j]=t['value'];mask[j]=1;counts[r['split'],t['family'],t['value']]+=1
        assert np.array_equal(v['y'],expected) and np.array_equal(v['mask'],mask)
        (probe if r['split']=='probe' else train).add(r['source_sha256'])
        checks.append(dict(path=r['path'],passed=True))
    assert not(train&probe) and len(train)==8 and len(probe)==4
    assert all(counts['train',f,y]>0 for f in FAMILIES for y in [0,1])
    write(W/'DATA_VERIFICATION.json',dict(at=stamp(),status='PASS_ROOT_SOURCE_ARRAY_AND_MASK_AUDIT',rows=len(checks),independent_human=False,review='Root source/event/guide linkage plus second feature reconstruction; agent preflight unavailable due usage limit.',checks=checks,source_split={'train':8,'probe':4},training_ready=True,risks=meta['limits']))
    paths=[Path(__file__),W/'prepare.py',W/'INPUTS_COMPLETE.json',W/'DATA_VERIFICATION.json',U/'context_role_training/RESULTS.json',U/'interval_pipeline.py',U/'occurrence_binding.py',O/'word100_guide_requirement_learning_20261111/requirement_core.py',O/'word100_guide_requirement_learning_20261111/order_binding.py']
    paths += [O/'word100_candidate_role_learning_20261111/runs'/str(s)/'model.pt' for s in [42,43,44]]
    paths += [O/'word100_guide_requirement_learning_20261111/order_runs'/str(s)/'model.pt' for s in [42,43,44]]
    write(W/'FROZEN.json',dict(at=stamp(),pins={str(p):sha(p) for p in paths},conditions=['mean','ordered'],seeds=[42,43,44],epochs=200,batch=8,lr=.0003,weight_decay=.01,gradient_clip=2,loss='Masked per-family BCE; train-only negative/positive weighting. 1200 optimizer steps for48 training windows, comparable order of magnitude to previous1800 updates. Not chosen from DEV.',initialization='Matching-seed existing role MLP only. Common100/TCN/GRU/shape/direction/order source models untouched.',meaning='Observed source guide-action membership in a window, not completion, correctness or calibrated user confidence.',selector='Dynamic parent groups only. Keep existing eligible phase+role candidate and require membership>=0.5; rank old joint-score times membership. Static and endpoint-child cards unchanged. No expected order or source indices in selection.',gates=['normal/speed source spans and observed indices identical to baseline for all seeds, preserving existing component evidence','correctly mapped dynamic reverse pairs increase in every seed; not merely more unknowns','all-hand-missing does not become support','order scores use same frozen model; only new query observations computed, old scores reused','no checkpoint/threshold/epoch selected from DEV'],scope='Already-exposed development only. Reserved final REAL17/18 not touched. No web adoption from this file alone.'))
    print('Frozen membership experiment',len(meta['rows']),'windows',dict(counts))

def train():
    cfg=read(W/'FROZEN.json');assert not (W/'TRAINING_COMPLETE.json').exists()
    for p,h in cfg['pins'].items():assert sha(p)==h
    torch.set_num_threads(4);torch.set_num_interop_threads(1);rows=read(W/'INPUTS_COMPLETE.json')['rows'];all_data=[]
    for r in rows:
        assert sha(r['path'])==r['sha256'];z=np.load(r['path']);all_data.append(dict(r,**{k:z[k].copy() for k in z.files}))
    fit=[r for r in all_data if r['split']=='train'];probe=[r for r in all_data if r['split']=='probe']
    y=np.stack([r['y'] for r in fit]);mask=np.stack([r['mask'] for r in fit]);pos=(y*mask).sum(0);neg=((1-y)*mask).sum(0);weight=torch.tensor(neg/pos,dtype=torch.float32);done=[]
    for condition in cfg['conditions']:
        key='mean' if condition=='mean' else 'local';x=torch.tensor(np.stack([r[key] for r in fit]));ty=torch.tensor(y);tm=torch.tensor(mask)
        for seed in cfg['seeds']:
            dst=W/'runs'/condition/str(seed);assert not (dst/'TRAINING.json').exists()
            torch.manual_seed(seed);net=Membership(seed);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();steps=0;history=[]
            for epoch in range(cfg['epochs']):
                losses=[];order=np.random.default_rng(seed*1000+epoch).permutation(len(fit))
                for j in range(0,len(order),cfg['batch']):
                    ii=order[j:j+cfg['batch']];logit=net(x[ii]);raw=torch.nn.functional.binary_cross_entropy_with_logits(logit,ty[ii],pos_weight=weight,reduction='none');loss=(raw*tm[ii]).sum()/tm[ii].sum()
                    assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']);opt.step();steps+=1;losses.append(float(loss.detach()))
                history.append(float(np.mean(losses)))
            dst.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),dst/'model.pt');net.eval()
            with torch.inference_mode():prob=net(torch.tensor(np.stack([r[key] for r in probe]))).sigmoid().numpy()
            pp=[dict(clip_id=r['clip_id'],span=r['span'],targets=r['targets'],probability=p.tolist()) for r,p in zip(probe,prob)]
            write(dst/'PROBE.json',dict(rows=pp,scope='REAL07 continuation-excluded only; exposed to older encoders/heads. Not fresh generalization.'))
            rec=dict(condition=condition,seed=seed,path=str(dst/'model.pt'),sha256=sha(dst/'model.pt'),seconds=time.perf_counter()-start,steps=steps,epochs=cfg['epochs'],train_rows=len(fit),probe_rows=len(probe),history=history,pos_weight=weight.tolist(),selection='last epoch fixed')
            write(dst/'TRAINING.json',rec);done.append(rec);print(condition,seed,'trained',round(rec['seconds'],2),'s',flush=True)
    write(W/'TRAINING_COMPLETE.json',dict(runs=done,web_changed=False,private=False))

def evaluate():
    cfg=read(W/'FROZEN.json');assert not (W/'EVALUATION_COMPLETE.json').exists()
    for p,h in cfg['pins'].items():assert sha(p)==h
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    role=load('_membership_role',O/'word100_candidate_role_learning_20261111/role_core.py');rc=load('_membership_requirements',O/'word100_guide_requirement_learning_20261111/requirement_core.py');sys.modules['requirement_core']=rc
    om=load('_membership_order',O/'word100_guide_requirement_learning_20261111/order_binding.py')
    baseline=[r for r in read(U/'context_role_training/RESULTS.json') if r['condition']=='source_context_control'];sources=read(O/'word100_candidate_role_learning_20261111/EVALUATION_INPUTS.json')['rows']+[dict(r,kind='dynamic_complete_blocks') for r in read(U/'dynamic_blocks/INPUTS_COMPLETE.json')['rows']]
    src={(r['clip_id'],r['kind']):r for r in sources};assert len(src)==82
    records=[];reused=0;new_order=0;memo={}
    for r in baseline:
        word=r['clip_id'].split('_')[2]
        for p in r['pairs']:
            if p['order_probability'] is None or any(s is None for s in p['spans']):continue
            lo,hi=sorted(int(round(sum(s)/2)) for s in p['spans']);ga,gb=[next(c['groups'][0][0] for c in rc.CARDS[word] if c['id']==i) for i in p['cards']]
            memo[r['clip_id'],r['kind'],r['seed'],lo,hi,ga,gb]=p['order_probability']
    for run in read(W/'TRAINING_COMPLETE.json')['runs']:
        assert sha(run['path'])==run['sha256'];condition,seed=run['condition'],run['seed'];net=Membership(seed);net.load_state_dict(torch.load(run['path'],weights_only=True,map_location='cpu'));net.eval();order=om.OrderedBinding();order.load_state_dict(torch.load(O/'word100_guide_requirement_learning_20261111/order_runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));order.eval()
        for b in [r for r in baseline if r['seed']==seed]:
            s=src[b['clip_id'],b['kind']];assert sha(s['path'])==s['sha256'];z=np.load(s['path']);word=b['clip_id'].split('_')[2];cards=json.loads(json.dumps(b['cards']));preds=[]
            for card in cards:
                if card['parent'] or card['family'] not in FAMILIES:continue
                j=FAMILIES.index(card['family'])
                for group in card['groups']:
                    viable=[]
                    for c in group['candidates']:
                        a,bb=c['span'];xx=feature(z['z'],a,bb,condition)
                        with torch.inference_mode():p=float(net(torch.tensor(xx[None])).sigmoid()[0,j])
                        c['membership']=p;preds.append(dict(card=card['id'],span=c['span'],probability=p))
                        # Candidate eligibility and role-supported anchor list
                        # are inherited, not replaced by model confidence.
                        if c['eligible'] and c['anchor_indices'] and p>=.5:viable.append(c)
                    winner=max(viable,key=lambda c:c['score']*c['membership']) if viable else None
                    group['span']=winner['span'] if winner else None;group['anchors']=winner['anchor_indices'] if winner else []
                spans=[g['span'] for g in card['groups'] if g['span']]
                card.update(span=[min(a for a,bb in spans),max(bb for a,bb in spans)] if len(spans)==len(card['groups']) else None,evidence_intervals=spans,observed_indices=union_indices(spans))
            main=[c for c in cards if c['required'] and not c['parent']];pairs=[]
            for a,bb in zip(main,main[1:]):
                if a['family']==bb['family']:continue
                aa,bs=a['span'],bb['span'];prob=None;state='association_unknown'
                if aa is not None and bs is not None:
                    lo,hi=sorted(int(round(sum(sp)/2)) for sp in [aa,bs]);ga,gb=[next(c['groups'][0][0] for c in rc.CARDS[word] if c['id']==i) for i in [a['id'],bb['id']]];key=b['clip_id'],b['kind'],seed,lo,hi,ga,gb
                    if key not in memo:
                        with torch.inference_mode():memo[key]=float(order(torch.tensor(rc.observation(z['fh'],lo,hi)[None]),torch.tensor(rc.requirement('order',guides=[ga,gb])[None])).sigmoid()[0]);new_order+=1
                    else:reused+=1
                    prob=memo[key];state='supported' if aa[1]<bs[0] and prob>=.65 else 'reversed' if bs[1]<aa[0] and prob<=.35 else 'ambiguous'
                pair=dict(cards=[a['id'],bb['id']],spans=[aa,bs],state=state,order_probability=prob,semantic_completion=None)
                if b['kind']=='dynamic_complete_blocks':
                    seams=[v['query_right'] for v in s['target']['splice_joins'] if v['synthetic_discontinuity']];pair['selected_span_crosses_splice']=any(sp[0]<j<=sp[1] for sp in [aa,bs] if sp for j in seams)
                    pair['limited_dynamic_target']=word=='WORD0739' and pair['cards']==['close_chest','raise_forearm'];pair['spans_within_target_occurrences']=None
                    if pair['limited_dynamic_target'] and aa and bs:
                        ba,ca=[s['target']['blocks'][k]['query_frame_span'] for k in ['B','C']];pair['spans_within_target_occurrences']=bool(ba[0]<=aa[0]<=aa[1]<=ba[1] and ca[0]<=bs[0]<=bs[1]<=ca[1])
                pairs.append(pair)
            records.append(dict(condition=condition,seed=seed,clip_id=b['clip_id'],kind=b['kind'],cards=cards,pairs=pairs,membership_predictions=preds))
        print(condition,seed,'evaluated',flush=True)
    write(W/'RESULTS.json',records);summaries=[]
    for condition in cfg['conditions']:
        for seed in cfg['seeds']:
            rr=[r for r in records if r['condition']==condition and r['seed']==seed];base={(r['clip_id'],r['kind']):r for r in baseline if r['seed']==seed};changed=[]
            for r in rr:
                if r['kind'] not in ['full_normal','speed075','speed125']:continue
                old=base[r['clip_id'],r['kind']]
                for c,o in zip(r['cards'],old['cards']):
                    if c['span']!=o['span'] or c['observed_indices']!=o['observed_indices']:changed.append(dict(clip_id=r['clip_id'],kind=r['kind'],card=c['id'],before=o['span'],after=c['span']))
            pairs=[p for r in rr if r['kind']=='dynamic_complete_blocks' for p in r['pairs'] if p.get('limited_dynamic_target')]
            summaries.append(dict(condition=condition,seed=seed,normal_speed_changed=changed,limited_dynamic=dict(correct=sum(p['state']=='reversed' and p['spans_within_target_occurrences'] is True and not p['selected_span_crosses_splice'] for p in pairs),total=len(pairs)),states={k:dict(collections.Counter(p['state'] for r in rr if r['kind']==k for p in r['pairs'])) for k in sorted({r['kind'] for r in rr})}))
    write(W/'EVALUATION_COMPLETE.json',dict(summary=summaries,old_order_scores_reused=reused,new_order_observations=new_order,web_changed=False,reserved_final_unread=True));print(json.dumps(summaries,ensure_ascii=False))

if __name__=='__main__':
    try:{'freeze':freeze,'train':train,'evaluate':evaluate}[sys.argv[1]]()
    except BaseException:
        (W/f'FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
