"""Independent saved arithmetic audit; no encoder, model forward or evaluation rerun."""
from pathlib import Path
from collections import Counter
import json, hashlib, datetime, itertools
import numpy as np

P=Path(__file__).resolve().parent; H=P.parent; GRAPH=H.parent; N=H/'null_candidate_learning'
F=[2,3,4,5,7]; FAMILIES=[3,4,5,7]; checks=[]; hashes={}
def rd(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    p=str(Path(p).resolve())
    if p not in hashes: hashes[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return hashes[p]
def ck(name,value,detail=None): checks.append(dict(name=name,ok=bool(value),detail=detail))
def pins(values,label):
    for p,h in values.items(): ck(label+':'+str(p),Path(p).is_file() and sha(p)==h)
def key(r): return r['seed'],r['clip_id'],r['kind']
def bucket(n): return 0 if n<=6 else 1 if n<=10 else 2 if n<=15 else 3 if n<=21 else 4
def orders(cards,ts):
    out=[]
    for i,j in itertools.combinations(range(len(cards)),2):
        if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]: continue
        aa=[c['chosen']['span'] if c['chosen'] else None for c in [cards[i],cards[j]]]
        if any(a is None for a in aa): state,detail,overlap='unobserved','unobserved',None
        else:
            a,b=[[int(ts[k]) for k in span] for span in aa]
            assert a[0]<a[1] and b[0]<b[1]
            overlap=max(0,min(a[1],b[1])-max(a[0],b[0]))
            if a[1]<=b[0]: state,detail='forward','forward_shared_boundary' if a[1]==b[0] else 'forward_separated'
            elif b[1]<=a[0]: state,detail='reverse','reverse_shared_boundary' if b[1]==a[0] else 'reverse_separated'
            else: state,detail='ambiguous_overlap','positive_duration_overlap'
        out.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state,detail=detail,overlap_duration_us=overlap))
    return out

def main():
    assert not(P/'VERIFICATION.json').exists()
    cfg=rd(P/'FROZEN.json'); tc=rd(P/'TRAINING_COMPLETE.json'); ef=rd(P/'evaluation/FROZEN.json')
    data=rd(P/'evaluation/COMPARISON.json'); prior=rd(N/'evaluation/COMPARISON.json')
    index=rd(N/'TRAINING_INPUT_INDEX.json'); rules=rd(N/'local_bag_evaluation/RULES_FROZEN.json'); audit=rd(P/'FIT_LABEL_AUDIT.json')
    pins(cfg['pins'],'training frozen'); pins(ef['pins'],'evaluation frozen')
    rows=index['rows']; masks=rules['partial_masks']; bags=[]
    ck('286 aligned training bags',len(rows)==len(masks)==286)
    fit=[i for i,r in enumerate(rows) if r['split']=='train']; probe=[i for i,r in enumerate(rows) if r['split']=='probe']
    ck('FIT and probe source disjoint',not {rows[i]['source_sha256'] for i in fit}&{rows[i]['source_sha256'] for i in probe})
    ck('FIT people 2..7 probe people 8/9',{rows[i]['person'] for i in fit}=={2,3,4,5,6,7} and {rows[i]['person'] for i in probe}=={8,9})
    ck('no untyped training families',all(r['family'] in FAMILIES for r in rows))
    for i,(r,m) in enumerate(zip(rows,masks)):
        ck('mask alignment:'+str(i),m['bag_index']==i and all(m[k]==r[k] for k in ['clip_id','family','split','view','reverse']))
        pins({r['path']:r['sha256']},'input bag'); z=np.load(r['path']); x=z['local_full']; spans=z['spans']; full=z['y']
        ck('bag shape and labels:'+str(i),x.shape==(len(spans),1160) and len(full)==len(spans) and full.dtype.kind=='b' and np.isfinite(x).all())
        partial=np.zeros(len(full),bool)
        for c in m['candidates']:
            k=c['candidate_index']; assert 0<=k<len(full); partial[k]=True
            ck('partial span:'+str((i,k)),c['transformed_span']==spans[k].tolist())
            a,b=map(int,spans[k]); original=[m['source_length']-1-b,m['source_length']-1-a] if r['reverse'] else [a,b]
            ck('reverse label mapping:'+str((i,k)),c['original_span']==original)
            valid=False
            for g in c['groups']:
                ws=set(g['all_group_witnesses']); observed={q for q in ws if original[0]<=q<=original[1]}
                valid |= g['outer_bounds'][0]<=original[0]<=original[1]<=g['outer_bounds'][1] and 2<=len(observed)<len(ws)
                ck('direct partial witness:'+str((i,k,g['group_id'])),observed==set(g['witnessed_frames']) and ws-observed==set(g['omitted_witnesses']))
            ck('partial has multiple direct witnesses:'+str((i,k)),valid)
        ck('partial and full disjoint:'+str(i),not (full&partial).any())
        hold=r['origin']=='reviewed_local_hold'; known=np.ones(len(full),bool) if hold else full|partial
        yy=np.zeros(len(full),int) if hold else (full|partial).astype(int)
        ck('hold never positive:'+str(i),not hold or not(full|partial).any())
        bags.append(dict(full=full,partial=partial,known=known,y=yy,buckets=np.array([bucket(b-a+1) for a,b in spans]),n=len(x)))
    pools=[]; common={}
    for f in FAMILIES:
        counts={}
        for label,b in itertools.product([0,1],range(5)):
            selected=[(i,np.flatnonzero(bags[i]['known']&(bags[i]['y']==label)&(bags[i]['buckets']==b))) for i in fit if rows[i]['family']==f]
            selected=[(i,k) for i,k in selected if len(k)]; counts[label,b]=len(selected)
            pools.append(dict(family=f,label=label,duration_bucket=b,bags=len(selected),candidates=sum(len(k) for _,k in selected),people=sorted({rows[i]['person'] for i,_ in selected})))
        common[str(f)]=[b for b in range(5) if counts[0,b] and counts[1,b]]
    ck('FIT pool audit exact',pools==audit['rows']); ck('matched duration buckets exact',common==audit['matched_buckets'] and all(common.values()))
    ck('fixed seeds and steps',cfg['seeds']==[42,43,44] and cfg['steps']==1200 and {r['seed'] for r in tc['rows']}=={42,43,44})
    bag_counts=Counter(); bag_out=[]
    for r in tc['rows']:
        pins({r[k]:r[k+'_sha256'] for k in ['model','predictions']},'trained output'); pins({r['initial_model']:r['initial_sha256']},'original ranker')
        ck('final checkpoint:'+str(r['seed']),r['log'][-1]['step']==1200 and all(np.isfinite(v['loss']) for v in r['log']))
        ar=np.load(r['predictions']); ck('saved prediction count:'+str(r['seed']),len(ar.files)==len(rows))
        for i,row in enumerate(rows):
            lo=ar[f'logits_{i}']; d=bags[i]; ck('bag logits:'+str((r['seed'],i)),lo.shape==(d['n'],5) and np.isfinite(lo).all())
            support=lo[:,F.index(row['family'])]>=0
            for name,mask in [('full',d['full']),('partial',d['partial']),('reviewed_hold',d['known']&(d['y']==0))]:
                if mask.any(): bag_out.append(dict(seed=r['seed'],bag=i,split=row['split'],family=row['family'],view=row['view'],reverse=row['reverse'],label_type=name,candidates=int(mask.sum()),positive=int((support&mask).sum()),scope='raw saved logits, before body admission gate; not independent test'))
    old={key(r):r for r in prior['rows']}; new={key(r):r for r in data['rows']}; pp={key(r):r for r in data['predictions']}; op={key(r):r for r in prior['predictions']}
    ck('246 scenarios with three seeds',len(new)==len(data['rows'])==len(pp)==246 and set(new)==set(old)==set(pp)==set(op))
    ck('82 per seed',all(sum(k[0]==s for k in new)==82 for s in [42,43,44]))
    sources={}; features=set(); family2=0; diagnostic=[]; changed=[]
    for kk,row in new.items():
        pr=pp[kk]; oldpr=op[kk]; before=old[kk]; pins({pr['path']:pr['sha256']},'new saved scores')
        ck('source and feature same:'+str(kk),all(pr[k]==oldpr[k] for k in ['source','source_sha256','feature_path','feature_sha256']))
        if pr['source'] not in sources:
            pins({pr['source']:pr['source_sha256']},'saved public input'); sources[pr['source']]=np.load(pr['source'])['ts']
        if pr['feature_path'] not in features: pins({pr['feature_path']:pr['feature_sha256']},'cached features'); features.add(pr['feature_path'])
        ts=sources[pr['source']]; ck('PTS strict integer:'+str(kk),np.isfinite(ts).all() and (np.diff(ts)>0).all() and np.array_equal(ts,ts.astype(np.int64)))
        ar=np.load(pr['path']); oa=np.load(oldpr['path']); cc=row['conditions']; ck('condition set:'+str(kk),set(cc)==set(ef['conditions']))
        for mode in ['baseline','local_full']:
            ck('original cards exact:'+str((kk,mode)),cc[mode]['cards']==before['conditions'][mode]['cards'])
            ck('raw order exact:'+str((kk,mode)),cc[mode]['original_order']==before['conditions'][mode]['order'])
            ck('same timestamp arithmetic:'+str((kk,mode)),cc[mode]['order']==orders(cc[mode]['cards'],ts))
        priorcards=before['conditions']['local_full']['cards']; now=cc['progression_separate']; ck('card count:'+str(kk),len(priorcards)==len(now['cards']))
        expected_diagnostics=[]
        for i,(c,oc) in enumerate(zip(now['cards'],priorcards)):
            if c['family']==2:
                ck('family2 exact:'+str((kk,i)),c==oc); family2+=1; continue
            spans=ar[f'spans_{i}']; presence=ar[f'presence_{i}']; logits=ar[f'progression_logits_{i}']
            ck('candidate span and presence invariant:'+str((kk,i)),np.array_equal(spans,oa[f'spans_{i}']) and np.array_equal(presence,oa[f'presence_{i}']))
            ck('finite score shape:'+str((kk,i)),len(spans)==len(presence)==len(logits) and np.isfinite(logits).all())
            eligible=presence>=0; support=eligible&(logits>=0); ci=None
            if oc['chosen'] is not None:
                indices=np.flatnonzero((spans==np.array(oc['chosen']['span'])).all(1)); assert len(indices)==1; ci=int(indices[0]); ck('original admission:'+str((kk,i)),eligible[ci])
            selected=bool(ci is not None and support[ci])
            state='no_body_gate_admission' if not eligible.any() else 'no_progression_support' if not support.any() else 'original_extent_unavailable' if ci is None else 'selected_extent_supported' if selected else 'progression_observed_extent_uncertain'
            ck('explicit evidence state:'+str((kk,i)),c['progression_state']==state and c['progression_observed']==bool(support.any()))
            ck('raw extent unchanged:'+str((kk,i)),c['original_extent']==oc['chosen'])
            ck('no new extent selected:'+str((kk,i)),c['chosen']==(oc['chosen'] if selected else None))
            excluded={'chosen','truth','original_extent','progression_state','progression_observed'}
            ck('all other card fields unchanged:'+str((kk,i)),{k:v for k,v in c.items() if k not in excluded}=={k:v for k,v in oc.items() if k not in excluded})
            ck('not semantic completion:'+str((kk,i)),c['semantic_completion'] is None)
            if 'truth' in c:
                t=c['truth']; ot=oc['truth']; seen=set(oc['chosen']['body_observed']) if selected else set()
                ck('truth labels unchanged:'+str((kk,i)),{k:v for k,v in t.items() if k not in ['all_witness','wrong_points']}=={k:v for k,v in ot.items() if k not in ['all_witness','wrong_points']})
                ck('truth arithmetic:'+str((kk,i)),t['all_witness']==(set(t['witness'])<=seen) and t['wrong_points']==len(set(t['wrong_witness'])&seen))
            dd=dict(card=c['card'],family=c['family'],state=state,eligible_candidates=int(eligible.sum()),progression_candidates=int(support.sum()),original_index=ci,original_extent_supported=selected,original_logit=float(logits[ci]) if ci is not None else None)
            expected_diagnostics.append(dd); diagnostic.append(dict(seed=kk[0],clip_id=kk[1],kind=kk[2],**dd))
            if c['chosen']!=oc['chosen']: changed.append(dict(seed=kk[0],clip_id=kk[1],kind=kk[2],card=c['card'],old_span=oc['chosen']['span'] if oc['chosen'] else None,state=state))
        ck('diagnostic arithmetic:'+str(kk),now['diagnostics']==expected_diagnostics)
        ck('supported order and raw order:'+str(kk),now['order']==orders(now['cards'],ts) and now['original_order']==cc['local_full']['order'])
    ck('82 cached source scenarios',len(sources)==len(features)==82); ck('family2 72 cards',family2==72)
    summary=[]
    for seed,mode,kind in itertools.product([42,43,44],ef['conditions'],sorted({r['kind'] for r in data['rows']})):
        rr=[r for r in data['rows'] if r['seed']==seed and r['kind']==kind]; cards=[c for r in rr for c in r['conditions'][mode]['cards']]; truth=[c['truth'] for c in cards if 'truth' in c]
        summary.append(dict(seed=seed,condition=mode,kind=kind,cases=len(rr),cards=len(cards),unsupported_extent=sum(c['chosen'] is None for c in cards),known_groups=len(truth),all_witness=sum(t['all_witness'] for t in truth),wrong_points=sum(t['wrong_points'] for t in truth),order=dict(Counter(o['state'] for r in rr for o in r['conditions'][mode]['order'])),progression_states=dict(Counter(c['progression_state'] for c in cards if 'progression_state' in c))))
    sk=lambda r:(r['seed'],r['condition'],r['kind']); ck('all summary denominators',sorted(summary,key=sk)==sorted(data['summary'],key=sk))
    ck('timing all rows and scope',len(data['timing'])==246 and {key(r) for r in data['timing']}==set(new) and all(r['seconds']>=0 and 'not total web latency' in r['scope'] for r in data['timing']))
    protections=rd(GRAPH/'PROTECTIONS.json'); pins(protections,'protected'); ck('477 protected files',len(protections)==477)
    ck('frozen core and ranker',tc['common100_changed'] is False and tc['ranker_changed'] is False and tc['web_changed'] is False)
    for obj,name in [(cfg,'training'),(ef,'evaluation')]:
        for k in ['private_access','final8','web_changed']: ck(name+k,obj[k] is False)
    normal=[r for r in diagnostic if r['kind'] in ['full_normal','speed075','speed125']]
    result=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='PASS' if all(c['ok'] for c in checks) else 'FAIL',checks=len(checks),failed=[c for c in checks if not c['ok']],rows=len(new),input_bags=len(rows),family2_preserved=family2,protected_files=len(protections),progression_states=dict(Counter(r['state'] for r in diagnostic)),normal_speed_states=dict(Counter(r['state'] for r in normal)),changed_extents=changed,summary=summary,diagnostics=diagnostic,bag_raw_logit_outcomes=bag_out,no_inference=True,no_training=True,web_changed=False,adopted=False,full_G2_pass=False,script_sha256=sha(__file__),reviewer='Separate Codex motion_contract_review_1113 arithmetic/code audit; not independent human semantic truth',limitations=['Progression support does not certify direction, correct extent, completion, or omission.','Probe08/09 and DEV11..13 were previously exposed; not new final evaluation.','Stored scores and label arithmetic checked; no model forward or independent RGB truth review.','Local training bag counts are before body-gate admission; use separately reviewed local evaluation for generalization.'])
    (P/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ['status','checks','failed','rows','input_bags','family2_preserved','protected_files','progression_states','normal_speed_states']},ensure_ascii=False))
    if result['failed']: raise SystemExit(1)

if __name__=='__main__': main()
