"""Audit saved paired-static preparation/training/evaluation; no model forwards."""
from pathlib import Path
from collections import Counter
import importlib.util,json,hashlib,datetime,itertools
import numpy as np
S=Path(__file__).resolve().parent; H=S.parent; P=H/'progression_presence_learning'; N=H/'null_candidate_learning'; GRAPH=H.parent
spec=importlib.util.spec_from_file_location('_separate_saved_arithmetic',P/'verify.py'); audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
rd,sha,ck,pins,key,orders=audit.rd,audit.sha,audit.ck,audit.pins,audit.key,audit.orders
F=[2,3,4,5,7]; AX={2:0,3:1,4:2,5:4,7:3}; FAMILIES=[3,4,5,7]

def main():
    assert not(S/'VERIFICATION.json').exists()
    cfg=rd(S/'FROZEN.json'); prep=rd(S/'PREPARATION_CONTRACT.json'); meta=rd(S/'INPUTS_COMPLETE.json'); tc=rd(S/'TRAINING_COMPLETE.json'); labels=rd(S/'FIT_LABEL_AUDIT.json')
    index=rd(N/'TRAINING_INPUT_INDEX.json'); rows=index['rows']; masks=rd(N/'local_bag_evaluation/RULES_FROZEN.json')['partial_masks']; previous=rd(P/'TRAINING_COMPLETE.json')
    pins(prep['pins'],'preparation frozen');pins(meta['pins'],'public cached source');pins(cfg['pins'],'training frozen')
    ck('286 source rows and masks',len(rows)==len(masks)==286)
    bags=[]
    for i,(r,m) in enumerate(zip(rows,masks)):
        pins({r['path']:r['sha256']},'original features');z=np.load(r['path']);full=z['y'];partial=np.zeros(len(full),bool)
        for c in m['candidates']:partial[c['candidate_index']]=True
        ck('mask same source:'+str(i),m['bag_index']==i and all(r[k]==m[k] for k in ['clip_id','split','family','view','reverse']))
        hold=r['origin']=='reviewed_local_hold';known=np.ones(len(full),bool) if hold else full|partial
        yy=np.zeros(len(full),bool) if hold else full|partial
        bags.append(dict(n=len(full),full=full,partial=partial,known=known,y=yy,spans=z['spans']))
        ck('known supervision disjoint:'+str(i),not(full&partial).any() and (not hold or not(full|partial).any()))
    expected=[i for i,r in enumerate(rows) if r['origin']=='positive_source_extent']
    ck('190 all positive-source bags',len(meta['bags'])==190 and [r['bag_index'] for r in meta['bags']]==expected)
    fit=[r for r in rows if r['split']=='train'];probe=[r for r in rows if r['split']=='probe']
    ck('FIT/probe identity split',{r['person'] for r in fit}=={2,3,4,5,6,7} and {r['person'] for r in probe}=={8,9} and not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in probe})
    static={};eligibility=[];max_relation_variation=0.;max_relation_std=0.
    for r in meta['bags']:
        i=r['bag_index'];d=bags[i];original=rows[i];pins({r['path']:r['sha256']},'paired static feature')
        ck('static bag metadata:'+str(i),all(r[k]==original[k] for k in ['clip_id','person','split','family','view','reverse','source_sha256']))
        z=np.load(r['path']);x=z['x'];mapping=z['mapping'];known=np.flatnonzero(d['known']);expected_map=[]
        for k in known:
            a,b=map(int,d['spans'][k]);expected_map.extend([[int(k),j,t] for j,t in enumerate([a,(a+b)//2,b])])
        ck('every positive exactly three anchors:'+str(i),np.array_equal(z['positive_indices'],known) and np.array_equal(mapping,np.asarray(expected_map,np.int32)))
        ck('static finite feature count:'+str(i),x.shape==(len(known)*3,1160) and np.isfinite(x).all() and r['static_candidates']==len(x) and r['positive_candidates']==len(known))
        ck('first original feature reproduction:'+str(i),r['original_feature_max_error']<1e-5)
        # In each 290-d block, indices64:290 are explicit relative pose/masks/hands.
        # Constant raw observation should stay constant across three pooled thirds;
        # the encoder64 may vary because it represents position/time too.
        parts=x.reshape(-1,4,290);explicit=parts[:,:3,64:];variation=float(np.max(np.abs(explicit-explicit[:,:1])))
        spread=float(np.max(np.abs(parts[:,3,64:])))
        max_relation_variation=max(max_relation_variation,variation);max_relation_std=max(max_relation_std,spread)
        ck('static explicit pose thirds agree:'+str(i),variation<1e-4,variation)
        ck('static explicit pose temporal std zero:'+str(i),spread<1e-4,spread)
        coverage=x[:,64+AX[r['family']]*2*10+8];ck('binary body observability:'+str(i),np.isin(coverage,[0,1]).all());eligible=coverage>.5
        ck('positive has at least one visible anchor:'+str(i),all(eligible[mapping[:,0]==k].any() for k in known))
        eligibility.append(dict(bag_index=i,clip_id=r['clip_id'],family=r['family'],split=r['split'],view=r['view'],reverse=r['reverse'],generated=len(eligible),eligible=int(eligible.sum()),excluded_unobserved=int((~eligible).sum())))
        static[i]=dict(mapping=mapping,eligible=eligible,n=len(x))
    ck('anchor eligibility audit exact',eligibility==labels['static_anchor_eligibility'])
    ck('same matched duration scope',labels['matched_buckets']==rd(P/'FIT_LABEL_AUDIT.json')['matched_buckets'])
    ck('three fixed final steps',cfg['seeds']==[42,43,44] and cfg['steps']==1200 and {r['seed'] for r in tc['rows']}=={42,43,44})
    priorruns={r['seed']:r for r in previous['rows']};static_out=[];real_out=[]
    for run in tc['rows']:
        for k in ['model','predictions','static_predictions']:pins({run[k]:run[k+'_sha256']},'saved training output')
        priorrun=priorruns[run['seed']];ck('initial P model exact:'+str(run['seed']),run['initial_model']==priorrun['model'] and run['initial_sha256']==priorrun['model_sha256'])
        pins({run['initial_model']:run['initial_sha256']},'previous frozen progression model')
        ck('final1200 training log:'+str(run['seed']),run['log'][-1]['step']==1200 and all(np.isfinite(r['loss']) for r in run['log']))
        ar=np.load(run['predictions']);st=np.load(run['static_predictions']);ck('all input prediction keys:'+str(run['seed']),set(ar.files)=={f'logits_{i}' for i in range(len(rows))} and set(st.files)=={f'logits_{i}' for i in static})
        for i,r in enumerate(rows):
            scores=ar[f'logits_{i}'];d=bags[i];ck('real saved score shape:'+str((run['seed'],i)),scores.shape==(d['n'],5) and np.isfinite(scores).all());positive=scores[:,F.index(r['family'])]>=0
            for name,mask in [('full',d['full']),('partial',d['partial']),('hold',d['known']&~d['y'])]:
                if mask.any():real_out.append(dict(seed=run['seed'],bag=i,split=r['split'],family=r['family'],label=name,candidates=int(mask.sum()),positive=int((positive&mask).sum()),scope='ungated saved logits; overlapping candidates, not independent clips'))
            if i in static:
                s=static[i];values=st[f'logits_{i}'];ck('static score shape:'+str((run['seed'],i)),values.shape==(s['n'],5) and np.isfinite(values).all());pred=values[:,F.index(r['family'])]>=0
                for j,name in enumerate(['start','middle','end']):
                    mask=s['eligible']&(s['mapping'][:,1]==j)
                    static_out.append(dict(seed=run['seed'],bag=i,split=r['split'],family=r['family'],anchor=name,eligible=int(mask.sum()),false_progression=int((pred&mask).sum())))
    ef=rd(S/'evaluation/FROZEN.json');result=rd(S/'evaluation/COMPARISON.json');prior=rd(N/'evaluation/COMPARISON.json')
    pins(ef['pins'],'full evaluation frozen');old={key(r):r for r in prior['rows']};new={key(r):r for r in result['rows']};npred={key(r):r for r in result['predictions']};opred={key(r):r for r in prior['predictions']}
    ck('246 saved rows aligned',len(new)==len(result['rows'])==len(npred)==246 and set(new)==set(old)==set(npred))
    mode=[x for x in ef['conditions'] if x not in ['baseline','local_full']];assert len(mode)==1;mode=mode[0]
    ck('82 per seed',all(sum(k[0]==s for k in new)==82 for s in [42,43,44]))
    source_cache={};feature_cache=set();diags=[];f2=0;changes=[]
    for kk,row in new.items():
        pr=npred[kk];oldpr=opred[kk];pins({pr['path']:pr['sha256']},'full prediction')
        ck('same source and features:'+str(kk),all(pr[k]==oldpr[k] for k in ['source','source_sha256','feature_path','feature_sha256']))
        if pr['source'] not in source_cache:pins({pr['source']:pr['source_sha256']},'source');source_cache[pr['source']]=np.load(pr['source'])['ts']
        if pr['feature_path'] not in feature_cache:pins({pr['feature_path']:pr['feature_sha256']},'feature');feature_cache.add(pr['feature_path'])
        ts=source_cache[pr['source']];ck('strict sampled timestamps:'+str(kk),np.isfinite(ts).all() and (np.diff(ts)>0).all() and np.array_equal(ts,ts.astype(np.int64)))
        cc=row['conditions'];before=old[kk]['conditions'];ck('same three conditions:'+str(kk),set(cc)==set(ef['conditions']))
        for name in ['baseline','local_full']:
            ck('old cards preserved:'+str((kk,name)),cc[name]['cards']==before[name]['cards'])
            ck('old raw order preserved:'+str((kk,name)),cc[name]['original_order']==before[name]['order'])
            ck('timestamp order:'+str((kk,name)),cc[name]['order']==orders(cc[name]['cards'],ts))
        ar=np.load(pr['path']);oa=np.load(oldpr['path']);current=cc[mode];priorcards=before['local_full']['cards'];expected_diags=[]
        ck('same card count:'+str(kk),len(current['cards'])==len(priorcards))
        for i,(c,oc) in enumerate(zip(current['cards'],priorcards)):
            if c['family']==2:ck('family2 preserved:'+str((kk,i)),c==oc);f2+=1;continue
            spans=ar[f'spans_{i}'];presence=ar[f'presence_{i}'];lo=ar[f'progression_logits_{i}'];eligible=presence>=0;support=eligible&(lo>=0)
            ck('same spans and presence gate:'+str((kk,i)),np.array_equal(spans,oa[f'spans_{i}']) and np.array_equal(presence,oa[f'presence_{i}']))
            ck('score finite shapes:'+str((kk,i)),len(lo)==len(spans)==len(presence) and np.isfinite(lo).all())
            ci=None
            if oc['chosen']:
                indices=np.flatnonzero((spans==np.asarray(oc['chosen']['span'])).all(1));assert len(indices)==1;ci=int(indices[0]);ck('old gate admission:'+str((kk,i)),eligible[ci])
            selected=bool(ci is not None and support[ci]);state='no_body_gate_admission' if not eligible.any() else 'no_progression_support' if not support.any() else 'original_extent_unavailable' if ci is None else 'selected_extent_supported' if selected else 'progression_observed_extent_uncertain'
            ck('state arithmetic:'+str((kk,i)),c['progression_state']==state and c['progression_observed']==bool(support.any()))
            ck('raw extent and no replacement:'+str((kk,i)),c['original_extent']==oc['chosen'] and c['chosen']==(oc['chosen'] if selected else None))
            ignored={'truth','chosen','original_extent','progression_state','progression_observed'}
            ck('other fields unchanged:'+str((kk,i)),{k:v for k,v in c.items() if k not in ignored}=={k:v for k,v in oc.items() if k not in ignored})
            ck('no semantic completion:'+str((kk,i)),c['semantic_completion'] is None)
            if 'truth' in c:
                t=c['truth'];ot=oc['truth'];obs=set(c['chosen']['body_observed']) if c['chosen'] else set()
                ck('truth labels invariant:'+str((kk,i)),{k:v for k,v in t.items() if k not in ['all_witness','wrong_points']}=={k:v for k,v in ot.items() if k not in ['all_witness','wrong_points']})
                ck('truth witness arithmetic:'+str((kk,i)),t['all_witness']==(set(t['witness'])<=obs) and t['wrong_points']==len(set(t['wrong_witness'])&obs))
            dd=dict(card=c['card'],family=c['family'],state=state,eligible_candidates=int(eligible.sum()),progression_candidates=int(support.sum()),original_index=ci,original_extent_supported=selected,original_logit=float(lo[ci]) if ci is not None else None)
            expected_diags.append(dd);diags.append(dict(seed=kk[0],clip_id=kk[1],kind=kk[2],**dd))
            if c['chosen']!=oc['chosen']:changes.append(dict(seed=kk[0],clip_id=kk[1],kind=kk[2],card=c['card'],old_span=oc['chosen']['span'] if oc['chosen'] else None,state=state))
        ck('diagnostics exact:'+str(kk),current['diagnostics']==expected_diags)
        ck('supported order and raw old order:'+str(kk),current['order']==orders(current['cards'],ts) and current['original_order']==cc['local_full']['order'])
    ck('family2 72 and source82',f2==72 and len(source_cache)==len(feature_cache)==82)
    summaries=[]
    for seed,name,kind in itertools.product([42,43,44],ef['conditions'],sorted({r['kind'] for r in result['rows']})):
        rr=[r for r in result['rows'] if r['seed']==seed and r['kind']==kind];cards=[c for r in rr for c in r['conditions'][name]['cards']];truth=[c['truth'] for c in cards if 'truth' in c]
        summaries.append(dict(seed=seed,condition=name,kind=kind,cases=len(rr),cards=len(cards),unsupported_extent=sum(c['chosen'] is None for c in cards),known_groups=len(truth),all_witness=sum(t['all_witness'] for t in truth),wrong_points=sum(t['wrong_points'] for t in truth),order=dict(Counter(o['state'] for r in rr for o in r['conditions'][name]['order'])),progression_states=dict(Counter(c['progression_state'] for c in cards if 'progression_state' in c))))
    sk=lambda x:(x['seed'],x['condition'],x['kind']);ck('all summary denominators',sorted(summaries,key=sk)==sorted(result['summary'],key=sk))
    protections=rd(GRAPH/'PROTECTIONS.json');pins(protections,'protected');ck('protected477',len(protections)==477)
    ck('old models and web unchanged',tc['common100_changed'] is False and tc['ranker_changed'] is False and tc['web_changed'] is False)
    for obj,label in [(cfg,'train'),(prep,'prepare'),(ef,'eval')]:
        for k in ['private_access','final8','web_changed']:ck(label+k,obj[k] is False)
    statsum=[]
    for seed,split in itertools.product([42,43,44],['train','probe']):
        rr=[r for r in static_out if r['seed']==seed and r['split']==split];statsum.append(dict(seed=seed,split=split,eligible=sum(r['eligible'] for r in rr),false_progression=sum(r['false_progression'] for r in rr)))
    normal=[r for r in diags if r['kind'] in ['full_normal','speed075','speed125']]
    result=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='PASS' if all(c['ok'] for c in audit.checks) else 'FAIL',checks=len(audit.checks),failed=[c for c in audit.checks if not c['ok']],rows=len(new),static_bags=len(static),family2_preserved=f2,protected_files=len(protections),max_static_explicit_pose_variation=max_relation_variation,max_static_explicit_pose_std=max_relation_std,static_summary=statsum,static_details=static_out,real_bag_logits=real_out,progression_states=dict(Counter(r['state'] for r in diags)),normal_speed_states=dict(Counter(r['state'] for r in normal)),changed_extents=changes,diagnostics=diags,summary=summaries,no_inference=True,no_training=True,adopted=False,full_G2_pass=False,script_sha256=sha(__file__),shared_arithmetic_sha256=sha(P/'verify.py'),limitations=['Synthetic static labels mean zero observed progression, not linguistic incorrectness or semantic omission.','Raw training/probe candidate counts are before body admission and correlated; not independent-person accuracy.','Only the first original feature per bag was regenerated in preparation; all original candidates were not reencoded.','New decoder cannot certify correct extent or completeness; normal regression still blocks adoption.'],reviewer='Separate Codex motion_contract_review_1113 saved-data audit, not human semantic truth')
    (S/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ['status','checks','failed','rows','static_bags','family2_preserved','protected_files','static_summary','normal_speed_states']},ensure_ascii=False))
    if result['failed']:raise SystemExit(1)

if __name__=='__main__':main()
