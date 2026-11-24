"""C5: saved FIT progression outputs only; no torch, labels or model forward."""
from pathlib import Path
import collections, hashlib, json, time, traceback
import numpy as np
U=Path(__file__).resolve().parent;H=U.parent;G=H.parent;W=G/'candidate_description_supervision'
B=W/'source_boundary_candidate_v3';F=[2,3,4,5,7]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def key(r):return r['clip_id'],r['view'],r['reverse']
def sigmoid(v):return 1/(1+np.exp(-np.clip(v.astype(np.float64),-700,700)))
def main():
    assert not(U/'RESULTS.json').exists()
    for d in[U,H,G]:
        for n in['STOP','STOP.txt','STOP_REQUESTED','STOP_REQUESTED.json']:assert not(d/n).exists()
    inputs=B/'INPUTS_COMPLETE.json';tr=B/'TRAINING_COMPLETE.json';rank=W/'local_rank_learning/INPUTS_COMPLETE.json'
    dense=G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json'
    ni=H/'null_candidate_learning/TRAINING_INPUT_INDEX.json';nm=H/'null_candidate_learning/local_bag_evaluation/RULES_FROZEN.json'
    pins={str(p):sha(p)for p in[Path(__file__),inputs,tr,rank,dense,ni,nm,W/'prepare_source_boundary.py',W/'source_boundary_core.py']}
    meta=read(inputs);train=read(tr);seq={key(r):(i,r)for i,r in enumerate(meta['sequences'])if r['split']=='train'}
    assert len(seq)==96 and {r['person']for i,r in seq.values()}=={2,4,5,6}
    sources={r['clip_id']:r for r in read(dense)['rows']if r['person']in[2,4,5,6]}
    assert len(sources)==16 and {k[0]for k in seq}==set(sources)
    truth={}
    for k,(i,r)in seq.items():
        assert sha(r['path'])==r['sha256'];pins[r['path']]=r['sha256']
        with np.load(r['path'],allow_pickle=False)as z:truth[k]={name:z[name].copy()for name in['y','mask','bands','valid']}
        assert np.all(truth[k]['bands']<=truth[k]['valid'])
    bags=[];omitted=collections.Counter()
    for r in read(rank)['bags']:
        if r['split']!='train':continue
        if key(r)not in seq:omitted['rank_bags_without_saved_boundary_source']+=1;continue
        assert sha(r['path'])==r['sha256'];pins[r['path']]=r['sha256']
        with np.load(r['path'],allow_pickle=False)as z:
            for c in np.flatnonzero(z['y']):bags.append(dict(key=key(r),family=r['family'],type='full_source_extent',span=z['spans'][c].tolist(),candidate_index=int(c)))
    nrows=read(ni)['rows'];masks=read(nm)['partial_masks'];assert len(nrows)==len(masks)
    for r,m in zip(nrows,masks):
        assert all(r[k]==m[k]for k in['clip_id','family','split','view','reverse'])
        if r['split']!='train':continue
        if key(r)not in seq:omitted['partial_bags_without_saved_boundary_source']+=1;continue
        for c in m['candidates']:bags.append(dict(key=key(r),family=r['family'],type='direct_partial_progression',span=c['transformed_span'],candidate_index=c['candidate_index']))
    base=[];bandrelations=[]
    for cid,s in sources.items():
        gs=sorted(s['groups'],key=lambda g:g['onset'][0])
        for a,b in zip(gs,gs[1:]):
            bandrelations.append(dict(clip_id=cid,left_family=a['family'],right_family=b['family'],left_offset=a['offset'],right_onset=b['onset'],
                bands_relation='positive_overlap'if max(a['offset'][0],b['onset'][0])<min(a['offset'][1],b['onset'][1])else'shared_instant'if a['offset'][1]==b['onset'][0]else'separated'if a['offset'][1]<b['onset'][0]else'other'))
    runs=[]
    for run in train['rows']:
        if run['condition']!='direct_boundary':continue
        assert sha(run['predictions'])==run['predictions_sha256'];pins[run['predictions']]=run['predictions_sha256']
        summary=collections.defaultdict(collections.Counter);perfamily=collections.defaultdict(collections.Counter);candidate_records=[];hold_records=[]
        with np.load(run['predictions'],allow_pickle=False)as pred:
            probabilities={}
            for k,(i,r)in seq.items():
                t=truth[k];logits=pred[f'logits_{i}'];p=sigmoid(logits[:,:,2]);probabilities[k]=p
                assert logits.shape==t['y'].shape+(3,) and np.isfinite(logits).all()
                for j,family in enumerate(F):
                    pos=(t['mask'][:,j]>0)&(t['y'][:,j]>0);neg=(t['mask'][:,j]>0)&~pos
                    c=perfamily[family];c['positive_frames']+=int(pos.sum());c['positive_supported']+=int((p[pos,j]>=.5).sum());c['negative_frames']+=int(neg.sum());c['negative_false_support']+=int((p[neg,j]>=.5).sum())
                    for h,which in enumerate(['onset','offset']):
                        valid=t['valid'][:,j,h];band=t['bands'][:,j,h]
                        if band.any():
                            a=int(np.argmax(np.where(valid,logits[:,j,h],-np.inf)));c[which+'_groups']+=1;c[which+'_argmax_in_band']+=int(band[a])
                s=sources[k[0]];last=max(s['groups'],key=lambda g:g['offset'][1]);family=last['family'];j=F.index(family)
                original=s['hold_sample_frames'];idx=np.array([len(p)-1-f if r['reverse']else f for f in original],np.int64)
                if len(idx):
                    observed=t['valid'][idx,j,0];known=t['mask'][idx,j]>0
                    row=dict(clip_id=k[0],view=k[1],reverse=k[2],family=family,source_frames=original,observed=int(observed.sum()),present_in_training_presence_mask=int(known.sum()),progression_probability=p[idx,j].tolist(),supported_observed=int((observed&(p[idx,j]>=.5)).sum()))
                    hold_records.append(row)
                    summary['own_terminal_hold']['annotated_frames']+=len(idx);summary['own_terminal_hold']['observed_frames']+=int(observed.sum());summary['own_terminal_hold']['in_presence_training_mask']+=int(known.sum());summary['own_terminal_hold']['false_progress_support']+=row['supported_observed']
            for b in bags:
                k=b['key'];j=F.index(b['family']);a,z=b['span'];t=truth[k];p=probabilities[k];valid=t['valid'][a:z+1,j,0];s=summary[b['type']];s['candidates']+=1
                if not valid.any():s['no_observed_point']+=1;continue
                ix=int(a+np.argmax(np.where(valid,p[a:z+1,j],-np.inf)));v=float(p[ix,j]);known=t['mask'][ix,j]>0
                state='direct_progression_witness'if known and t['y'][ix,j]>0 else'reviewed_negative'if known else'unlabelled'
                s[state]+=1;s['argmax_supported']+=int(v>=.5)
                candidate_records.append(dict(clip_id=k[0],view=k[1],reverse=k[2],family=b['family'],type=b['type'],candidate_index=b['candidate_index'],span=[a,z],argmax_frame=ix,probability=v,anchor_evidence=state))
        runs.append(dict(seed=run['seed'],per_family={str(k):dict(v)for k,v in perfamily.items()},summary={k:dict(v)for k,v in summary.items()},candidate_anchor_diagnostics=candidate_records,own_hold_diagnostics=hold_records))
    assert {r['seed']for r in runs}=={42,43,44}
    for p,h in pins.items():assert sha(p)==h,p
    write(U/'RESULTS.json',dict(status='FIT_SAVED_OUTPUT_DIAGNOSIS_ONLY_NO_NEW_FORWARD_OR_LABELS',fit_source_clips=16,fit_people=[2,4,5,6],sequence_views_and_reversals=96,
        partial_scope='Existing N/P direct partial masks only; family2 excluded in those masks. No new partial labels created.',
        candidates_by_type=dict(collections.Counter(r['type']for r in bags)),excluded_bags=dict(omitted),band_relations=bandrelations,runs=runs,
        warning='FIT same-training-output diagnostic, not held-out accuracy. A maximum on unlabelled frames is unknown, not wrong. Own reviewed hold is a progression-negative observation, not an incorrect-sign label. No new anchor decoder or gate adopted.',
        no_new_inference=True,no_training=True,no_probe_array_access=True,no_private=True,no_final8=True,no_web=True,G2_pass=None,G3_pass=None,pins=pins))
    print(json.dumps(dict(candidates=dict(collections.Counter(r['type']for r in bags)),runs=[dict(seed=r['seed'],summary=r['summary'],per_family=r['per_family'])for r in runs]),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        with (U/f'FIRST_FAILURE_{time.time_ns()}.txt').open('x',encoding='utf-8')as f:f.write(traceback.format_exc())
        raise
