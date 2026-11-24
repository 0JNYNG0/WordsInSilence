"""Separate all-output reproduction, without importing the producer/candidate functions."""
from pathlib import Path
import json,hashlib,time,traceback
from statistics import median
from collections import defaultdict
import numpy as np
U=Path(__file__).resolve().parent;P=U.parent/'word100_metric_evidence_20261020';O=U.parent/'word100_multiref_observation_compare_20261019'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')

def run():
    beg=time.time();assert not (U/'VERIFICATION.json').exists()
    f=read(U/'FROZEN.json');pins=read(U/'INPUT_PINS.json');rows=read(U/'ROWS.json');real=read(U/'ACTUAL_GAPS.json');summ=read(U/'SUMMARY.json')
    assert sha(U/'audit.py')==f['runner_sha256'] and sha(P/'candidate.py')==f['candidate_sha256']
    for name,h in f['previous_artifact_sha256'].items():assert sha(P/name)==h,name
    for p,h in f['protected'].items():assert sha(p)==h,p
    for section in pins.values():
        for p,h in section.items():assert sha(p)==h,p
    byword=defaultdict(list)
    for row in rows:byword[row['word']].append(row)
    source={r['clip_id']:r for r in read(O/'FROZEN.json')['inputs']}
    values=0;pairs_checked=0;maxerror=0.;selfchecked=0
    for word,wr in byword.items():
        old=read(O/'words/sensor'/(word+'.json'));features={q['clip_id']:load(P/'features/sensor'/(q['clip_id']+'.npz')) for q in old['queries']}
        for row in wr:
            cid=row['clip_id'];s=row['side'];q=features[cid];n=len(q['timestamp_us']);indices=set(row['mask_indices']);mode=row['condition'];ts=q['timestamp_us']/1e6
            assert row['signer_id'] in (11,12,13)
            if mode=='whole':assert indices==set(range(n))
            else:
                a,b=max(source[cid]['intervals'],key=lambda x:x[1]-x[0]);width=(b-a)*(.2 if mode=='peak20' else .1)
                center=(a+b)/2
                if mode!='center10':
                    options=[float(t) for t in ts if a<t<b and a+width/2<=t<=b-width/2]
                    def mass(c):
                        return sum(float(np.linalg.norm(q['position_'+s][i]-q['position_'+s][i-1])) for i in range(1,n) if c-width/2<=ts[i]<=c+width/2 and q['position_valid_'+s][i] and q['position_valid_'+s][i-1])
                    if options:center=max(options,key=mass)
                expected={i for i,t in enumerate(ts) if a<t<b and center-width/2<=t<=center+width/2}
                assert indices==expected,(cid,s,mode)
            assert row['removed_angle_cells']==sum(int(np.isfinite(q['angles_'+s][i]).sum()) for i in indices)
            assert row['removed_position_frames']==sum(bool(q['position_valid_'+s][i]) for i in indices)
            av=[[] for _ in range(15)];lv=[];mv=[]
            for pair in old['pairs']:
                if pair['query_clip']!=cid:continue
                assert 1<=pair['reference_signer']<=9
                r=features[pair['reference_clip']];path=pair['diagnostics']['path'];pairs_checked+=1
                if pair['result']['status']!='OK':continue
                for j in range(15):
                    selected=[(a,b) for a,b in path if a not in indices and np.isfinite(q['angles_'+s][a,j]) and np.isfinite(r['angles_'+s][b,j])]
                    if len(selected)/len(path)<.5 or min(len({a for a,b in selected}),len({b for a,b in selected}))<8:continue
                    av[j].append(sum(abs(float(q['angles_'+s][a,j])-float(r['angles_'+s][b,j])) for a,b in selected)/len(selected))
                selected=[(a,b) for a,b in path if a not in indices and q['position_valid_'+s][a] and r['position_valid_'+s][b]]
                if len(selected)/len(path)>=.5 and min(len({a for a,b in selected}),len({b for a,b in selected}))>=8:
                    aidx,bidx=zip(*selected);qa=q['position_'+s][list(aidx)];rb=r['position_'+s][list(bidx)]
                    lv.append(float(np.sqrt(np.sum((qa-rb)**2,axis=1)).mean()*100))
                    oq=q['position_'+s][next(i for i,v in enumerate(q['position_valid_'+s]) if v)];orr=r['position_'+s][next(i for i,v in enumerate(r['position_valid_'+s]) if v)]
                    mv.append(float(np.sqrt(np.sum(((qa-oq)-(rb-orr))**2,axis=1)).mean()*100))
            aa=[float(median(v)) if len(v)>=5 else None for v in av]
            actual={'handshape':sum(aa)/15 if all(v is not None for v in aa) else None,'location':float(median(lv)) if len(lv)>=5 else None,'movement':float(median(mv)) if len(mv)>=5 else None}
            assert row['result']['angles_ready']==sum(v is not None for v in aa)
            for key,x in actual.items():
                y=row['result'][key];values+=1
                assert (x is None)==(y is None)
                if x is not None:
                    err=abs(x-y);maxerror=max(maxerror,err);assert err<1e-4,(cid,s,mode,key,err)
            for x,y in zip(aa,row['result']['angles']):
                values+=1;assert (x is None)==(y is None)
                if x is not None:assert abs(x-y)<1e-10
            if mode=='peak10':
                c=next(x for x in read(U/'SELF_CONTROLS.json') if x['clip_id']==cid and x['side']==s)
                eligible=sum(sum(i not in indices and np.isfinite(q['angles_'+s][i,j]) for i in range(n))>=max(8,n*.5) for j in range(15))
                assert c['angle_zero_count']==eligible;selfchecked+=1
        print('verified',word,flush=True) if len(byword)<10 else None
    assert len(byword)==100 and pairs_checked==summ['scope']['counterfactual_pair_hand_computations']
    # Recompute all aggregate outcome counts; published comparisons only use actual removed evidence.
    for mode,expected in summ['conditions'].items():
        rr=[r for r in rows if r['condition']==mode];assert expected['hand_units']==len(rr)==591
        for k,x in expected['metrics'].items():
            keep=[r for r in rr if r['removed_angle_cells' if k=='handshape' else 'removed_position_frames']>0 and r['baseline'][k] is not None and r['result'][k] is not None]
            ds=[r['result'][k]-r['baseline'][k] for r in keep]
            assert x['paired_effective']==len(ds)
            assert x['decreased']==sum(d < -1e-4 for d in ds)
            assert x['unchanged']==sum(abs(d)<=1e-4 for d in ds)
            assert x['masked_available']==sum(r['result'][k] is not None for r in rr)
    assert sum(r['inside_missing_frames'] for r in real)==355
    assert sum(r['inside_missing_frames']>0 and r['baseline']['handshape'] is not None for r in real)==summ['actual_gap']['whole_angle_available_despite_witnessed_gap']
    write('VERIFICATION.json',{'status':'PASS','same_operator_separate_implementation':True,'public_inputs_hashed':len(pins['features']),'saved_word_files_hashed':len(pins['words']),'rows':len(rows),'reproduced_values':values,'pair_hand_conditions':pairs_checked,'self_controls':selfchecked,'max_numeric_error':maxerror,'protected_files':len(f['protected']),'old_results_unchanged':True,'seconds':time.time()-beg})
    print('VERIFY PASS',flush=True)

if __name__=='__main__':
    try:run()
    except Exception:
        p=U/'FIRST_ERROR_verify.txt'
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
