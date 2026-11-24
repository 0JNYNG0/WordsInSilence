"""Frozen training-only cross-signer reference variability; no acceptance threshold."""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
U=Path(__file__).resolve().parent
sys.path.insert(0,str(U.parent/'word100_training_20261013'))
from common import BASE,OUT,read,write,sha
from wis_lab.core.observation import Observation
from wis_lab.service import word100_analysis as A
from wis_lab.core import features as F
from wis_lab.core.dtw import frame_cost_matrix,_optimal_path
D=U/'dtw96'

def stats(v):
    v=np.asarray([x for x in v if x is not None],float)
    return dict(n=len(v),min=float(v.min()),q25=float(np.quantile(v,.25)),median=float(np.median(v)),q75=float(np.quantile(v,.75)),max=float(v.max())) if len(v) else dict(n=0)

def freeze():
    assert not (D/'FROZEN_PLAN.json').exists()
    manifest=read(U.parent/'word100_training_20261013/MANIFEST.json')
    inv=read(U.parent/'word100_dtw_graphs_20261013/BASIS_INVENTORY.json')
    refs=A.references();pairs=[];pins={}
    for item in inv['items']:
        code=item['word_id'];reference=refs[code];assert reference['signer_id']==1
        rows=[r for r in manifest if r['word_code']==code and r['role']=='train_candidate']
        assert len(rows)==9 and {r['signer_id'] for r in rows}==set(range(1,10))
        for r in sorted(rows,key=lambda r:r['signer_id']):
            rp=OUT/'records'/f"{r['clip_id']}.json";rec=read(rp)
            assert sha(rec['observation'])==rec['observation_sha256']
            pins[str(rp)]=sha(rp);pins[rec['observation']]=rec['observation_sha256']
            side=Path(rec['observation']).with_suffix('.sidecar.json');pins[str(side)]=sha(side)
            if r['signer_id']==1:
                assert rec['observation_sha256']==reference['observation_sha256'];continue
            pairs.append(dict(code=code,label=item['label'],query_clip=r['clip_id'],query_signer=r['signer_id'],query_observation=rec['observation'],query_sha256=rec['observation_sha256'],reference_clip=reference['clip_id'],reference_observation=reference['observation'],reference_sha256=reference['observation_sha256']))
    assert len(pairs)==96
    for p in [Path(A.__file__),Path(F.__file__),BASE/'src/wis_lab/core/dtw.py',BASE/'src/wis_lab/core/observation.py',Path(__file__)]:pins[str(p)]=sha(p)
    write(D/'FROZEN_PLAN.json',dict(status='FROZEN_BEFORE_COMPARISON',purpose='Same-label descriptive variability, not linguistically certified correct/incorrect movements',pairs=pairs,source_sha256=pins,
        rules=dict(reference_signer=1,query_signers=list(range(2,10)),words=12,pairs=96,features='current production compare unmodified',budget=96,band=.2,threshold_fitting=False,model_selection=False,user_data=False),
        completion='96 pair records,12 self sanity controls, saved alignment independent axis recount, per-word quartiles/support; do not create correctness thresholds'))

def run():
    f=read(D/'FROZEN_PLAN.json');assert all(sha(p)==h for p,h in f['source_sha256'].items())
    assert not (D/'PAIRS.json').exists(),'Reuse saved results'
    (D/'aligned').mkdir(parents=True,exist_ok=True);out=[];cache={};controls=[]
    def load(p):
        if p not in cache:cache[p]=Observation.load(p)
        return cache[p]
    for i,p in enumerate(f['pairs']):
        qo=load(p['query_observation']);ro=load(p['reference_observation'])
        if not any(x['code']==p['code'] for x in controls):
            self_result=A.compare(ro,ro);assert self_result['status']=='OK'
            assert all(m['value'] is None or abs(m['value'])<1e-6 for m in self_result['metrics'])
            controls.append(dict(code=p['code'],result=self_result))
        result=A.compare(qo,ro)
        q=A._sample(qo);r=A._sample(ro);cost,evidence=frame_cost_matrix(q,r,hand_participation=False,one_sided_mode='flat');path=np.asarray(_optimal_path(cost,.2),int)
        artifact=D/'aligned'/f"{p['query_clip']}.npz"
        np.savez_compressed(artifact,q_features=q['features'],r_features=r['features'],q_present=q['present'],r_present=r['present'],q_frame_id=q['frame_id'],r_frame_id=r['frame_id'],path=path,cost=cost[path[:,0],path[:,1]],evidence=evidence[path[:,0],path[:,1]])
        diagonal=(np.diff(path,axis=0)==[1,1]).all(1)
        alignment=dict(non_diagonal_fraction=float((~diagonal).mean()),path_length=len(path),query_samples=len(q['features']),reference_samples=len(r['features']),max_normalized_time_offset=float(np.max(np.abs(path[:,0]/(len(q['features'])-1)-path[:,1]/(len(r['features'])-1)))))
        out.append(dict(p,result=result,alignment=alignment,query_quality=qo.quality_summary(),reference_quality=ro.quality_summary(),aligned_file=str(artifact),aligned_sha256=sha(artifact)))
        if (i+1)%12==0:print(f'DTW {i+1}/96',flush=True)
    write(D/'PAIRS.json',out);write(D/'SELF_CONTROLS.json',controls)
    write(D/'SUMMARY.json',summarize(out))

def summarize(rows):
    def group(rr):
        return dict(pairs=len(rr),status_counts={k:sum(r['result']['status']==k for r in rr) for k in sorted({r['result']['status'] for r in rr})},
          metrics={key:stats([next((m['value'] for m in r['result']['metrics'] if m['key']==key),None) for r in rr]) for key in ['handshape','location','movement']},
          axis_support={key:stats([next((m['support_fraction'] for m in r['result']['metrics'] if m['key']==key),None) for r in rr]) for key in ['handshape','location','movement']},
          non_diagonal_fraction=stats([r['alignment']['non_diagonal_fraction'] for r in rr]))
    return dict(status='COMPLETE_PENDING_INDEPENDENT_CHECK',all=group(rows),by_word={code:dict(label=next(r['label'] for r in rows if r['code']==code),**group([r for r in rows if r['code']==code])) for code in sorted({r['code'] for r in rows})},
                note='8 comparisons per word against one shared reference. Correlated repeated signers and shared reference; no pass threshold, no accuracy or incorrect-handshape claim.')

def verify():
    assert not (D/'INDEPENDENT_CHECK.json').exists()
    f=read(D/'FROZEN_PLAN.json');rows=read(D/'PAIRS.json');assert len(rows)==96
    assert all(sha(p)==h for p,h in f['source_sha256'].items())
    assert [(r['code'],r['query_signer']) for r in rows]==[(r['code'],r['query_signer']) for r in f['pairs']]
    offsets={};cursor=0
    for name,(width,weight) in F.GROUPS.items():offsets[name]=slice(cursor,cursor+width);cursor+=width
    comparisons=0
    for row in rows:
        assert sha(row['aligned_file'])==row['aligned_sha256']
        with np.load(row['aligned_file'],allow_pickle=False) as z:
            q=z['q_features'];r=z['r_features'];qok=z['q_present'];rok=z['r_present'];path=z['path'];cost=z['cost'];evidence=z['evidence']
        assert np.array_equal(path[0],[0,0]) and np.array_equal(path[-1],[len(q)-1,len(r)-1])
        assert all(tuple(step) in [(1,1),(0,1),(1,0)] for step in np.diff(path,axis=0))
        out=row['result'];assert out['accepted'] is None and out['is_a_verdict'] is False
        assert np.isclose(out['alignment_evidence_fraction'],evidence.mean())
        if out['status']!='OK':continue
        assert np.isclose(out['dtw_distance'],cost.mean())
        for metric in out['metrics']:
            values=[];supports=[]
            for side in ['L','R']:
                key=('bend' if metric['key']=='handshape' else 'handpos')+'_'+side;sl=offsets[key];gi=F.GROUP_ORDER.index(key)
                validq=qok[:,gi]&np.isfinite(q[:,sl]).all(1);validr=rok[:,gi]&np.isfinite(r[:,sl]).all(1)
                paired=[(a,b) for a,b in path if validq[a] and validr[b]];detail=next(s for s in metric['sides'] if s['side']==side)
                assert len(paired)==detail['compared_pairs'];supports.append(len(paired))
                if not paired:assert detail['value'] is None;continue
                vals=[];originq=q[np.flatnonzero(validq)[0],sl];originr=r[np.flatnonzero(validr)[0],sl]
                for a,b in paired:
                    qa=q[a,sl].astype(np.float64);rb=r[b,sl].astype(np.float64)
                    if metric['key']=='handshape':val=float(np.mean([abs(float(qa[k])-float(rb[k]))*180 for k in range(15)]))
                    elif metric['key']=='location':val=float(np.sqrt(sum((qa-rb)**2))*100)
                    else:val=float(np.sqrt(sum(((qa-originq)-(rb-originr))**2))*100)
                    vals.append(val)
                v=float(np.mean(vals));assert np.isclose(v,detail['value'],rtol=2e-5,atol=2e-5);values.append(v)
                assert len({a for a,b in paired})==detail['query_distinct_frames'] and len({b for a,b in paired})==detail['reference_distinct_frames']
                comparisons+=1
            expected=float(np.mean(values)) if values else None
            assert expected is None and metric['value'] is None or np.isclose(expected,metric['value'],rtol=2e-5,atol=2e-5)
            assert np.isclose(metric['support_fraction'],sum(supports)/(2*len(path)))
    summary=read(D/'SUMMARY.json');assert summary==summarize(rows)
    write(D/'INDEPENDENT_CHECK.json',dict(status='PASS',pairs=96,self_controls=12,hand_axis_values_recomputed=comparisons,method='Separate float64 scalar arithmetic from saved aligned feature/mask/path arrays, endpoint/monotonicity, sources, unique support and denominator verification',new_training=0,new_model_inference=0,thresholds_created=0))
    print(json.dumps(summary['all'],ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','verify']);a=p.parse_args();globals()[a.action]()
