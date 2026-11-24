from pathlib import Path
from datetime import datetime
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[4];LAB=ROOT/'research/front_keyphase_v1';U=Path(__file__).resolve().parent;OLD=LAB/'ops/word100_face_body_stage_20261027'
sys.path.insert(0,str(LAB/'src'))
from wis_lab.core.observation import Observation,sha256_file
from wis_lab.capture.face_landmarks import join_face_observation
from wis_lab.core.features import sequence_features,uniform_time_sample
from wis_lab.core.dtw import frame_cost_matrix,_optimal_path
from wis_lab.core.hand_face_evidence import build_evidence,arm_motion_direction,motion_cost,reference_projection_status
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item()),encoding='utf-8')

def freeze():
    p=U/'FROZEN.json'
    if p.exists():raise FileExistsError(p)
    f=read(OLD/'FROZEN.json');sources={}
    for r in f['rows']:
        for name in ('masked.npz','face_roi.npz','DONE.json','RECOVERED.json'):
            path=OLD/'data'/r['clip_id']/name;sources[str(path)]=sha256_file(str(path))
    for name in ('FROZEN.json','FIXED_SCOPE.json','VERIFIED.json'):
        path=OLD/name;sources[str(path)]=sha256_file(str(path))
    for p0,h in f['protected'].items():assert sha256_file(p0)==h
    write(p,dict(at=datetime.now().astimezone().isoformat(),public_rows=f['rows'],source_pins=sources,protected=f['protected'],
       baseline='previous B: same full-frame-derived ROI observations, existing feature cost and active span',
       candidate='baseline cost + 0.25 * mean observed elbow/wrist motion-direction disagreement',
       motion=dict(window_us=100000,min_displacement_shoulder_ratio=.03,weight=.25,max_bridge_gap_us=100000),
       no_search=True,no_boundary_or_label_used_by_alignment=True,no_trim_change=True,
       reference='two other people, keep each separate; query excluded from reference',
       overlap='projected hand hull or forearm capsule relative to six face patches; never proves occlusion/contact/error',
       no_expressions=True,no_classifier_inference=True,no_training=True,no_user_or_mentor=True,no_web_replacement=True,
       known_mapping=['uniform0.75length','uniform1.5length','nonlinear1.37length power1.35'],
       acceptance=['No hidden source mutation','No query-dependent reference denominator',
                   'Do not count projection alone as wrong','Known-time alignment nonregression',
                   'Inspect public normal crossing and hand lowering, no frames trimmed',
                   'Report coarse annotation metric separately from actual visual phase correctness'],
       prior_audit='45 annotation-discordant frames: 33 within100ms; none is automatically a known sign error'))
    print('FROZEN')

def load():
    f=read(U/'FROZEN.json');cache={};(U/'data').mkdir(exist_ok=True)
    for r in f['public_rows']:
        cid=r['clip_id'];p=OLD/'data'/cid
        obs=Observation.load(str(p/'masked.npz'));face=join_face_observation(p/'face_roi.npz',obs)
        evidence=build_evidence(obs,face);d,g=arm_motion_direction(obs)
        np.savez_compressed(U/'data'/f'{cid}_evidence.npz',**evidence)
        cache[cid]=(obs,evidence,d,g)
    return f,cache

def paths(a,b,ad,ag,bd,bg):
    sa=uniform_time_sample(sequence_features(a,trim=True),96);sb=uniform_time_sample(sequence_features(b,trim=True),96)
    cost,ev=frame_cost_matrix(sa,sb);ia,ib=sa['frame_id'],sb['frame_id']
    extra,count=motion_cost(ad[ia],ag[ia],bd[ib],bg[ib]);ans={}
    for name,c in (('B',cost),('D',cost+.25*extra)):
        path=np.array(_optimal_path(c,.2),int)
        ans[name]=(ia[path[:,0]],ib[path[:,1]])
    return ans

def map_to_query(a,b,n):
    out=np.full(n,-1,int)
    for i in range(int(b.min()),int(b.max())+1):
        nearest=np.abs(b-i).min();out[i]=int(np.median(a[np.abs(b-i)==nearest]))
    return out

def coarse(ref,q,ro,qo,a,b):
    mapping=map_to_query(a,b,qo.n_frames);required=np.zeros(qo.n_frames,bool)
    for x,y in q['intervals']:required|=(qo.timestamp_us>=x*1e6)&(qo.timestamp_us<=y*1e6)
    ids=np.flatnonzero(required);inside=ids[mapping[ids]>=0];rtime=ro.timestamp_us[mapping[inside]]
    matched=np.zeros(len(inside),bool)
    for x,y in ref['intervals']:matched|=(rtime>=x*1e6)&(rtime<=y*1e6)
    return dict(required=len(ids),unmapped=len(ids)-len(inside),annotation_discordant=int((~matched).sum()))

def warp(obs,mode):
    n=obs.n_frames;fraction={'short':.75,'long':1.5,'nonlinear':1.37}[mode];m=round(n*fraction)
    x=np.linspace(0,1,m);x=x**1.35 if mode=='nonlinear' else x;ids=np.round(x*(n-1)).astype(int)
    q=Observation(obs.xy_px[ids].copy(),np.arange(m,dtype=np.int64)*33333,np.arange(m,dtype=np.int64),
                  obs.observed[ids].copy(),obs.usable[ids].copy(),obs.conf[ids].copy(),obs.hand_score[ids].copy(),dict(obs.sidecar))
    q.validate();return q,ids

def evaluate():
    if (U/'RESULT.json').exists():raise FileExistsError('Completed results retained')
    f,cache=load();rows=f['public_rows'];cross=[];known=[];pathcache={}
    for q in rows:
        qo,qe,qd,qg=cache[q['clip_id']]
        for ref in (r for r in rows if r['word']==q['word'] and r['signer']!=q['signer']):
            ro,re,rd,rg=cache[ref['clip_id']];pair=ref['clip_id']+'__'+q['clip_id']
            pp=paths(ro,qo,rd,rg,qd,qg)
            # Reuse existing baseline paths where available, assert exact same result.
            if ref['signer']==1:
                with np.load(OLD/'data'/q['clip_id']/'cross_paths.npz') as z:
                    assert np.array_equal(pp['B'][0],z['B_reference']) and np.array_equal(pp['B'][1],z['B_query'])
            np.savez_compressed(U/'data'/f'{pair}_paths.npz',**{name+'_'+side:p[i] for name,p in pp.items() for i,side in enumerate(('reference','query'))})
            for name,(a,b) in pp.items():
                cross.append(dict(query=q['clip_id'],reference=ref['clip_id'],condition=name,**coarse(ref,q,ro,qo,a,b)))
                pathcache[(q['clip_id'],ref['clip_id'],name)]=(a,b)
        for mode in ('short','long','nonlinear'):
            w,truth=warp(qo,mode);wd,wg=arm_motion_direction(w);pp=paths(qo,w,qd,qg,wd,wg)
            np.savez_compressed(U/'data'/f"{q['clip_id']}_{mode}_paths.npz",truth=truth,**{name+'_'+side:p[i] for name,p in pp.items() for i,side in enumerate(('reference','query'))})
            for name,(a,b) in pp.items():
                err=np.abs(qo.timestamp_us[a]-qo.timestamp_us[truth[b]])/1000
                known.append(dict(clip_id=q['clip_id'],mode=mode,condition=name,mae_ms=float(err.mean()),p95_ms=float(np.percentile(err,95)),pairs=len(a)))
    # Relation evidence on retained B path: candidate isn't silently adopted based on scores.
    comparisons=[]
    for q in rows:
        qo,qe,_,_=cache[q['clip_id']];refs=[r for r in rows if r['word']==q['word'] and r['signer']!=q['signer']]
        flags=[];knowns=[];distances=[];avail=[];directs=[];mappings=[]
        for ref in refs:
            ro,re,_,_=cache[ref['clip_id']];a,b=pathcache[(q['clip_id'],ref['clip_id'],'B')];m=map_to_query(a,b,qo.n_frames);ok=m>=0;pick=np.maximum(m,0)
            flags.append(re['projected'][pick]);knowns.append(re['projection_known'][pick]&ok[:,None,None,None])
            distances.append(re['distance'][pick]);avail.append(re['available'][pick]&ok[:,None,None,None]);directs.append(re['direct_comparable'][pick]&ok[:,None,None,None]);mappings.append(m)
        flags=np.array(flags);knowns=np.array(knowns);ds=np.array(distances);av=np.array(avail);direct=np.array(directs)
        status=reference_projection_status(qe['projected'],qe['projection_known'],flags,knowns)
        expected=av.all(axis=0);compared=expected&qe['available'];direct_expected=direct.all(axis=0);direct_compared=direct_expected&qe['direct_comparable']
        with np.errstate(invalid='ignore'):
            low=np.min(ds,axis=0);high=np.max(ds,axis=0)
            delta=np.maximum(low-qe['distance'],0)+np.maximum(qe['distance']-high,0)
        low[~expected]=np.nan;high[~expected]=np.nan;delta[~compared]=np.nan
        np.savez_compressed(U/'data'/f"{q['clip_id']}_reference_comparison.npz",projection_status=status,
            reference_ids=np.array([r['clip_id'] for r in refs]),reference_rows=np.array(mappings),
            expected=expected,compared=compared,direct_expected=direct_expected,direct_compared=direct_compared,
            low=low,high=high,outside_two_reference_range=delta)
        comparisons.append(dict(query=q['clip_id'],references=[r['clip_id'] for r in refs],
           projection_counts={str(k):int((status==k).sum()) for k in range(5)},
           expected_cells=int(expected.sum()),compared_cells=int(compared.sum()),
           direct_expected_cells=int(direct_expected.sum()),direct_compared_cells=int(direct_compared.sum()),
           automatic_sign_errors=0,contact_confirmed=0))
    summary={}
    for name in 'BD':
        cr=[r for r in cross if r['condition']==name];kr=[r for r in known if r['condition']==name]
        summary[name]=dict(cross_pairs=len(cr),required_frames=sum(r['required'] for r in cr),
                         unmapped=sum(r['unmapped'] for r in cr),annotation_discordant=sum(r['annotation_discordant'] for r in cr),
                         known_tests=len(kr),mean_known_mae_ms=float(np.mean([r['mae_ms'] for r in kr])))
    regress=sum(d['mae_ms']>b['mae_ms']+1e-6 for b,d in zip(known[::2],known[1::2]))
    improve=sum(d['mae_ms']<b['mae_ms']-1e-6 for b,d in zip(known[::2],known[1::2]))
    write(U/'RESULT.json',dict(cross=cross,known_mapping=known,relation_comparisons=comparisons,
       summary=summary,known_mapping_regressed=regress,known_mapping_improved=improve,
       projection_status_codes={'0':'insufficient','1':'no projection in compared data','2':'projection supported by two references','3':'references vary','4':'different projection; review only'},
       active_alignment='B preserved; candidate D evaluation only',expression_features=0,user_access=0,training=0))
    print(json.dumps(dict(summary=summary,known_regress=regress,known_improve=improve),ensure_ascii=False))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');{'freeze':freeze,'evaluate':evaluate}[sys.argv[1]]()
