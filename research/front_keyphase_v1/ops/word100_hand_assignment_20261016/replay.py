"""Frozen raw detector replay: only side assignment changes, no model inference."""
from pathlib import Path
from collections import Counter
import json,sys,time
from types import SimpleNamespace
import numpy as np

U=Path(__file__).resolve().parent;LAB=U.parents[1]
sys.path[:0]=[str(LAB/'src'),str(LAB.parents[1]/'research/sign_model_pilot_v1/src')]
from wis_lab.core.observation import Observation,empty_frames,sha256_file
from wis_lab.capture.landmarker import Landmarker
from pilot.web_input import to_native_like,active_span
from pilot.data import build_channels
from assignment import HandAssigner
sys.stdout.reconfigure(encoding='utf-8')

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write_new(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def same(a,b):return np.array_equal(a,b,equal_nan=True) if np.issubdtype(a.dtype,np.floating) else np.array_equal(a,b)

def run():
    assert not (U/'REPLAY_RESULT.json').exists(),'Reuse completed result'
    capture=read(U/'RAW_CAPTURE_RESULT.json');ctx=read(U/'RAW_CAPTURE_CONTEXT.json')
    for p,h in ctx['pins'].items():assert sha256_file(p)==h,p
    files=[U/x for x in ['SCOPE.json','assignment.py','research_landmarker.py','replay.py','RAW_CAPTURE_CONTEXT.json','RAW_CAPTURE_RESULT.json']]
    for r in capture['rows']:
        for p,h in r['output_sha256'].items():assert sha256_file(p)==h,p
        files.extend(Path(p) for p in r['output_sha256'])
    pins={str(p):sha256_file(p) for p in files}
    write_new(U/'REPLAY_CONTEXT.json',{'pins':pins,'candidate_id':HandAssigner.identity(),
        'scope':'same raw detections, assignment only, TRAIN12/native+crop24, no classifier',
        'dataset_role':'engineering diagnostic fixtures, not independent recognition evaluation'})
    contract=read(LAB/'config/contract.json');margin=contract['usable_rule']['bounds_margin_ratio'];thr=contract['usable_rule']['hand_score_threshold']
    rows=[];total=Counter();start=time.perf_counter()
    (U/'replay').mkdir(exist_ok=True)
    for rec in capture['rows']:
        cid,mode=rec['clip_id'],rec['mode'];d=U/'raw_capture'/cid/mode
        with np.load(d/'raw.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        old=Observation.load(str(d/'baseline.npz'));out=empty_frames(old.n_frames)
        for name in ['xy_px','observed','usable','conf']:out[name][:,:33]=getattr(old,name)[:,:33]
        out['timestamp_us'][:]=old.timestamp_us;out['frame_id'][:]=old.frame_id
        candidate=HandAssigner();trace=[];counts=Counter();changed=[]
        base=Landmarker.__new__(Landmarker);base.handedness_comparisons=0;base.handedness_disagreements=0
        side_array=np.full((old.n_frames,2),-1,np.int8)
        for f in range(old.n_frames):
            n=int(raw['raw_count'][f]);w,h=int(raw['width'][f]),int(raw['height'][f])
            hands=raw['hand_xy_px'][f,:n];labels=raw['raw_label'][f,:n];scores=raw['raw_score'][f,:n]
            # Old method replay with raw float64 wrist values and original float32 pose.
            res=SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=p[0]/w,y=p[1]/h) for p in pts] for pts in hands],
                handedness=[[SimpleNamespace(score=float(scores[i]),category_name=str(raw['raw_label_name'][f,i]))] for i in range(n)])
            baseline=base._assign_sides(res,raw['pose_xy_px'][f],raw['pose_usable'][f],w,h) if n else {}
            assert [baseline[i][0] if baseline[i][0] is not None else -1 for i in range(n)]==raw['baseline_side'][f,:n].tolist(),(cid,mode,f,'baseline replay mismatch')
            sides,diag=candidate.assign(hands,labels,scores,raw['pose_xy_px'][f],raw['pose_usable'][f],int(old.timestamp_us[f]),w,h)
            diag.update(frame=f,timestamp_us=int(old.timestamp_us[f]));trace.append(diag)
            for i in range(n):
                s=sides[i];b=int(raw['baseline_side'][f,i]);side_array[f,i]=-1 if s is None else s
                category='unchanged' if s==b or (s is None and b==-1) else ('newly_assigned' if b==-1 else 'dropped' if s is None else 'side_changed')
                counts[category]+=1;counts['raw_detections']+=1
                counts['reason_'+diag['reasons'][i]]+=1
                if category!='unchanged':changed.append({'frame':f,'raw_index':i,'old_side':b,'new_side':-1 if s is None else s,'category':category,'reason':diag['reasons'][i]})
                if s is None:continue
                sl=slice(33+21*s,54+21*s);points=hands[i].astype(np.float32)
                out['xy_px'][f,sl]=points;out['observed'][f,sl]=True;out['hand_score'][f,s]=scores[i]
                mx,my=margin*w,margin*h
                out['usable'][f,sl]=np.isfinite(points).all(1)&(points[:,0]>=-mx)&(points[:,0]<=w+mx)&(points[:,1]>=-my)&(points[:,1]<=h+my)&(scores[i]>=thr)
            counts['zero_detection_frames']+=int(n==0)
        ob=Observation(**out);ob.sidecar=dict(old.sidecar)
        ob.sidecar.update(extractor_id='handassoc2_'+HandAssigner.identity()[:16],hand_assignment_policy='research_causal_bounded_hand_association',
            replay_raw_sha256=sha256_file(d/'raw.npz'),candidate_code_sha256=HandAssigner.identity(),
            handedness_label_comparisons=sum(int(s>=0) for v in side_array for s in v),
            handedness_label_disagreements=int(((side_array>=0)&(side_array!=raw['raw_label'])).sum()),
            unassigned_hand_detections=int(sum(int(raw['raw_count'][f])-(side_array[f]>=0).sum() for f in range(old.n_frames))))
        ob.sidecar['quality']=ob.quality_summary();ob.validate()
        nat,pr=to_native_like(ob.xy_px,ob.usable);span=active_span(pr);channels=None
        if span:
            lo,hi=span;ch=build_channels(nat[lo:hi+1],pr[lo:hi+1],ob.timestamp_us[lo:hi+1]);x=ch['x']
            assert x.shape[1]==280 and np.isfinite(x).all()
            assert (x[:,:102].reshape(-1,51,2)[~x[:,186:237].astype(bool)]==0).all()
            assert (x[:,102:186].reshape(-1,42,2)[~x[:,237:279].astype(bool)]==0).all()
            channels={'shape':list(x.shape),'finite':True,'masked_coordinates_zero':True,'internal_time_preserved':True}
        dest=U/'replay'/cid/mode;dest.mkdir(parents=True)
        ob.save(str(dest/'candidate.npz'))
        np.savez_compressed(dest/'assignment.npz',sides=side_array)
        write_new(dest/'trace.json',trace);write_new(dest/'changes.json',changed)
        row={'clip_id':cid,'label':rec['label'],'mode':mode,'frames':old.n_frames,'counts':dict(counts),
            'baseline_replay_exact':True,'channels':channels,'before_quality':old.quality_summary(),'after_quality':ob.quality_summary(),
            'outputs':{str(p):sha256_file(p) for p in dest.iterdir() if p.is_file()}}
        rows.append(row);total.update(counts)
        print(json.dumps({'clip':cid,'mode':mode,'counts':dict(counts)},ensure_ascii=False),flush=True)
    for p,h in pins.items():assert sha256_file(p)==h,p
    write_new(U/'REPLAY_RESULT.json',{'status':'COMPUTED_PENDING_INDEPENDENT_AND_VISUAL_REVIEW','conditions':len(rows),
        'counts':dict(total),'rows':rows,'seconds':time.perf_counter()-start,'training':0,'classifier_inference':0,'production_changes':0})

if __name__=='__main__':run()
