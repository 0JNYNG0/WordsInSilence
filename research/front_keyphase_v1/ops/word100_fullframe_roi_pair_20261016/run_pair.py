"""Frozen TRAIN-only observation comparison; no training/classifier/camera imports."""
from pathlib import Path
import argparse, importlib.util, json, os, sys, time, traceback
import numpy as np

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent
LAB=U.parents[1]; ROOT=LAB.parents[1]
P=U.with_name('word100_hand_assignment_20261016')
J=U.with_name('word100_hand_assignment_joint_20261016')
A=U.with_name('word100_hand_armchain_20261016')
sys.path[:0]=[str(LAB/'src'),str(ROOT/'research/sign_model_pilot_v1/src'),str(U)]
from wis_lab.core.observation import Observation, empty_frames, sha256_file

def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def load_module(name,p):
    spec=importlib.util.spec_from_file_location(name,p); mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def pins_ok(pins):
    for p,h in pins.items(): assert sha256_file(p)==h,p
def event(**kw):
    kw.update(pid=os.getpid(),time=time.time())
    with (U/'PROGRESS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(kw,ensure_ascii=False)+'\n')
    print(json.dumps(kw,ensure_ascii=False),flush=True)

def prepare():
    assert not (U/'PLAN.json').exists()
    contexts=[read(P/'RAW_CAPTURE_CONTEXT.json'),read(J/'BROADENED_CAPTURE_CONTEXT.json')]
    items=[it for c in contexts for it in c['items']]
    assert len(items)==20 and len(set(it['clip_id'] for it in items))==20
    manifest=read(A/'INPUT_MANIFEST.json')
    rows=[];pins={}
    for it in items:
        assert it['source_role']=='train_candidate' and it['signer_id'] in (1,2,3)
        cid=it['clip_id'];w,h=it['original_size'];box=it['box']
        assert box[1]==0
        row={k:it[k] for k in ('clip_id','label','signer_id','source_role','video_path','source_sha256','original_size','box')}
        row['bottom_box']=[0,0,w,box[3]];row['cached']={}
        assert sha256_file(it['video_path'])==it['source_sha256']
        pins[it['video_path']]=it['source_sha256']
        for mode in ('native','crop'):
            rec=next(r for r in manifest['rows'] if r['clip_id']==cid and r['mode']==mode)
            pins.update(rec['pins'])
            d=J/'replay'/cid/mode
            for name in ('candidate.npz','candidate.sidecar.json','assignment.npz','trace.json'):
                pins[str(d/name)]=sha256_file(d/name)
            row['cached'][mode]={'observation':str(d/'candidate.npz'),'assignment':str(d/'assignment.npz'),'raw_dir':rec['raw_dir']}
        rows.append(row)
    fixed=[Path(__file__),U/'roi_adapter.py',LAB/'config/contract.json',LAB/'src/wis_lab/capture/landmarker.py',
        LAB/'src/wis_lab/capture/video_source.py',LAB/'src/wis_lab/core/observation.py',
        ROOT/'research/sign_model_pilot_v1/src/pilot/data.py',ROOT/'research/sign_model_pilot_v1/src/pilot/web_input.py',
        P/'capture_raw.py',J/'assignment.py',J/'SCOPE.json',J/'POINT_CONTRACT.json',
        ROOT/'Tools/models/hand_landmarker.task',ROOT/'Tools/models/pose_landmarker_lite.task',
        LAB/'runs/word100_tcn_v1/whole_context/best.pt']
    pins.update({str(p):sha256_file(p) for p in fixed})
    pins_ok(pins)
    write(U/'PLAN.json',{'status':'FROZEN_BEFORE_PAIR_RESULTS','items':rows,'pins':pins,
        'authorization':'User approved whole-frame detection then upper-body model masking, same videos, actual offscreen condition; review before training',
        'dataset_role':'20 previously fixed public TRAIN engineering clips; no user/dev/test inputs',
        'conditions':['legacy_crop_then_mask','full_then_legacy_mask','bottom_crop_then_mask','full_then_bottom_mask'],
        'assignment':'same frozen joint_v3 for every condition; no coefficient changes',
        'roi_policy':'Legacy matched pair preserves old exact boxes. Bottom matched pair [0,0,original_width,old_bottom] preserves full top/side extent. Old per-clip lower boundary is reused without optimization.',
        'sensor_boundary':'bottom_crop detector receives only RGB[:old_bottom,:]; new VIDEO state per clip, no original lower pixels/history; approximate webcam boundary, not real-user evidence',
        'mask':'all75 points strict bounds, missing NaN, usable and observed masked before280 features; original frame count/PTS retained; no interpolation',
        'normalization':'existing per-clip shoulder and hand scale median on masked usable points; completed-clip features, not causal streaming claim',
        'metrics':['per-side usable points/any/palm5/all21','same-frame gain/loss of observed evidence, not correctness','entry/reentry events from full-frame masked reference after >=3 missing frames, first6-frame window','known failure-frame comparison','visual paired input inspection'],
        'acceptance':{'engineering':'frame/time/input hash preservation, strict ROI, no hidden-coordinate feature influence, partial-hand validity, same assignment and no synthetic joints',
            'observation':'report gains and losses separately; no automatic adoption; hand-count gains are not recognition accuracy; visually inspect fixed6prior issues plus deterministic largest losses',
            'training':'explicitly not run; wait for user result review'},
        'new_detector_conditions':20,'training':0,'classifier_inference':0,'production_changes':0,'user_data':0})
    event(stage='prepared',clips=20,pinned_files=len(pins))

def associate(raw, baseline, assigner_cls, contract):
    out=empty_frames(baseline.n_frames)
    for key in ('xy_px','observed','usable','conf'):out[key][:,:33]=getattr(baseline,key)[:,:33]
    out['frame_id'][:]=baseline.frame_id;out['timestamp_us'][:]=baseline.timestamp_us
    ass=assigner_cls();sides=np.full((baseline.n_frames,2),-1,np.int8);trace=[]
    margin=contract['usable_rule']['bounds_margin_ratio'];threshold=contract['usable_rule']['hand_score_threshold']
    for f in range(baseline.n_frames):
        n=int(raw['raw_count'][f]);w=int(raw['width'][f]);h=int(raw['height'][f])
        chosen,diag=ass.assign(raw['hand_xy_px'][f,:n],raw['raw_label'][f,:n],raw['raw_score'][f,:n],
            raw['pose_xy_px'][f],raw['pose_usable'][f],int(baseline.timestamp_us[f]),w,h)
        diag.update(frame=f,timestamp_us=int(baseline.timestamp_us[f]));trace.append(diag)
        for i in range(n):
            side=chosen[i]
            if side is None:continue
            sides[f,i]=side;sl=slice(33+21*side,54+21*side);pts=raw['hand_xy_px'][f,i].astype(np.float32);score=raw['raw_score'][f,i]
            out['xy_px'][f,sl]=pts;out['observed'][f,sl]=True;out['hand_score'][f,side]=score
            out['usable'][f,sl]=np.isfinite(pts).all(1)&(pts[:,0]>=-margin*w)&(pts[:,0]<=w*(1+margin))&(pts[:,1]>=-margin*h)&(pts[:,1]<=h*(1+margin))&(score>=threshold)
    ob=Observation(**out);ob.sidecar=dict(baseline.sidecar,hand_assignment_policy='frozen_joint_v3',assignment_sha256=assigner_cls.identity())
    ob.validate();return ob,sides,trace

def capture():
    from wis_lab.capture.video_source import extract
    plan=read(U/'PLAN.json');pins_ok(plan['pins'])
    if (U/'CAPTURE_RESULT.json').exists():
        for r in read(U/'CAPTURE_RESULT.json')['rows']:pins_ok(r['outputs'])
        print('Verified completed capture reused; no detector rerun.',flush=True)
        return
    capmod=load_module('roi_frozen_raw_capture',P/'capture_raw.py')
    assignment=load_module('roi_frozen_joint_v3',J/'assignment.py')
    contract=read(LAB/'config/contract.json');start=time.perf_counter();rows=[]
    for it in plan['items']:
        cid=it['clip_id'];d=U/'sensor_crop'/cid
        if (d/'SUCCESS.json').exists():
            done=read(d/'SUCCESS.json');assert done['plan_sha256']==sha256_file(U/'PLAN.json');pins_ok(done['outputs']);rows.append(done);continue
        assert not d.exists(),f'Preserve partial evidence before recovery: {d}'
        d.mkdir(parents=True);event(stage='capture_start',clip=cid)
        lm=capmod.RawCaptureLandmarker(contract);token=lm.instance_token
        try:ob=extract(it['video_path'],capmod.CropProxy(lm,it['bottom_box']),cid)
        finally:lm.close()
        raw={k:np.stack([r[k] for r in lm.raw_rows]) for k in lm.raw_rows[0]}
        capmod.verify_capture(raw,ob,contract)
        ob.sidecar.update(source_bbox_px=it['bottom_box'],width=it['bottom_box'][2],height=it['bottom_box'][3],detector_instance=token)
        ob.save(str(d/'baseline.npz'))
        with (d/'raw.npz').open('xb') as f:np.savez_compressed(f,**raw)
        candidate,sides,trace=associate(raw,ob,assignment.HandAssigner,contract)
        candidate.save(str(d/'candidate.npz'));np.savez_compressed(d/'assignment.npz',sides=sides);write(d/'trace.json',trace)
        native=Observation.load(it['cached']['native']['observation'])
        assert np.array_equal(ob.frame_id,native.frame_id) and np.array_equal(ob.timestamp_us,native.timestamp_us)
        done={'clip_id':cid,'frames':ob.n_frames,'plan_sha256':sha256_file(U/'PLAN.json'),'sensor_box':it['bottom_box'],
            'detector_instance':token,'source_sha256':it['source_sha256'],'outputs':{str(p):sha256_file(p) for p in d.iterdir() if p.is_file()}}
        write(d/'SUCCESS.json',done);rows.append(done);event(stage='capture_done',clip=cid,frames=ob.n_frames)
    pins_ok(plan['pins'])
    assert len(set(r['detector_instance'] for r in rows))==20
    write(U/'CAPTURE_RESULT.json',{'conditions':20,'frames':sum(r['frames'] for r in rows),'seconds':time.perf_counter()-start,'rows':rows,'training':0})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','capture']);args=parser.parse_args()
    try:globals()[args.action]()
    except Exception:
        with (U/'FIRST_ERRORS.log').open('a',encoding='utf-8') as f:traceback.print_exc(file=f)
        raise
