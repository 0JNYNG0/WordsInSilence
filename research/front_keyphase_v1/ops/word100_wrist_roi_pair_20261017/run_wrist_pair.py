"""Frozen resting-wrist ROI comparison, public TRAIN only, no model execution."""
from pathlib import Path
import sys, json, os, time, argparse, importlib.util, traceback
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent
OLD=U.with_name('word100_fullframe_roi_pair_20261016')
sys.path.insert(0,str(OLD))
import run_pair as base
from roi_adapter import apply_roi, build_roi_channels
from compare_pair import metrics,flags
read=base.read;write=base.write;pins_ok=base.pins_ok;sha=base.sha256_file
Observation=base.Observation
def event(**data):
    data.update(pid=os.getpid(),time=time.time())
    with (U/'PROGRESS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n')
    print(json.dumps(data,ensure_ascii=False),flush=True)

def prepare():
    assert not (U/'PLAN.json').exists()
    previous=read(OLD/'PLAN.json');pins=dict(previous['pins']);rows=[]
    for r in previous['items']:
        row=dict(r);ob=Observation.load(r['cached']['native']['observation'])
        assert ob.n_frames>=5 and ob.usable[:5,[11,12,33,54]].all()
        xy=ob.xy_px;w,h=r['original_size']
        med=np.median(xy[:5,[33,54],1],axis=0)
        width=float(np.median(np.linalg.norm(xy[:5,11]-xy[:5,12],axis=1)))
        b=int(np.floor(min(med)-.02*width))
        assert r['bottom_box'][3]<b<h
        row.update(wrist_box=[0,0,w,b],wrist_median_y=med.tolist(),shoulder_width_px=width,
                   calibration_frames=[0,1,2,3,4],calibration_wrist_y_range=np.ptp(xy[:5,[33,54],1],axis=0).tolist())
        row['old_models']={k:str(OLD/'outputs'/r['clip_id']/m/'model_input.npz') for k,m in [('old_full','full_then_bottom_mask'),('old_sensor','bottom_crop_then_mask')]}
        row['old_sensor_raw']=str(OLD/'sensor_crop'/r['clip_id']/'raw.npz')
        rows.append(row)
    for rec in read(OLD/'CAPTURE_RESULT.json')['rows']:pins.update(rec['outputs'])
    for rec in read(OLD/'COMPARISON.json')['rows']:pins.update(rec['outputs'])
    for p in [Path(__file__),OLD/'PLAN.json',OLD/'CAPTURE_RESULT.json',OLD/'COMPARISON.json',OLD/'roi_adapter.py',OLD/'compare_pair.py']:
        pins[str(p)]=sha(p)
    pins_ok(pins)
    write(U/'PLAN.json',dict(status='FROZEN_BEFORE_EVALUATION',items=rows,pins=pins,
        authorization='User requests compare fixed wrist-above ROI on prior problem videos, idle hand exclusion, active preservation, regressions; train only after review.',
        rule='First5 frames, each hand detector wrist y median, minimum of two - 0.02 median shoulder Euclidean width, floor. Fixed per clip. No label-specific thresholds or result-based boundary changes. Initial pose reliability explicitly audited; no fallback if invalid.',
        scope='Same20 public TRAIN engineering clips,3signers. No user/dev/test data, no recognition inference/training/production change.',
        conditions=['old_full','old_sensor','wrist_full','wrist_sensor'],
        new_detector_conditions=20,
        sensor_boundary='Crop original RGB to [0:wr_bottom,0:width] before MediaPipe VIDEO with fresh state per clip; same frozen joint_v3. A sensor-boundary simulation, not actual webcam testing.',
        validity='The ROI may be calibrated from offscreen original wrists for dataset preparation; real webcam boundary cannot read invisible hands. Resting startup must be audited, not assumed.',
        metrics=['per side usable points,any,palm5,all21','same-frame gains and losses for full mask change and actual crop change','first/last5 and0.5s visible hand evidence; human contact-sheet posture audit','fixed prior failure frames and deterministic adverse new crop examples'],
        interpretation='Same-full-source inclusion monotonicity is geometric, not detector improvement. Interior frames are not labelled active gestures. Counts are not ground truth or recognition accuracy.',
        acceptance='No lost old full-source points, exact timestamps/frame IDs and masked channel contract, report all idle leakage/gain/regression; no automatic adoption or retuning after results.',
        first_known_caveat='First5 wrist positions move notably in WORD1133; keep same rule and report rather than tuning.'))
    event(stage='prepared',clips=len(rows),pins=len(pins))

def capture():
    from wis_lab.capture.video_source import extract
    plan=read(U/'PLAN.json');pins_ok(plan['pins'])
    if (U/'CAPTURE_RESULT.json').exists():
        for r in read(U/'CAPTURE_RESULT.json')['rows']:pins_ok(r['outputs'])
        return
    cap=base.load_module('wrist_frozen_raw_capture',base.P/'capture_raw.py')
    ass=base.load_module('wrist_frozen_joint_v3',base.J/'assignment.py')
    contract=read(base.LAB/'config/contract.json');rows=[];start=time.perf_counter()
    for it in plan['items']:
        cid=it['clip_id'];d=U/'sensor_crop'/cid
        if (d/'SUCCESS.json').exists():
            done=read(d/'SUCCESS.json');assert done['plan_sha256']==sha(U/'PLAN.json');pins_ok(done['outputs']);rows.append(done);continue
        assert not d.exists(),f'Partial result must be preserved: {d}'
        d.mkdir(parents=True);event(stage='capture_start',clip=cid)
        lm=cap.RawCaptureLandmarker(contract);token=lm.instance_token
        try:ob=extract(it['video_path'],cap.CropProxy(lm,it['wrist_box']),cid)
        finally:lm.close()
        raw={k:np.stack([r[k] for r in lm.raw_rows]) for k in lm.raw_rows[0]}
        cap.verify_capture(raw,ob,contract)
        ob.sidecar.update(source_bbox_px=it['wrist_box'],width=it['wrist_box'][2],height=it['wrist_box'][3],detector_instance=token)
        ob.save(str(d/'baseline.npz'))
        with (d/'raw.npz').open('xb') as f:np.savez_compressed(f,**raw)
        candidate,sides,trace=base.associate(raw,ob,ass.HandAssigner,contract)
        candidate.save(str(d/'candidate.npz'));np.savez_compressed(d/'assignment.npz',sides=sides);write(d/'trace.json',trace)
        native=Observation.load(it['cached']['native']['observation'])
        assert np.array_equal(ob.frame_id,native.frame_id) and np.array_equal(ob.timestamp_us,native.timestamp_us)
        done=dict(clip_id=cid,frames=ob.n_frames,plan_sha256=sha(U/'PLAN.json'),sensor_box=it['wrist_box'],detector_instance=token,
                  source_sha256=it['source_sha256'],outputs={str(p):sha(p) for p in d.iterdir() if p.is_file()})
        write(d/'SUCCESS.json',done);rows.append(done);event(stage='capture_done',clip=cid,frames=ob.n_frames)
    pins_ok(plan['pins']);assert len(set(r['detector_instance'] for r in rows))==20
    write(U/'CAPTURE_RESULT.json',dict(conditions=20,frames=sum(r['frames'] for r in rows),seconds=time.perf_counter()-start,rows=rows,training=0))

def extra_metrics(ob):
    m=metrics(ob);times=ob.timestamp_us
    windows={'first5':np.arange(min(5,ob.n_frames)), 'last5':np.arange(max(0,ob.n_frames-5),ob.n_frames),
             'first_half_second':np.flatnonzero(times-times[0]<=500000),'last_half_second':np.flatnonzero(times[-1]-times<=500000)}
    m['end_windows']={}
    for n,idx in windows.items():
        m['end_windows'][n]={'frames':idx.tolist(),'hands':{}}
        for side,label in enumerate(['left','right']):
            v=ob.usable[idx,33+side*21:54+side*21]
            m['end_windows'][n]['hands'][label]=dict(points=int(v.sum()),any_frames=int(v.any(1).sum()),all21_frames=int(v.all(1).sum()))
    return m

def diff(old,new):
    assert np.array_equal(old.timestamp_us,new.timestamp_us)
    out={}
    for side,label in enumerate(['left','right']):
        a=flags(old,side);b=flags(new,side);d={}
        for k in a:
            d[k]=dict(old=int(a[k].sum()),new=int(b[k].sum()),gain=int((b[k]&~a[k]).sum()),loss=int((a[k]&~b[k]).sum()),
                      gain_frames=np.flatnonzero(b[k]&~a[k]).tolist(),loss_frames=np.flatnonzero(a[k]&~b[k]).tolist())
        sl=slice(33+side*21,54+side*21);a=old.usable[:,sl];b=new.usable[:,sl]
        d['points']=dict(old=int(a.sum()),new=int(b.sum()),gain=int((b&~a).sum()),loss=int((a&~b).sum()))
        out[label]=d
    return out

def compare():
    assert not (U/'COMPARISON.json').exists(),'Reuse completed results'
    plan=read(U/'PLAN.json');pins_ok(plan['pins'])
    for rec in read(U/'CAPTURE_RESULT.json')['rows']:pins_ok(rec['outputs'])
    rows=[]
    for it in plan['items']:
        cid=it['clip_id'];obs={k:Observation.load(p) for k,p in it['old_models'].items()};outpins={}
        for name,path in [('wrist_full',it['cached']['native']['observation']),('wrist_sensor',str(U/'sensor_crop'/cid/'candidate.npz'))]:
            ob=apply_roi(Observation.load(path),it['wrist_box']);ch=build_roi_channels(ob);x=ch['x']
            assert x.shape==(ob.n_frames,280) and np.isfinite(x).all()
            d=U/'outputs'/cid/name;d.mkdir(parents=True);ob.save(str(d/'model_input.npz'))
            np.savez_compressed(d/'channels.npz',x=x,frame_id=ob.frame_id,timestamp_us=ob.timestamp_us)
            obs[name]=ob;outpins.update({str(p):sha(p) for p in d.iterdir() if p.is_file()})
        assert not np.any(obs['old_full'].usable & ~obs['wrist_full'].usable)
        pairs={name:diff(obs[a],obs[b]) for name,a,b in [('full_boundary','old_full','wrist_full'),('sensor_boundary','old_sensor','wrist_sensor'),('wrist_source','wrist_full','wrist_sensor')]}
        rows.append(dict(clip_id=cid,label=it['label'],frames=obs['wrist_full'].n_frames,metrics={k:extra_metrics(v) for k,v in obs.items()},pairs=pairs,outputs=outpins))
        event(stage='compared',clip=cid)
    aggregate={}
    for pair in rows[0]['pairs']:
        aggregate[pair]={k:{f:sum(r['pairs'][pair][s][k][f] for r in rows for s in ['left','right']) for f in ['old','new','gain','loss']} for k in ['any','palm5','all21','points']}
    ends={}
    for mode in plan['conditions']:
        ends[mode]={}
        for win in ['first5','last5','first_half_second','last_half_second']:
            endrows=[dict(clip_id=r['clip_id'],label=r['label'],side=s,**r['metrics'][mode]['end_windows'][win]['hands'][s]) for r in rows for s in ['left','right'] if r['metrics'][mode]['end_windows'][win]['hands'][s]['points']>0]
            ends[mode][win]=dict(total_hand_point_frames=sum(q['points'] for q in endrows),nonempty_clip_sides=len(endrows),rows=endrows)
    pins_ok(plan['pins'])
    write(U/'COMPARISON.json',dict(status='COMPUTED_PENDING_INDEPENDENT_VISUAL_REVIEW',clips=20,conditions=80,unique_video_frames=sum(r['frames'] for r in rows),new_model_views=40,
        aggregate=aggregate,end_window_audit=ends,rows=rows,training=0,classifier_inference=0,production_changes=0,
        limits='Counts describe observation evidence, not detection truth/recognition. Endpoint windows are not labelled rest ground truth; manual image review required.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','capture','compare']);args=p.parse_args()
    try:globals()[args.action]()
    except Exception:
        with (U/'FIRST_ERRORS.log').open('a',encoding='utf-8') as f:traceback.print_exc(file=f)
        raise
