"""One pre-registered public-TRAIN ROI candidate. No training or inference."""
from pathlib import Path
import sys,json,os,time,argparse,traceback
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent
PRIOR=U.with_name('word100_wrist_roi_pair_20261017')
sys.path.insert(0,str(PRIOR))
import run_wrist_pair as prev
base=prev.base; Observation=base.Observation
read=base.read; write=base.write; sha=base.sha256_file; pins_ok=base.pins_ok
apply_roi=prev.apply_roi; build_roi_channels=prev.build_roi_channels
from stable_rest import calibrate_rest,suppress_rest_fragments

def event(**data):
    data.update(pid=os.getpid(),time=time.time())
    with (U/'PROGRESS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n')
    print(json.dumps(data,ensure_ascii=False),flush=True)

def prepare():
    assert not (U/'PLAN.json').exists()
    old=read(PRIOR/'PLAN.json'); pins=dict(old['pins'])
    pins.update({str(PRIOR/p):h for p,h in read(PRIOR/'COMPLETION.json')['artifact_pins'].items()})
    for p in [Path(__file__),U/'stable_rest.py',U/'SCOPE.json',U/'test_stable_rest.py',PRIOR/'COMPLETION.json']:
        pins[str(p)]=sha(p)
    pins_ok(pins); rows=[]
    for r in old['items']:
        row=dict(r); cid=r['clip_id']; ob=Observation.load(r['cached']['native']['observation'])
        cal=calibrate_rest(ob,*r['original_size']); d=U/'outputs'/cid;d.mkdir(parents=True)
        write(d/'calibration.json',cal)
        row.update(calibration=cal,calibration_path=str(d/'calibration.json'),
                   old_models={m:str(PRIOR/'outputs'/cid/m/'model_input.npz') for m in ['wrist_full','wrist_sensor']})
        if cal['status']=='CALIBRATED':row['stable_box']=cal['box_source_pixels']
        pins[str(d/'calibration.json')]=sha(d/'calibration.json');rows.append(row)
    write(U/'PLAN.json',dict(status='FROZEN_SINGLE_CANDIDATE',scope=read(U/'SCOPE.json'),pins=pins,items=rows,
        calibrated=sum(r['calibration']['status']=='CALIBRATED' for r in rows),
        excluded=[r['clip_id'] for r in rows if r['calibration']['status']!='CALIBRATED']))
    event(stage='prepared',calibrated=sum('stable_box' in r for r in rows),total=len(rows))

def capture():
    from wis_lab.capture.video_source import extract
    plan=read(U/'PLAN.json');pins_ok(plan['pins'])
    if (U/'CAPTURE_RESULT.json').exists():
        for r in read(U/'CAPTURE_RESULT.json')['rows']:pins_ok(r['outputs'])
        return
    cap=base.load_module('stable_frozen_capture',base.P/'capture_raw.py')
    ass=base.load_module('stable_frozen_assigner',base.J/'assignment.py')
    contract=read(base.LAB/'config/contract.json');rows=[];start=time.perf_counter()
    for it in plan['items']:
        if 'stable_box' not in it:continue
        cid=it['clip_id'];d=U/'sensor_crop'/cid
        if (d/'SUCCESS.json').exists():
            rec=read(d/'SUCCESS.json');assert rec['plan_sha256']==sha(U/'PLAN.json');pins_ok(rec['outputs']);rows.append(rec);continue
        assert not d.exists(),f'Preserve partial failure: {d}'
        d.mkdir(parents=True);event(stage='capture_start',clip=cid,box=it['stable_box'])
        if it['stable_box']==it['wrist_box']:
            oldrec=read(PRIOR/'sensor_crop'/cid/'SUCCESS.json');pins_ok(oldrec['outputs'])
            rec=dict(clip_id=cid,frames=oldrec['frames'],plan_sha256=sha(U/'PLAN.json'),reused=True,
                     observation=str(PRIOR/'sensor_crop'/cid/'candidate.npz'),raw=str(PRIOR/'sensor_crop'/cid/'raw.npz'),
                     outputs=oldrec['outputs'],sensor_box=it['stable_box'],detector_instance=oldrec['detector_instance'])
        else:
            lm=cap.RawCaptureLandmarker(contract);token=lm.instance_token
            try:ob=extract(it['video_path'],cap.CropProxy(lm,it['stable_box']),cid)
            finally:lm.close()
            raw={k:np.stack([r[k] for r in lm.raw_rows]) for k in lm.raw_rows[0]}
            cap.verify_capture(raw,ob,contract)
            ob.sidecar.update(source_bbox_px=it['stable_box'],width=it['stable_box'][2],height=it['stable_box'][3],detector_instance=token)
            ob.save(str(d/'baseline.npz'))
            with (d/'raw.npz').open('xb') as f:np.savez_compressed(f,**raw)
            candidate,sides,trace=base.associate(raw,ob,ass.HandAssigner,contract)
            candidate.save(str(d/'candidate.npz'));np.savez_compressed(d/'assignment.npz',sides=sides);write(d/'trace.json',trace)
            native=Observation.load(it['cached']['native']['observation'])
            assert np.array_equal(ob.frame_id,native.frame_id) and np.array_equal(ob.timestamp_us,native.timestamp_us)
            rec=dict(clip_id=cid,frames=ob.n_frames,plan_sha256=sha(U/'PLAN.json'),reused=False,
                     observation=str(d/'candidate.npz'),raw=str(d/'raw.npz'),sensor_box=it['stable_box'],detector_instance=token,
                     source_sha256=it['source_sha256'],outputs={str(p):sha(p) for p in d.iterdir() if p.is_file()})
        write(d/'SUCCESS.json',rec);rows.append(rec);event(stage='capture_done',clip=cid,reused=rec['reused'])
    pins_ok(plan['pins'])
    write(U/'CAPTURE_RESULT.json',dict(conditions=len(rows),new_capture=sum(not r['reused'] for r in rows),
        reused=sum(r['reused'] for r in rows),seconds=time.perf_counter()-start,rows=rows,training=0))

def compare():
    assert not (U/'COMPARISON.json').exists(),'Reuse completed results'
    plan=read(U/'PLAN.json');pins_ok(plan['pins'])
    captures={r['clip_id']:r for r in read(U/'CAPTURE_RESULT.json')['rows']};rows=[]
    for it in plan['items']:
        cid=it['clip_id']
        if 'stable_box' not in it:continue
        rec=captures[cid];pins_ok(rec['outputs']);outpins={};obs={k:Observation.load(p) for k,p in it['old_models'].items()}
        guards={}
        for source,path in [('full',it['cached']['native']['observation']),('sensor',rec['observation'])]:
            raw=apply_roi(Observation.load(path),it['stable_box'])
            guarded,trace=suppress_rest_fragments(raw,it['calibration']);guards[source]=trace['suppressed_frame_counts']
            for kind,o in [('raw',raw),('guarded',guarded)]:
                mode=f'stable_{source}_{kind}';d=U/'outputs'/cid/mode;d.mkdir(parents=True)
                o.save(str(d/'model_input.npz'));ch=build_roi_channels(o);x=ch['x']
                assert x.shape==(o.n_frames,280) and np.isfinite(x).all()
                np.savez_compressed(d/'channels.npz',x=x,frame_id=o.frame_id,timestamp_us=o.timestamp_us)
                if kind=='guarded':write(d/'guardtrace.json',trace)
                obs[mode]=o;outpins.update({str(p):sha(p) for p in d.iterdir() if p.is_file()})
        pairdefs=[('full_boundary','wrist_full','stable_full_raw'),('sensor_boundary','wrist_sensor','stable_sensor_raw'),
                  ('full_guard','stable_full_raw','stable_full_guarded'),('sensor_guard','stable_sensor_raw','stable_sensor_guarded'),
                  ('full_combined','wrist_full','stable_full_guarded'),('sensor_combined','wrist_sensor','stable_sensor_guarded')]
        pairs={k:prev.diff(obs[a],obs[b]) for k,a,b in pairdefs}
        rows.append(dict(clip_id=cid,label=it['label'],frames=raw.n_frames,old_box=it['wrist_box'],new_box=it['stable_box'],
                         calibration=it['calibration'],guards=guards,metrics={k:prev.extra_metrics(o) for k,o in obs.items()},pairs=pairs,outputs=outpins))
        event(stage='compared',clip=cid,guards=guards)
    aggregate={};ends={}
    if rows:
        for pair in rows[0]['pairs']:
            aggregate[pair]={k:{f:sum(r['pairs'][pair][s][k][f] for r in rows for s in ['left','right']) for f in ['old','new','gain','loss']} for k in ['any','palm5','all21','points']}
        for mode in plan['scope']['conditions']:
            ends[mode]={}
            for win in ['first5','last5','first_half_second','last_half_second']:
                rr=[dict(clip_id=r['clip_id'],label=r['label'],side=s,**r['metrics'][mode]['end_windows'][win]['hands'][s]) for r in rows for s in ['left','right'] if r['metrics'][mode]['end_windows'][win]['hands'][s]['points']>0]
                ends[mode][win]=dict(total_hand_point_frames=sum(q['points'] for q in rr),nonempty_clip_sides=len(rr),rows=rr)
    pins_ok(plan['pins'])
    write(U/'COMPARISON.json',dict(status='COMPUTED_PENDING_INDEPENDENT_VISUAL_REVIEW',requested_clips=20,calibrated_clips=len(rows),
        not_calibrated=plan['excluded'],conditions=6*len(rows),unique_video_frames=sum(r['frames'] for r in rows),
        aggregate=aggregate,end_window_audit=ends,rows=rows,training=0,classifier_inference=0,production_changes=0,
        limits='Same engineering cases. Observation metrics not truth/recognition. Excluded calibration failures are explicit; no fallback. Sensor calibration is offline and may use full-source wrists.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','capture','compare']);args=p.parse_args()
    try:globals()[args.action]()
    except Exception:
        with (U/'FIRST_ERRORS.log').open('a',encoding='utf-8') as f:traceback.print_exc(file=f)
        raise
