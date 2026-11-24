"""Fixed additional TRAIN8 raw capture; no candidate replay/classifier/training.

Run with python -B capture_broadened.py prepare, then run. All files are written
inside this new unit. Original archives/observations and earlier units are read-only.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
import zipfile
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np

U=Path(__file__).resolve().parent
LAB=U.parents[1];ROOT=LAB.parents[1]
P=U.with_name('word100_hand_assignment_20261016')
MANIFEST=U.with_name('word100_training_20261013')/'MANIFEST.json'
PRIOR_PLAN=U.with_name('word100_upperbody_preflight_20261014')/'PLAN.json'
RECORDS=Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013/records')
PLAN=U/'BROADENED_TRAIN_PLAN.json';CTX=U/'BROADENED_CAPTURE_CONTEXT.json'
OUT=U/'broadened_raw_capture';SOURCES=U/'source_clips'


def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write_new(p,value):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def log(**row):
    row.update(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid())
    with (U/'BROADENED_CAPTURE_PROGRESS.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def prepare():
    assert (U/'SCOPE.json').is_file()
    assert not PLAN.exists() and not CTX.exists(),'Plan/context immutable; verified recovery required'
    old={r['word_id'] for r in read(PRIOR_PLAN)['items']}
    eligible=[r for r in read(MANIFEST) if r['role']=='train_candidate'
        and r['signer_id'] in [2,3] and r['word_code'] not in old]
    seed='word100_hand_assignment_joint_20261016:broadened_train:'
    codes=sorted({r['word_code'] for r in eligible},key=lambda c:hashlib.sha256((seed+c).encode()).hexdigest())[:8]
    assert codes==['WORD2348','WORD2042','WORD1844','WORD1675','WORD2721','WORD0766','WORD1519','WORD0651']
    items=[];pins={};archive_stat={}
    for i,code in enumerate(codes):
        signer=2+i%2
        row=next(r for r in eligible if r['word_code']==code and r['signer_id']==signer)
        cid=row['clip_id'];rp=RECORDS/(cid+'.json');record=read(rp)
        op=Path(record['observation']);sp=op.with_suffix('.sidecar.json');side=read(sp)
        assert sha(op)==record['observation_sha256'];z=load(op)
        good=z['usable'][:,[11,12,23,24]].all(1);assert good.any()
        shoulder=np.median(z['xy_px'][good][:,[11,12],:].mean(1),axis=0)
        hip=np.median(z['xy_px'][good][:,[23,24],:].mean(1),axis=0)
        w,h=int(side['width']),int(side['height']);assert hip[1]>shoulder[1]
        bottom=int(round(shoulder[1]+.75*(hip[1]-shoulder[1])));assert 0<bottom<h
        crop_width=int(round(bottom*w/h));x0=max(0,min(w-crop_width,int(round(shoulder[0]-crop_width/2))))
        box=[x0,0,x0+crop_width,bottom]
        entry=row['video_entries'][0];ap=Path(entry['archive']);st=ap.stat()
        archive_stat[str(ap)]=[st.st_size,st.st_mtime_ns]
        with zipfile.ZipFile(ap) as archive:
            zi=archive.getinfo(entry['member']);assert zi.CRC==entry['crc32'] and zi.file_size==entry['bytes']
        for path in [rp,op,sp]:pins[str(path)]=sha(path)
        items.append(dict(clip_id=cid,word_id=code,label=row['canonical_label'],signer_id=signer,
            source_role='train_candidate',source_archive=str(ap),source_member=entry['member'],
            source_bytes=entry['bytes'],source_crc32=entry['crc32'],source_sha256=record['source_sha256'],
            video_path=str(SOURCES/(cid+'.mp4')),observation=str(op),observation_sha256=record['observation_sha256'],
            frames=record['frames'],original_size=[w,h],box=box,core_seconds=row['intervals'][0],
            reference_shoulder=shoulder.tolist(),reference_hip=hip.tolist(),
            selection_digest=hashlib.sha256((seed+code).encode()).hexdigest()))
    fixed=[Path(__file__),U/'SCOPE.json',U/'POINT_CONTRACT.json',U/'PROTECTED_FILES.json',MANIFEST,PRIOR_PLAN,
        P/'capture_raw.py',LAB/'config/contract.json',LAB/'src/wis_lab/capture/landmarker.py',
        LAB/'src/wis_lab/capture/video_source.py',LAB/'src/wis_lab/core/observation.py',
        ROOT/'Tools/models/hand_landmarker.task',ROOT/'Tools/models/pose_landmarker_lite.task']
    pins.update({str(p):sha(p) for p in fixed})
    write_new(PLAN,dict(status='FROZEN_BEFORE_SOURCE_COPY_AND_DETECTOR_CAPTURE',
        selection='Sort88 distinct TRAIN word codes excluding prior12 by SHA256(fixedseed+wordcode); first8; alternate signer02/03 four each. No candidate results used.',
        selection_seed=seed,eligible_word_ids=88,conditions=['native','crop'],items=items,pins=pins,
        archive_stat=archive_stat,total_source_bytes=sum(r['source_bytes'] for r in items),
        total_frame_conditions=2*sum(r['frames'] for r in items),
        limitation='Additional TRAIN engineering regression fixtures, not unseen-user recognition evaluation.',
        crop_rule='Same fixed original shoulder/hip median rule as prior PLAN, no detector outcome-dependent crop',
        classifier_inference=0,training=0,candidate_replay=0))
    SOURCES.mkdir(exist_ok=True)
    for item in items:
        dest=Path(item['video_path']);assert dest.resolve().parent==SOURCES.resolve()
        assert not dest.exists(),'No source overwrite'
        with zipfile.ZipFile(item['source_archive']) as archive:
            zi=archive.getinfo(item['source_member'])
            assert zi.CRC==item['source_crc32'] and zi.file_size==item['source_bytes']
            with archive.open(zi) as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst,1048576)
        assert dest.stat().st_size==item['source_bytes'] and sha(dest)==item['source_sha256']
        pins[str(dest)]=sha(dest)
    pins[str(PLAN)]=sha(PLAN)
    check(pins);check(read(U/'PROTECTED_FILES.json'))
    write_new(CTX,dict(status='FROZEN_BEFORE_RAW_CAPTURE',pins=pins,archive_stat=archive_stat,
        conditions=['native','crop'],items=items,source_copy_bytes=sum(r['source_bytes'] for r in items),
        fresh_detectors_per_condition=True,raw_schema='wis_raw_hand_detections_v1',
        raw_capture_implementation=str(P/'capture_raw.py'),source_role='TRAIN02/03 only',
        raw_hand_float64=True,baseline_pose_float32=True,bytecode_writing_disabled=True))
    log(event='prepare_complete',source_clips=8,conditions=16,bytes=sum(r['source_bytes'] for r in items),pins=len(pins))


def run():
    assert not (U/'BROADENED_CAPTURE_RESULT.json').exists(),'Reuse completed raw captures'
    context=read(CTX);context_hash=sha(CTX);check(context['pins']);check(read(U/'PROTECTED_FILES.json'))
    spec=importlib.util.spec_from_file_location('readonly_raw_capture_library',P/'capture_raw.py')
    lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    # Import executes no capture; old module's main is guarded. -B prevents prior-unit bytecode writes.
    contract=read(LAB/'config/contract.json');rows=[];start=time.perf_counter()
    OUT.mkdir(exist_ok=True);log(event='capture_start',conditions=16,context_sha256=context_hash)
    for item in context['items']:
        cid=item['clip_id'];pair={}
        for mode in context['conditions']:
            dest=OUT/cid/mode;success=dest/'SUCCESS.json'
            if success.exists():
                result=read(success);assert result['context_sha256']==context_hash;check(result['output_sha256'])
                pair[mode]=lib.Observation.load(str(dest/'baseline.npz'));rows.append(result)
                log(event='reuse_success',clip_id=cid,mode=mode,completed=len(rows));continue
            assert not dest.exists(),f'Partial output preserved for verified recovery: {dest}'
            dest.mkdir(parents=True);st=time.perf_counter()
            log(event='item_start',clip_id=cid,label=item['label'],mode=mode)
            with lib.RawCaptureLandmarker(contract) as lm:
                assert lm._last_ms==-1 and not lm.raw_rows
                token=lm.instance_token
                interface=lib.CropProxy(lm,item['box']) if mode=='crop' else lm
                baseline=lib.extract(item['video_path'],interface,cid)
                raw={k:np.stack([r[k] for r in lm.raw_rows]) for k in lm.raw_rows[0]}
            if mode=='crop':
                x0,y0,x1,y1=item['box'];baseline.sidecar.update(width=x1-x0,height=y1-y0)
            lib.verify_capture(raw,baseline,contract)
            assert baseline.n_frames==item['frames']
            baseline.sidecar.update(broadened_capture_context_sha256=context_hash,mode=mode,
                source_bbox_px=item['box'] if mode=='crop' else [0,0,*item['original_size']],
                original_size=item['original_size'],source_archive=item['source_archive'],
                source_member=item['source_member'],detector_instance_token=token)
            baseline.sidecar['quality']=baseline.quality_summary();baseline.save(str(dest/'baseline.npz'))
            with (dest/'raw.npz').open('xb') as f:np.savez_compressed(f,schema=np.array('wis_raw_hand_detections_v1'),**raw)
            equal=None
            if mode=='native':
                original=lib.Observation.load(item['observation'])
                equal={k:lib.equal_array(getattr(original,k),getattr(baseline,k))
                    for k in ['xy_px','conf','observed','usable','hand_score','timestamp_us','frame_id']}
            sidecar=dict(schema='wis_raw_hand_detections_v1',clip_id=cid,label=item['label'],mode=mode,
                signer_id=item['signer_id'],source_role=item['source_role'],context_sha256=context_hash,
                source_sha256=item['source_sha256'],source_path=item['video_path'],
                raw_sha256=sha(dest/'raw.npz'),frames=baseline.n_frames,width=baseline.sidecar['width'],height=baseline.sidecar['height'],
                box=baseline.sidecar['source_bbox_px'],original_baseline_reconstruction_exact=True,
                existing_native_observation_equal=equal,raw_detections=int(raw['raw_count'].sum()),
                zero_hand_frames=int((raw['raw_count']==0).sum()),detector_instance_token=token,process_id=os.getpid(),
                label_mapping=dict(Left=0,Right=1,unknown_or_padding=-1),raw_hand_precision='float64',
                score_note='Raw handedness score, not per-joint confidence',time_axis='Original frame IDs and PTS microseconds preserved')
            write_new(dest/'raw.sidecar.json',sidecar)
            result=dict(clip_id=cid,label=item['label'],signer_id=item['signer_id'],mode=mode,status='COMPLETE',
                context_sha256=context_hash,frames=baseline.n_frames,raw_detections=sidecar['raw_detections'],
                zero_hand_frames=sidecar['zero_hand_frames'],seconds=time.perf_counter()-st,
                detector_instance_token=token,original_baseline_reconstruction_exact=True,
                existing_native_observation_equal=equal,
                output_sha256={str(dest/k):sha(dest/k) for k in ['raw.npz','raw.sidecar.json','baseline.npz','baseline.sidecar.json']})
            write_new(success,result);rows.append(result);pair[mode]=baseline
            log(event='item_complete',clip_id=cid,mode=mode,completed=len(rows),total=16,seconds=result['seconds'])
        for k in ['timestamp_us','frame_id']:assert np.array_equal(getattr(pair['native'],k),getattr(pair['crop'],k))
    check(context['pins']);check(read(U/'PROTECTED_FILES.json'));assert sha(CTX)==context_hash
    for p,stat in context['archive_stat'].items():
        s=Path(p).stat();assert [s.st_size,s.st_mtime_ns]==stat
    summary=dict(status='RAW_CAPTURE_COMPLETE_NO_CANDIDATE_REPLAY',source_clips=8,conditions=16,
        context_sha256=context_hash,rows=rows,frames=sum(r['frames'] for r in rows),
        raw_detections=sum(r['raw_detections'] for r in rows),zero_hand_frames=sum(r['zero_hand_frames'] for r in rows),
        native_all_arrays_exact=sum(all(r['existing_native_observation_equal'].values()) for r in rows if r['mode']=='native'),
        elapsed_seconds=time.perf_counter()-start,protected_files_unchanged=True,
        classifier_inference=0,training=0,candidate_replay=0)
    write_new(U/'BROADENED_CAPTURE_RESULT.json',summary)
    log(event='capture_complete',conditions=16,frames=summary['frames'],seconds=summary['elapsed_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    try:prepare() if args.action=='prepare' else run()
    except Exception as exc:
        path=U/'BROADENED_CAPTURE_FIRST_FAILURE.json'
        if not path.exists():write_new(path,dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
