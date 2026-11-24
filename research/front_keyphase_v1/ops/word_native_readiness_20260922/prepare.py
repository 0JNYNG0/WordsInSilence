"""Bounded candidate acquisition with existing native reader and input adapter."""
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib, json, math, re, sys, time
import numpy as np

H = Path(__file__).resolve().parent
L = H.parents[1]
P = L.parent / 'sign_model_pilot_v1'
OLD = L / 'ops/word_keypoint_archive_proposal_20260922'
CACHE = L / 'runs/word_real02_native_candidates_v1'
sys.path[:0] = [str(P), str(P/'src')]
import expand_build as existing
from pilot.data import build_channels

read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')

def native_clip_id(ident):
    m = re.match(r'(NIA_SL_(?:SEN|WORD)\d+_REAL\d+_[A-Z])', ident)
    return m.group(1) if m else ident

def main():
    started = time.monotonic()
    original = read(OLD/'INSPECTION_PLAN.json')
    previous_audit = read(OLD/'INDEPENDENT_AUDIT.json')
    assert previous_audit['status'] == 'PASS_DIRECTORY_AND_TARGET_LOCAL_HEADER_AGREEMENT'
    for name in ['TARGET_MEMBER_INDEX.json', 'INSPECTION_PLAN.json']:
        assert sha(OLD/name) == previous_audit['input_artifact_hashes'][name]
    assert sha(original['source_label_audit']) == original['source_label_audit_sha256']
    labels = read(original['source_label_audit'])['source_rows']
    labels = sorted([r for r in labels if r['signer_id'] == 2], key=lambda r:r['clip_id'])
    assert len(labels) == 16 and all(r['split_role'] == 'train_candidate' for r in labels)
    assert all(r['semantic_target_id'] is None and not r['training_approved'] for r in labels)
    index = read(OLD/'TARGET_MEMBER_INDEX.json')
    archive = Path(index['archive']); before = archive.stat()
    assert (before.st_size,before.st_mtime_ns) == (original['archive_bytes'],original['archive_mtime_ns'])
    by_clip = defaultdict(list)
    for member in index['members']:
        by_clip[member['clip_id']].append(member)
    assert set(by_clip) == {r['clip_id'] for r in labels}
    policy = read(L/'ops/partial_encoder_pair_20260922/comparison_policy.json')
    meta_path = Path(policy['paths']['metadata'])
    assert sha(meta_path) == policy['pins'][str(meta_path)]
    meta = read(meta_path)
    names = {r['word_id']:r['display_name'] for r in read(L/'config/words.json')['words']}
    pin_paths = [Path(existing.__file__),Path(existing.PREP)/'zip_probe.py',P/'src/pilot/data.py',meta_path,OLD/'TARGET_MEMBER_INDEX.json',Path(original['source_label_audit'])]
    pins = {str(p):sha(p) for p in pin_paths}
    assert pins[str(Path(existing.PREP)/'zip_probe.py')] == original['reader_sha256']
    plan = {'at_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'user_instruction':'그래서 이 다운로드 받은 데이터로 뭘 할 계획이고, 기댓값은 뭐고, 부족한 게 있다면 뭔지 등등 세부내용을 설명해줘. 그리고 작업 진행해줘.',
        'scope':'Existing16 REAL02 front candidates only; CRC/decode, full and annotated native input preparation, metadata overlap; no training or inference',
        'source_pins':pins,'archive':str(archive),'archive_bytes':before.st_size,'archive_mtime_ns':before.st_mtime_ns,
        'clips':[r['clip_id'] for r in labels],'semantic_mapping_approved':False,
        'timebase':'Existing nominal30fps contract; original video PTS/alignment not verified',
        'new_frame_selection':'Use previous1660-member index; no directory rescan or other views/signers',
        'forbidden':'No interpolation, smoothing, hand dropping, augmentation, label merging, model changes, user/held-out arrays or new downloads',
        'output':str(CACHE)}
    write(H/'PLAN.json',plan)
    CACHE.mkdir(exist_ok=False)
    target_names = {names[t] for r in labels for t in r['target_candidates_not_approved']}
    overlap = {'scope':'Current2122 base and224 context metadata, not a claim about every historical model/data lineage',
        'base_rows':len(meta['base']),'context_rows':len(meta['pair_context']),
        'existing_native_base_signers':sorted({r['signer_id'] for r in meta['base'] if 'signer_id' in r}),
        'source_clip_id_intersection_by_group':{g:sorted(set(by_clip)&{native_clip_id(r['id']) for r in meta[g]}) for g in ['base','pair_context','annotation_dev','whole_dev']},
        'per_target':[], 'new_person_added':False,
        'exact_content_duplicate_check':'UNVERIFIED across different source IDs; coordinate/source-ID difference alone does not establish independent physical performances'}
    for name in sorted(target_names):
        base = [r for r in meta['base'] if r['label'] == name]
        overlap['per_target'].append({'target_name':name,'existing_native_train_rows':sum(not r['id'].startswith('NIKL_') for r in base),
            'existing_nikl_train_rows':sum(r['id'].startswith('NIKL_') for r in base),
            'existing_train_signers':sorted({r['signer_id'] for r in base if 'signer_id' in r}),
            'existing_dev_rows':sum(r['label']==name for r in meta['annotation_dev']),
            'new_candidate_word_ids':[r['word_number'] for r in labels if any(names[t]==name for t in r['target_candidates_not_approved'])]})
    write(H/'SOURCE_OVERLAP.json',overlap)
    rows=[]; failures=[]
    with archive.open('rb') as handle:
        for label in labels:
            ident=label['clip_id']; members=sorted(by_clip[ident],key=lambda m:m['frame'])
            record={k:label[k] for k in ['clip_id','word_number','raw_label','signer_id','split_role','target_candidates_not_approved','annotation_start_s','annotation_end_s','duration_s']}
            record.update(training_approved=False,semantic_target_id=None)
            try:
                frames=np.array([m['frame'] for m in members],dtype=np.int64)
                assert np.array_equal(frames,np.arange(len(frames)))
                arr=np.stack([existing.decode(handle,m) for m in members])
                assert arr.shape==(len(frames),67,3) and np.isfinite(arr).all()
                present=np.isfinite(arr).all(-1)&~np.all(arr[...,:2]==0,axis=-1)
                ts=np.rint(frames/existing.FPS*1e6).astype(np.int64)
                lo=math.ceil(label['annotation_start_s']*existing.FPS-1e-9)
                hi=math.ceil(label['annotation_end_s']*existing.FPS-1e-9)
                take=np.flatnonzero((frames>=lo)&(frames<hi))
                assert len(take)>=6 and np.array_equal(frames[take],np.arange(lo,hi))
                full=build_channels(arr[...,:2],present,ts)
                cut=build_channels(arr[take,:,:2],present[take],ts[take])
                assert full['x'].shape==(len(frames),280) and cut['x'].shape==(len(take),280)
                assert np.isfinite(full['x']).all() and np.isfinite(cut['x']).all()
                path=CACHE/(ident+'_candidate.npz')
                with path.open('xb') as f:
                    np.savez_compressed(f,xy_native=arr[...,:2],confidence_native=arr[...,2],coordinate_present=present,
                        frame_index=frames,timestamp_nominal_us=ts,annotation_rows=take,x_full=full['x'],x_annotation=cut['x'])
                hands={}
                for side,start,body_wrist,other_wrist in [('left',25,7,4),('right',46,4,7)]:
                    ok=present[:,start]&present[:,body_wrist]&present[:,other_wrist]
                    distance=np.linalg.norm(arr[:,start,:2]-arr[:,body_wrist,:2],axis=1)
                    opposite=np.linalg.norm(arr[:,start,:2]-arr[:,other_wrist,:2],axis=1)
                    hands[side]={'all21_present_frames':int(present[:,start:start+21].all(1).sum()),
                        'no_coordinates_frames':int((~present[:,start:start+21].any(1)).sum()),
                        'wrist_pair_check_frames':int(ok.sum()),'wrist_nearer_same_body_side_frames':int(((distance<opposite)&ok).sum()),
                        'full_hand_scale':float(full['hand_scales'][side]) if np.isfinite(full['hand_scales'][side]) else None}
                shoulder_ok=present[:,5]&present[:,2]
                raw_hash=hashlib.sha256(arr.tobytes()).hexdigest()
                record.update(status='PASS_NUMERIC_INPUT_ONLY',npz=str(path),npz_sha256=sha(path),npz_bytes=path.stat().st_size,
                    frames=len(frames),annotation_frames=len(take),crc_checked_frames=len(members),
                    native_array_sha256=raw_hash,confidence_min_max=[float(arr[...,2].min()),float(arr[...,2].max())],
                    confidence_constant=bool(np.ptp(arr[...,2])==0),coordinate_present_fraction=float(present.mean()),
                    shoulders_present_frames=int(shoulder_ok.sum()),
                    anatomical_left_shoulder_right_on_image_frames=int((shoulder_ok&(arr[:,5,0]>arr[:,2,0])).sum()),
                    hands=hands,body_scale_full=full['body_scale'],body_scale_annotation=cut['body_scale'],
                    native_last_frame_s=float(frames[-1]/existing.FPS),
                    native_time_origin_and_video_alignment='UNVERIFIED',web_mediapipe_equivalence='UNVERIFIED')
            except Exception as exc:
                record.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
                failures.append({'clip_id':ident,'error':record['error']})
            rows.append(record)
    assert (archive.stat().st_size,archive.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns)
    assert pins=={str(p):sha(p) for p in pin_paths}
    passed=[r for r in rows if r['status']=='PASS_NUMERIC_INPUT_ONLY']
    hashes=defaultdict(list)
    for r in passed:hashes[r['native_array_sha256']].append(r['clip_id'])
    result={'status':'PREPARED_PENDING_INDEPENDENT_AUDIT' if len(passed)==16 and not failures else 'INCOMPLETE',
        'rows':rows,'failures':failures,'counts':{'clips':len(passed),'full_frames':sum(r['frames'] for r in passed),
        'annotation_frames':sum(r['annotation_frames'] for r in passed),'cache_bytes':sum(r['npz_bytes'] for r in passed)},
        'exact_coordinate_duplicates_within_new_candidates':[v for v in hashes.values() if len(v)>1],
        'source_pins_unchanged':True,'archive_unchanged':True,'seconds':time.monotonic()-started,
        'training_approved':False,'training_runs':0,'model_inference':0,'user_arrays_read':0,'reserved_arrays_read':0,
        'limits':['Numeric contract pass is not coordinate accuracy or semantic equivalence','Confidence can be constant and is not observation quality','Source-provided coordinates are not new MediaPipe detections','Nominal frame-time convention retained without PTS verification']}
    write(H/'PREPARATION.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','limits']},ensure_ascii=False),flush=True)
    return 0 if not failures and len(passed)==16 else 1

if __name__=='__main__':
    raise SystemExit(main())
