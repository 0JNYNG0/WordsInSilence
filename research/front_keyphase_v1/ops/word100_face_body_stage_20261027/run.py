from __future__ import annotations
import sys, json, shutil, zipfile, hashlib, time
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LAB = ROOT/'research/front_keyphase_v1'
U = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB/'src'))
from wis_lab.core.observation import Observation, sha256_file
from wis_lab.capture.face_landmarks import extract_face_video, join_face_observation, recover_face_video
from wis_lab.core.face_body_relations import relations, compare_path, REGIONS
from wis_lab.core.features import sequence_features, uniform_time_sample, body_frame
from wis_lab.core.dtw import frame_cost_matrix, _optimal_path

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def freeze():
    if (U/'FROZEN.json').exists():
        raise FileExistsError('freeze already exists')
    (U/'before').mkdir(parents=True, exist_ok=True)
    for name in ('WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md'):
        shutil.copy2(ROOT/name, U/'before'/name)
    shutil.copy2(LAB/'ops/STATE.json', U/'before/STATE.json')
    prep = read(LAB/'ops/word100_body_boundary_20261017/PREP_CONTEXT.json')
    # Public examples cover face vicinity, crossing and internal hand lowering.
    words = ('WORD0004','WORD0106','WORD0949','WORD1139','WORD1290','WORD2551')
    rows = []
    for code in words:
        for signer in (1,2,3):
            r = next(r for r in prep['ordered_rows'] if r['word_code']==code and r['signer_id']==signer)
            assert r['role']=='train_candidate'
            record_path = LAB/'ops/word100_body_boundary_20261017/data/records'/f"{r['clip_id']}.json"
            record = read(record_path)
            obs = record['views']['m0_native']['observation']
            rows.append(dict(clip_id=r['clip_id'], word=code, label=r['canonical_label'], signer=signer,
                role=r['role'], source=record['source'], retained_video=r.get('retained_video'),
                observation=obs, observation_sha256=sha256_file(obs),
                record=str(record_path), record_sha256=sha256_file(str(record_path)),
                roi=record['calibration']['box_source_pixels'], intervals=r['intervals']))
    protected = read(LAB/'ops/word100_dtw_alignment_evidence_20261027/FROZEN.json')['protected']
    for p, h in protected.items():
        assert sha256_file(p)==h, f'Protected changed before start: {p}'
    f=dict(at=datetime.now().astimezone().isoformat(), rows=rows, protected=protected,
           public_only=True, no_user_or_mentor_access=True, no_training=True,
           no_operating_web_changes=True, no_yolo_hamer_mano=True,
           face_model=read(ROOT/'Tools/models/face_landmarker.source.json'),
           region_schema='approximate surface vicinity; no contact or correctness claim',
           body_input='unchanged original MediaPipe observations; separate face sidecar',
           conditions={'A':'original reference / ROI-limited query, existing active_span',
                       'B':'same frozen full-frame ROI mask both, existing active_span',
                       'C':'B plus common wrist-elevation activity envelope, 3 consecutive frames, 3-frame pad'},
           activity_rule={'above_bottom_shoulder_ratio':.12,'min_consecutive':3,'pad_frames':3,
                          'union_hands':True,'interior_frames_never_removed':True,
                          'interpretation':'activity-envelope candidate, not semantic phase truth'},
           known_time_mapping='query 137% length, nonlinear source index round(N*t**1.35), spatial ROI masking only',
           dtw={'budget':96,'band':.2,'cost':'existing unmodified grouped feature cost'},
           cross_people='REAL01 reference vs REAL02 and REAL03; annotated coarse preparation/action/end audit only',
           acceptance=['source/model hashes unchanged','frame/PTS join <=1000us',
                       'mask preserves all points inside ROI and all frame IDs',
                       'reference coverage denominator unchanged after query deletion',
                       'normal crossing coordinates untouched',
                       'known-map timing error + coarse phase mismatch + retained annotation span reported'],
           limitations=['Face detection rate is not landmark accuracy',
                        'Face region boundaries need visual and later broader verification',
                        'Synthetic time warp is not a real speed study',
                        'Cross-person annotations identify sign interval, not exact internal phases',
                        'No direct inferred facial touch or sign correctness'])
    write(U/'FROZEN.json',f)
    write(U/'AUTHORIZATION.json',dict(user_request='Add MediaPipe facial regions; proceed public body/phase comparison',
        allowed=['official face model download','public full-video face extraction','separate diagnostic modules','public validation'],
        forbidden=['user learning/tuning/selection','TCN retraining','operating replacement','expression grading']))
    print('FROZEN',len(rows),flush=True)

def mask(obs, box):
    a=Observation(obs.xy_px.copy(),obs.timestamp_us.copy(),obs.frame_id.copy(),obs.observed.copy(),
                  obs.usable.copy(),obs.conf.copy(),obs.hand_score.copy(),dict(obs.sidecar))
    x0,y0,x1,y1=box
    keep=(a.xy_px[:,:,0]>=x0)&(a.xy_px[:,:,0]<x1)&(a.xy_px[:,:,1]>=y0)&(a.xy_px[:,:,1]<y1)
    a.observed &= keep; a.usable &= keep
    a.xy_px[~a.observed]=np.nan; a.conf[~a.observed]=np.nan
    for side,base in enumerate((33,54)):
        a.hand_score[~a.observed[:,base:base+21].any(axis=1),side]=np.nan
    a.validate(); return a

def slice_obs(obs, ids, new_times=None):
    a=Observation(obs.xy_px[ids].copy(),obs.timestamp_us[ids].copy(),obs.frame_id[ids].copy(),
                  obs.observed[ids].copy(),obs.usable[ids].copy(),obs.conf[ids].copy(),obs.hand_score[ids].copy(),dict(obs.sidecar))
    if new_times is not None:
        a.timestamp_us=np.asarray(new_times,dtype=np.int64); a.frame_id=np.arange(len(ids),dtype=np.int64)
    a.validate();return a

def envelope(obs, box):
    _, width, _=body_frame(obs)
    active=np.zeros(obs.n_frames,bool)
    for base in (33,54):
        active |= obs.usable[:,base] & (obs.xy_px[:,base,1] < box[3]-.12*width)
    # Only boundary support needs persistence; never delete an interior gap/crossing.
    stable=np.convolve(active.astype(int),np.ones(3,dtype=int),mode='same')==3
    ids=np.flatnonzero(stable)
    if not ids.size: return (0,obs.n_frames-1,'NO_RELIABLE_BOUNDARY_KEEP_ALL')
    return (max(0,int(ids[0])-3),min(obs.n_frames-1,int(ids[-1])+3),'WRIST_ELEVATION_ENVELOPE')

def seq(obs,box,condition):
    work=obs
    if condition=='C':
        lo,hi,_=envelope(obs,box); work=slice_obs(obs,np.arange(lo,hi+1))
        s=sequence_features(work,trim=False)
    else: s=sequence_features(work,trim=True)
    return uniform_time_sample(s,96)

def prepare():
    f=read(U/'FROZEN.json'); (U/'data').mkdir(exist_ok=True)
    for i,r in enumerate(f['rows']):
        cid=r['clip_id']; out=U/'data'/cid;out.mkdir(exist_ok=True)
        marker=out/'DONE.json'
        if marker.exists():
            done=read(marker)
            for p,h in done['outputs'].items(): assert sha256_file(p)==h
            continue
        video=Path(r['retained_video']) if r.get('retained_video') and Path(r['retained_video']).exists() else out/'source.mp4'
        if not video.exists():
            source=r['source']
            with zipfile.ZipFile(source['archive']) as z:
                inf=z.getinfo(source['member']);assert inf.CRC==source['crc32']
                video.write_bytes(z.read(inf))
        assert sha256_file(str(video))==r['source']['sha256']
        assert sha256_file(r['observation'])==r['observation_sha256']
        obs=Observation.load(r['observation']); face_path=out/'face.npz'
        if not face_path.exists():
            extract_face_video(video,ROOT/'Tools/models/face_landmarker.task',face_path)
        face=join_face_observation(face_path,obs)
        limited=mask(obs,r['roi']); rel=relations(limited,face)
        np.savez_compressed(out/'relations.npz',**rel)
        limited.save(str(out/'masked.npz'))
        with np.load(face_path) as z:
            found=int(z['found'].sum()); elapsed=z['inference_seconds']
        intervals=r['intervals']; lo,hi,reason=envelope(limited,r['roi'])
        annotated=np.zeros(obs.n_frames,bool)
        for a,b in intervals:annotated |= (obs.timestamp_us>=a*1e6)&(obs.timestamp_us<=b*1e6)
        retained=annotated.copy();retained[:lo]=False;retained[hi+1:]=False
        d=dict(clip_id=cid,video=str(video),frames=obs.n_frames,face_found_frames=found,
          face_join_max_us=face['max_pts_delta_us'],face_ms_median=float(np.median(elapsed)*1000),
          face_ms_p95=float(np.percentile(elapsed,95)*1000),
          valid_region_frames=rel['region_valid'].sum(axis=0).tolist(),
          projected_hand_overlap_frames=rel['projected_hand_overlap'].sum(axis=0).tolist(),
          envelope=[lo,hi],envelope_reason=reason,annotated_frames=int(annotated.sum()),
          envelope_retained_annotated_frames=int(retained.sum()),
          outputs={str(p):sha256_file(str(p)) for p in (face_path,out/'relations.npz',out/'masked.npz')})
        write(marker,d)
        print(f'{i+1}/{len(f["rows"])} {cid}: face {found}/{obs.n_frames}',flush=True)
    print('PREPARED',flush=True)

def align(a,b,boxa,boxb,condition):
    sa,sb=seq(a,boxa,condition),seq(b,boxb,condition)
    cost,evidence=frame_cost_matrix(sa,sb)
    path=np.asarray(_optimal_path(cost,.2),dtype=int)
    ai,bi=sa['frame_id'][path[:,0]],sb['frame_id'][path[:,1]]
    return ai,bi,float(np.mean(cost[path[:,0],path[:,1]])),float(np.mean(evidence[path[:,0],path[:,1]]))

def phase(times,intervals):
    if not intervals:return np.full(len(times),-1)
    a=min(i[0] for i in intervals)*1e6;b=max(i[1] for i in intervals)*1e6
    return np.where(times<a,0,np.where(times>b,2,1))

def evaluate():
    if (U/'RESULT.json').exists(): raise FileExistsError('Completed result is frozen')
    f=read(U/'FROZEN.json'); result=dict(clips=[],known_mapping=[],cross_people=[],regions=list(REGIONS))
    cache={}
    for r in f['rows']:
        cid=r['clip_id'];out=U/'data'/cid;d=read(out/'DONE.json');result['clips'].append(d)
        original=Observation.load(r['observation']);limited=Observation.load(str(out/'masked.npz'))
        relation={k:v for k,v in np.load(out/'relations_roi.npz').items()}
        cache[cid]=(original,limited,relation)
        n=original.n_frames; m=round(n*1.37)
        source=np.round(np.linspace(0,1,m)**1.35*(n-1)).astype(int)
        query=slice_obs(limited,source,np.arange(m,dtype=np.int64)*33333)
        qrel={k:(v[source] if len(v)==n else v) for k,v in relation.items()}
        paths={}
        for cond in 'ABC':
            ref=original if cond=='A' else limited
            ai,bi,cost,ev=align(ref,query,r['roi'],r['roi'],cond)
            errors=np.abs(original.timestamp_us[ai]-original.timestamp_us[source[bi]])/1000
            same=phase(original.timestamp_us[ai],r['intervals'])==phase(original.timestamp_us[source[bi]],r['intervals'])
            result['known_mapping'].append(dict(clip_id=cid,condition=cond,pairs=len(ai),
                mean_mapping_error_ms=float(errors.mean()),p95_mapping_error_ms=float(np.percentile(errors,95)),
                coarse_phase_mismatch=int((~same).sum()),coarse_phase_pairs=len(same),cost=cost,evidence=ev,
                relations=compare_path(relation,qrel,ai,bi)))
            paths[cond+'_reference']=ai;paths[cond+'_query']=bi
        np.savez_compressed(out/'known_paths.npz',source_mapping=source,**paths)
    for r in f['rows']:
        if r['signer']!=1:continue
        for q in (x for x in f['rows'] if x['word']==r['word'] and x['signer']!=1):
            ro,rm,rr=cache[r['clip_id']];qo,qm,qr=cache[q['clip_id']]
            paths={}
            for cond in 'ABC':
                ai,bi,cost,ev=align(ro if cond=='A' else rm,qm,r['roi'],q['roi'],cond)
                same=phase(ro.timestamp_us[ai],r['intervals'])==phase(qo.timestamp_us[bi],q['intervals'])
                result['cross_people'].append(dict(reference=r['clip_id'],query=q['clip_id'],condition=cond,
                    pairs=len(ai),coarse_phase_mismatch=int((~same).sum()),cost=cost,evidence=ev,
                    relations=compare_path(rr,qr,ai,bi)))
                paths[cond+'_reference']=ai;paths[cond+'_query']=bi
            np.savez_compressed(U/'data'/q['clip_id']/'cross_paths.npz',**paths)
    for section in ('known_mapping','cross_people'):
        result[section+'_summary']={}
        for cond in 'ABC':
            rows=[r for r in result[section] if r['condition']==cond]
            s=dict(comparisons=len(rows),coarse_phase_mismatch=sum(r['coarse_phase_mismatch'] for r in rows),
                   pairs=sum(r['pairs'] for r in rows))
            if section=='known_mapping':s['mean_clip_mapping_error_ms']=float(np.mean([r['mean_mapping_error_ms'] for r in rows]))
            result[section+'_summary'][cond]=s
    write(U/'RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k.endswith('_summary')},ensure_ascii=False),flush=True)

def recover():
    f=read(U/'FROZEN.json')
    fix=U/'RECOVERY_PROTOCOL.json'
    if not fix.exists():
        write(fix,dict(reason='18/18 full-frame face detector produced zero results; short-range face scale is suspected',
            rule='Missing full-frame face only; current full-frame Pose33 face center and shoulder scale define a head RGB ROI',
            half_width_shoulder_ratio=.65,half_height_shoulder_ratio=.70,
            original_results='data/*/face.npz and DONE.json retained',
            output='face_roi.npz; original image coordinate system; no hand/body/TCN input change',
            parameter_selection='geometry before recovered outputs; no user data, labels or result tuning'))
    for r in f['rows']:
        out=U/'data'/r['clip_id'];d=read(out/'DONE.json')
        done=out/'RECOVERED.json'
        if done.exists():
            for p,h in read(done)['outputs'].items():assert sha256_file(p)==h
            continue
        obs=Observation.load(r['observation']);p=out/'face_roi.npz'
        if not p.exists():recover_face_video(d['video'],ROOT/'Tools/models/face_landmarker.task',out/'face.npz',obs,p)
        face=join_face_observation(p,obs);limited=Observation.load(str(out/'masked.npz'))
        rel=relations(limited,face);np.savez_compressed(out/'relations_roi.npz',**rel)
        with np.load(p) as z:
            v=dict(clip_id=r['clip_id'],frames=obs.n_frames,found_frames=int(z['found'].sum()),
                source_counts={str(k):int((z['source_kind']==k).sum()) for k in (0,1,2)},
                inference_ms_median=float(np.median(z['inference_seconds'])*1000),
                inference_ms_p95=float(np.percentile(z['inference_seconds'],95)*1000),
                max_pts_delta_us=face['max_pts_delta_us'],
                region_valid_frames=rel['region_valid'].sum(axis=0).tolist(),
                projected_hand_overlap_frames=rel['projected_hand_overlap'].sum(axis=0).tolist(),
                outputs={str(a):sha256_file(str(a)) for a in (p,out/'relations_roi.npz')})
        write(done,v);print(r['clip_id'],v['found_frames'],'/',obs.n_frames,flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    {'freeze':freeze,'prepare':prepare,'recover':recover,'evaluate':evaluate}[sys.argv[1]]()
