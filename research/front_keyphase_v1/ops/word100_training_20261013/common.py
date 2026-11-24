"""Local, frozen100-word experiment utilities. No camera/network/production writes."""
import hashlib
import json
from pathlib import Path
import sys

U=Path(__file__).resolve().parent
BASE=U.parents[1]
ROOT=BASE.parents[1]
OUT=Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013')
RUN=BASE/'runs/word100_tcn_v1'
sys.path[:0]=[str(BASE/'src'),str(ROOT/'research/sign_model_pilot_v1/src')]

def read(path): return json.loads(Path(path).read_text('utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def write(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    tmp.replace(path)
def pin_check():
    f=read(U/'FROZEN_EXPERIMENT.json')
    assert all(sha(p)==h for p,h in f['source_sha256'].items()), 'Frozen source modified'
    for p,s in f['archive_stat'].items():
        st=Path(p).stat()
        assert [st.st_size,st.st_mtime_ns]==s, 'Archive stat changed'
    return f

def freeze():
    assert not (U/'FROZEN_EXPERIMENT.json').exists(), 'Reuse frozen experiment'
    planpath=U.parent/'word100_dayplan_20261013/PLAN.json'
    jp=U.parent/'word_dataset_plan_20261013/FRONT_VIDEO_JOIN.jsonl'
    vp=U.parent/'word_label_time_contract_20261013/VOCABULARY_V2.json'
    plan=read(planpath); codes=sorted(plan['planning_sample100_codes'])
    vd={x['code']:x for x in read(vp)}
    labels=[next(iter(vd[c]['labels'])) for c in codes]
    assert len(set(labels))==len(codes)==100 and all(vd[c]['candidate'] for c in codes)
    rows=[json.loads(s) for s in jp.read_text('utf-8').splitlines()]
    selected=[]; excluded=[]
    for r in rows:
        if r['word_code'] not in codes: continue
        if r['role'] not in ['train_candidate','development_only']: continue
        if not r['metadata_video_join_eligible']:
            excluded.append(r);continue
        y=codes.index(r['word_code'])
        assert r['labels'][0].strip()==labels[y], 'Never correct evaluation label by training majority'
        assert len(r['intervals'])==1 and len(r['video_entries'])==1
        e=r['video_entries'][0]; sid=e['filename_signer_id']
        assert sid in (range(1,10) if r['role']=='train_candidate' else range(11,14))
        selected.append(dict(r,label_index=y,canonical_label=labels[y],signer_id=sid))
    selected.sort(key=lambda r:(r['role']!='train_candidate',r['video_entries'][0]['archive'],r['clip_id']))
    assert sum(x['role']=='train_candidate' for x in selected)==899
    assert sum(x['role']=='development_only' for x in selected)==300
    write(U/'MANIFEST.json',selected)
    write(U/'EXCLUDED_METADATA.json',excluded)
    pins=[planpath,jp,vp,U/'MANIFEST.json',BASE/'config/contract.json',
        BASE/'src/wis_lab/capture/landmarker.py',BASE/'src/wis_lab/capture/video_source.py',
        BASE/'src/wis_lab/core/observation.py',BASE/'src/wis_lab/service/neural.py',
        ROOT/'research/sign_model_pilot_v1/src/pilot/data.py',ROOT/'research/sign_model_pilot_v1/src/pilot/web_input.py',
        ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py',
        ROOT/'Tools/models/hand_landmarker.task',ROOT/'Tools/models/pose_landmarker_lite.task']
    # Training code pinned when it is ready, before it can see development scores.
    archives={r['video_entries'][0]['archive'] for r in selected}
    f=dict(status='FROZEN_BEFORE_EXTRACTION_AND_TRAINING',authorization='2026-10-13 user: 오케이 이해했어. 그럼100개 단어에 대해서 진행하면 될 것 같아.',
        author='Codex local implementation and separate verification',classes=codes,labels=labels,
        output=str(OUT),run=str(RUN),train_clips=899,dev_clips=300,train_signers=list(range(1,10)),dev_signers=[11,12,13],
        test_signers_read=[],input='Same production MediaPipe full native PTS ->75point observation ->280channels; no hand filling, no sparse resampling; original missing masks and elapsed time.',
        source_sha256={str(p):sha(p) for p in pins},archive_stat={p:[Path(p).stat().st_size,Path(p).stat().st_mtime_ns] for p in archives},
        model=dict(type='MaskedSignTCN',in_channels=280,width=32,dilations=[1,2,4,8],kernel_size=3,dropout=.1,n_classes=100),
        train=dict(seed=20261013,batch_size=32,epochs_max=80,early_stopping_patience=12,
                   optimizer='AdamW',lr=.001,weight_decay=.001,loss='cross_entropy',gradient_clip_norm=1.,
                   initialization='fresh random, identical state in both arms; no NIKL/KSL/old checkpoint weights',
                   order='seed+epoch permutation of all accepted training clips, once each per epoch',
                   checkpoint_selection='minimum mean development CE; strict improvement1e-6; tie earliest epoch',
                   cpu_threads=4,arms=['whole','whole_context'],
                   whole='existing observed-hand-span input; no annotation trimming',
                   whole_context='same input on alternating epochs; otherwise core +/-0.30s (stable SHA256 clip parity). Core times used for training clips only.',
                   wall_safety_seconds_per_arm=3600,if_time_limit='save state and mark incomplete; never label a prematurely interrupted run fully trained'),
        success='Complete auditable100-class baseline development results. Candidate method effect only if primary300-row top1 increases; report macro and all3 signers/regressions. No claim about final blind test/generalization or superiority over old93-class model.',
        evaluation=dict(main='full observed-hand span only; no requested label/core passed to forward',denominator=300,
                        invalid_input='retained failure in300-row primary denominator; valid-only score additional',
                        phase='development/model selection, not untouched final test',paired='compare saved same300 rows, not different vocabularies'),
        chunks=dict(clips=300,wall_seconds=7200,workers=1,total_preparation_seconds=14400),
        quarantine='CRC/decode/hash/timestamp failures stop. Input-gate failures recorded, not synthesized. Training stop if any class has zero usable clips or >5percent training failures.',
        constraints=['no heldout14-18 or signer10 arrays','no user/mentor/NIKL/KSL train or tune','no RGB bulk','no camera/download/external upload','no production model replacement','no Git changes'])
    write(U/'FROZEN_EXPERIMENT.json',f)
    print(json.dumps(dict(classes=100,train=899,dev=300,excluded=len(excluded)),ensure_ascii=False))

if __name__=='__main__': freeze()
