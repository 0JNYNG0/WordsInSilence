from pathlib import Path
import json,hashlib,collections
import numpy as np
from state_model import features,CLASSES
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data):
    with (U/name).open('x',encoding='utf-8')as f:json.dump(data,f,ensure_ascii=False,indent=2)
labels=read(U/'STATE_LABELS.json');manifest={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']}
arrays={};rows=[];pins={};loaded={};checks=0
for r in labels['rows']:
    cid=r['clip_id'];m=manifest[cid]
    assert r['person'] in range(2,10) and m['person']==r['person']
    assert m['source_sha256']==r['source_sha256'];checks+=2
    if cid not in loaded:
        assert sha(Path(m['path']))==m['sha256'] and sha(Path(m['channel']))==m['channel_sha256']
        assert sha(Path(r['source']))==r['source_sha256'];checks+=3
        z=np.load(m['channel']);raw=np.load(m['path'])
        assert np.array_equal(z['frame_id'],raw['frame_id'])
        loaded[cid]=(z,raw);pins.update({m['path']:m['sha256'],m['channel']:m['channel_sha256'],r['source']:r['source_sha256']})
        arrays[cid+'_x']=z['feedback'];arrays[cid+'_frame_id']=z['frame_id'];arrays[cid+'_timestamp_us']=z['timestamp_us']
    z,raw=loaded[cid];matches=np.flatnonzero(z['frame_id']==r['frame']);assert len(matches)==1;i=int(matches[0])
    f,ok=features(z['feedback'][i:i+1],r['domain'],r['hand'])
    out=dict(r,index=len(rows),array_index=i,available=bool(ok[0]),feature=f[0].tolist(),target=-1 if r['label']is None else CLASSES[r['domain']].index(r['label']))
    rows.append(out);checks+=1
assert len(loaded)==32
for domain in CLASSES:
    fit=[r for r in rows if r['domain']==domain and r['split']=='fit'and r['target']>=0 and r['available']]
    assert {r['target']for r in fit}==set(range(len(CLASSES[domain])))
    assert {r['person']for r in fit}==set(range(2,8))
np.savez_compressed(U/'source_arrays.npz',**arrays)
write('LEARNING_INPUTS.json',dict(rows=rows,pins=pins,arrays=str(U/'source_arrays.npz'),arrays_sha256=sha(U/'source_arrays.npz'),checks=checks,
 summary={s:{d:dict(reviewed=sum(r['split']==s and r['domain']==d for r in rows),known=sum(r['split']==s and r['domain']==d and r['target']>=0 for r in rows),available_known=sum(r['split']==s and r['domain']==d and r['target']>=0 and r['available']for r in rows))for d in CLASSES}for s in ['fit','exposed_probe']}))
prev=read(O/'word100_action_completion_binding_20261114/TRAINING_FROZEN.json')
protected=prev['protected']
for p,h in protected.items():assert sha(Path(p))==h
for name in ['PLAN.json','STATE_LABELS.json','LEARNING_INPUTS.json','source_arrays.npz','state_model.py','train_states.py','evaluate_states.py']:
    pins[str(U/name)]=sha(U/name)
write('TRAINING_FROZEN.json',dict(fit_record_ids=list(range(2,8)),probe_record_ids=[8,9],
 purpose='Observable goal-state components separated from immutable action localization; not whole-sign correctness.',
 models=['linear','mlp'],seeds=[42,43,44],primary='mlp three-seed consensus; linear is a matched supervised control, not old web model',
 steps=800,lr=.01,weight_decay=.001,batch='full per-domain class/clip-balanced loss',hidden=16,
 confidence=.8,consensus='all three seeds agree, all >=0.8; no tuning on exposed probe',
 features='body-relative frame geometry only; no time, source index, word or future context',
 invariance='translation, uniform camera scale, small in-plane rotation, mirrored hand role swap; no invented 3D',
 unknown_labels='excluded from class loss; retained for ambiguity audit, not assumed errors',
 temporal_tests='reviewed source states only; normal/shortened timestamps/early-stop/goal-only/reversed/missing; synthetic controls not human mistakes',
 admission=dict(probe_component='all known visible states correct for primary consensus with no abstention; ambiguous-state outputs audited, not accuracy-scored',
  temporal='no false goal on source-start-only, no complete action from goal-only, no unfinished claim from missing tail; preserve short ordered state change',
  integration='all 246 saved existing comparisons preserved byte-equivalent after removing new diagnostic fields',
  scope='A passed component is research-observation validated only. Educational completeness, automatic boundary localization and unseen-user safety remain unvalidated; no web promotion this unit.'),
 protected=protected,pins=pins,private_used=False,final8_opened=False,web_changed=False))
write('CURRENT_EXECUTION.json',dict(status='INPUTS_VERIFIED_TRAINING_FROZEN',training_running=False,source_videos=32))
print(json.dumps(read(U/'LEARNING_INPUTS.json')['summary'],ensure_ascii=False))
