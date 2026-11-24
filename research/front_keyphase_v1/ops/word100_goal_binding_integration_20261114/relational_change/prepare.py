from pathlib import Path
import json,hashlib,collections
import numpy as np
from core import measurement
U=Path(__file__).resolve().parent;O=U.parent.parent;G=O/'word100_web_completion_graph_20261111';S=O/'word100_goal_state_learning_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
source=G/'typed_motion_recovery_20261113/body_action_binding_review/BODY_POINT_CONTRACT.json';audit=read(source)
arr=np.load(S/'source_arrays.npz');print('array sample',arr.files[:3])
episodes=[r for p in [G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json']for r in read(p)['rows']]
rows=[];excluded=[]
def add(cid,person,frame_a,frame_b,axis,target,origin,reverse=False,freeze=False):
    x=arr[cid+'_x'];fid=arr[cid+'_frame_id'];ts=arr[cid+'_timestamp_us'];aa=np.flatnonzero(fid==frame_a);bb=np.flatnonzero(fid==frame_b)
    if len(aa)!=1 or len(bb)!=1:
        excluded.append(dict(clip_id=cid,frames=[frame_a,frame_b],reason='exact_source_endpoint_unavailable'));return
    a,b=int(aa[0]),int(bb[0]);assert a<b
    xx=x[a:b+1].copy();tt=ts[a:b+1].copy()
    if reverse:xx=xx[::-1].copy();tt=tt[0]+tt[-1]-tt[::-1]
    if freeze:xx[:]=xx[0]
    measurements={}
    for mode in ['body_wrist','stable_palm']:
        val,ok,meta=measurement(xx,tt,[0,len(xx)-1],axis,palm=mode=='stable_palm')
        measurements[mode]=dict(delta=val,available=ok,details=meta)
    rows.append(dict(index=len(rows),clip_id=cid,person=person,split='fit'if person<=7 else'exposed_probe',
       source_frames=[frame_a,frame_b],axis=axis,target=target,origin=origin,reverse=reverse,freeze=freeze,
       measurements=measurements,source_semantic_completion=None))
for r in audit['segment_rows']:
    label=r['existing_body_label'];person=r['person'];assert 2<=person<=7
    assert label in ['hold','wrist_lowering','forearm_raising']
    axes=[0,1]if label=='hold'else[0]if label=='wrist_lowering'else[1]
    # Preserve unknown and simultaneous unlabelled dimensions; they are NOT
    # turned into negatives for the other observed source action.
    for axis in axes:
        a,b=r['source_frames']
        if a==b:continue
        target=1 if label=='hold'else 2
        add(r['clip_id'],person,a,b,axis,target,'direct_RGB_body_role')
        if target==2:add(r['clip_id'],person,a,b,axis,0,'reversed_direct_RGB_change',reverse=True)
for r in episodes:
    if r['person']not in [8,9] or r['word_id']!='WORD0739':continue
    for g in r['groups']:
        if g['family']not in [2,3]:continue
        axis=g['family']-2;a=g['onset'][0];b=g['offset'][1]
        add(r['clip_id'],r['person'],a,b,axis,2,'reviewed_source_action_endpoints')
        add(r['clip_id'],r['person'],a,b,axis,0,'reversed_reviewed_source',reverse=True)
        add(r['clip_id'],r['person'],a,b,axis,1,'fixed_start_posture_control',freeze=True)
assert {r['axis']for r in rows if r['target']==2 and r['split']=='fit'}=={0,1},collections.Counter((r['axis'],r['target'])for r in rows)
write('INPUTS.json',dict(rows=rows,excluded=excluded,source_pins={str(source):sha(source),str(S/'source_arrays.npz'):sha(S/'source_arrays.npz')},
 target_origin='Existing direct RGB role labels and reviewed episode endpoints; reverse/freeze targets are physical change controls, not wrong-sign lessons',
 source_role_not_exclusive_physics=True,no_probe_relabel=True))
write('FROZEN.json',dict(seeds=[42,43,44],steps=800,lr=.05,weight_decay=.001,models=['body_wrist','stable_palm'],
 architecture='Two independent monotone three-way change heads, 4 learned calibration parameters. Not a replacement for common100 Conv+GRU.',
 confidence=.8,all_seeds_agree=True,comparison='Same labelled source intervals: old body wrist displacement vs stable palm displacement; independent forearm upright change',
 core_sha256=sha(U/'core.py'),inputs_sha256=sha(U/'INPUTS.json'),
 fit_record_ids=[2,3,4,5,6,7],exposed_probe_record_ids=[8,9],
 no_goal_angle_threshold=True,no_training_full_completion=True,no_final8=True,
 limitations='Change direction and posture hold only; cannot learn chest region/contact/fullcompletion from these labels. Does not repair all G2 assignments.',
 gates=['All seeds keep original favourable normal changes','Opposite changes explicit; hold not opposite','Unavailable endpoints unknown','No pose-angle class used as completion']))
print(collections.Counter((r['split'],r['axis'],r['target'])for r in rows))
