from prepare_review import *
from collections import Counter,defaultdict
FINGERS=['thumb','index','middle','ring','little'];CODES={'E':0,'B':1,'F':2,'U':-1}
# Individual native-frame judgements transcribed after RGB inspection; not generated from class names.
M={
'WORD2523_REAL01':{31:('EEUUU','EEUUU'),49:('UBUUU','UBUUU'),55:('UBUUU','UBUUU')},
'WORD2523_REAL04':{73:('EEUUU','EEUUU'),93:('UBUUU','UBUUU'),102:('UBUUU','UBUUU')},
'WORD2523_REAL07':{39:('EEUUU','EEUUU'),51:('UBUUU','UBUUU'),57:('UBUUU','UBUUU')},
'WORD1140_REAL01':{50:('EBBBU','EBBBB'),69:('EBBBB','EUBBB'),78:('EBBBB','EUBBB')},
'WORD1140_REAL02':{45:('EBUBB','EUBBB'),59:('EBBBU','EUBBB'),73:('EBBBB','EUBBB')},
'WORD1140_REAL03':{55:('EBUBU','EUBBB'),64:('EBBBB','EBBBB'),81:('EBBBB','EUBBB')},
'WORD0196_REAL01':{50:('UFFFF','UFFFF'),59:('UFFFF','EFFFF'),77:('EUUUU','EFFFF')},
'WORD0196_REAL02':{44:('UFFFF','UFFFF'),54:('UFFFF','UFFFF'),84:('EUUUU','EFFFF')},
'WORD0196_REAL03':{75:('UFFFF','UFFFF'),87:('UFFFF','UFFFF'),111:('EUUUU','EFFFF')},
'WORD2551_REAL01':{37:('UFFFF',None),44:('UBBBB',None),52:('UEEEE',None)},
'WORD2551_REAL02':{54:('UFFFF',None),62:('UBBBB',None),79:('UUEEE',None)},
'WORD2551_REAL03':{61:('UBBBB',None),69:('UBBBB',None),77:('UEEEE',None)}
}
basis={
'WORD2523':dict(known='삼각형처럼 손을 세운 시점은 엄지·검지의 편 윤곽을 확인했다. 옆으로 모은 시점은 검지가 손바닥 축에서 구부러진 윤곽을 확인했다. 다른 손가락의 가려진 마디와 엄지 모음은 개별 굽힘으로 단정하지 않았다.',unknown='손날 방향에서 다른 긴 손가락이 포개지고 엄지 모음/굽힘이 구분되지 않음.',state_role='held',motion_role='pose_with_hand_relocation'),
'WORD1140':dict(known='엄지는 손바닥에서 길게 편 윤곽이며 긴 손가락은 손가락 마디와 끝이 안으로 돌아가는 둥근 윤곽이다. 주먹처럼 손바닥까지 완전히 말린 모습과 구분했다. 각 시점에서 구부림이 불분명한 검지/중지/소지는 U로 남겼다.',unknown='정면 단축·끝마디 겹침 또는 움직임 흐림 때문에 약한 굽힘과 펴짐을 확정하지 못함.',state_role='held',motion_role='moving_bilateral_hands'),
'WORD0196':dict(known='주먹 시점은 긴 손가락 마디가 손바닥으로 말려 들어간 접힌 윤곽을 각각 확인했다. 마지막 선택 시점은 왼 엄지가 펴지고 나머지 네 손가락은 접힌 윤곽이다. 오른손은 손바닥이 카메라를 향해 짧게 투영되어 엄지만 확정했다.',unknown='주먹을 덮는 엄지의 굽힘·모음 구별 불가 또는 손바닥 쪽으로 짧게 투영된 긴 손가락의 중간마디가 불분명함.',state_role='held',motion_role='moving_or_rotating_fist_then_other_hand_relation'),
'WORD2551':dict(known='손등에서 보이는 주먹의 접힌 윤곽, 손가락 끝이 안으로 구부러진 펼침 중간 상태, 길게 펴진 네 손가락을 원본 각 시점에서 구분했다. 엄지는 가려져 제외했다.',unknown='엄지 가림 또는 검지의 약한 굽힘과 정면 단축을 구별하기 어려움.',state_role='transition_or_held',motion_role='right_hand_opening')
}
inputs=rd(U/'ANCHOR_INPUTS.json')['rows'];rows=[];duplicates=[]
old=[]
for name in ['ORDINARY_LABELS.json','EXTERNAL_LABELS.json']:
 old+=rd(O/'word100_feature_expansion_20261113/annotation_review'/name)['rows']
oldkeys={(r['clip_id'],r['frame'],r['side']) for r in old}
for r in inputs:
 key=r['clip_id'].removeprefix('NIA_SL_').removesuffix('_F');ss=M[key][r['frame']];word=r['word'];b=basis[word]
 for side,text in zip(['R','L'],ss):
  if text is None:continue
  if (r['clip_id'],r['frame'],side) in oldkeys:
   prior=next(q for q in old if (q['clip_id'],q['frame'],q['side'])==(r['clip_id'],r['frame'],side))
   assert prior['states']==list(text),'conflicting prior anchor'
   duplicates.append(dict(clip_id=r['clip_id'],frame=r['frame'],side=side,states=list(text),disposition='identical old anchor rechecked; excluded new training cells'))
   continue
  phase='main_candidate_rgb_review_only'
  state_role=b['state_role']
  if word=='WORD2551':
   state_role='shape_transition' if 'B' in text else 'held_at_native_anchor'
  rec=dict(r,side=side,y=[CODES[c] for c in text],states=list(text),known=sum(c!='U' for c in text),annotation_basis=b['known'],unknown_reason=b['unknown'],reviewer='Codex expansion_labels_review_1113 actual RGB review',review_status='agent_rgb_review_pending_root',derived_from_word_identity=False,derived_from_model=False,phase_role=phase,state_role=state_role,motion_role=b['motion_role'],source_clause_binding_status='not_attached_to_educational_requirement',training_allowed_pending_root=True)
  rec['finger_evidence']=[dict(finger=f,state=c,visibility='clear_state_outline' if c!='U' else 'not_resolved',state_basis=b['known'] if c!='U' else b['unknown'],geometry_status='state_cues_present_but_no_exact_3d_claim' if c!='U' else 'not_assessable') for f,c in zip(FINGERS,text)]
  # Specific observed projection ambiguity; not silently relabelled from keypoint geometry.
  if word=='WORD1140' and r['person'] in [2,3]:
   for e in rec['finger_evidence']:
    if e['state']=='B':e['geometry_status']='projected_bend_shallow_or_shortened_some_joints'
  rec['geometry_review_note']='RGB/RTMW overlay reviewed: no obvious whole-hand swap at this selected anchor; 2D points are not exact3D joint truth.'
  rows.append(rec)
wr(U/'EXTRA_LABELS.json',dict(schema=dict(fingers=FINGERS,codes=CODES,U='unresolved: ignore supervised state loss; retain coverage denominator'),rows=rows,complete=True,expert_gold=False,selection='FIT-only RGB, fixed clips and native anchor timestamps; never inferred from digit/word identity',excluded_active_scope='Star inactive left hand not supervised; no automatic nearby-frame propagation',source_review_images=135,geometry_review_frames=36))
wr(U/'DUPLICATE_RECHECKS.json',dict(rows=duplicates,first_failure_preserved='FIRST_FAILURE.json'))
counts=Counter(c for r in rows for c in r['states']);byhand={s:dict(Counter(c for r in rows if r['side']==s for c in r['states'])) for s in ['R','L']}
matrix=[]
for side in ['R','L']:
 for finger,idx in zip(FINGERS,range(5)):
  for state in ['E','B','F']:
   rr=[r for r in rows if r['side']==side and r['states'][idx]==state]
   matrix.append(dict(side=side,finger=finger,state=state,cells=len(rr),clips=len({r['clip_id'] for r in rr}),record_ids=sorted({r['person'] for r in rr}),expressions=sorted({r['word'] for r in rr}),minimum_coverage_met=len({r['person'] for r in rr})>=3 and len({r['word'] for r in rr})>=2))
wr(U/'COVERAGE.json',dict(clips=len({r['clip_id'] for r in rows}),frames=len(inputs),hand_rows=len(rows),known_cells=sum(c!='U' for r in rows for c in r['states']),counts=dict(counts),by_hand=byhand,matrix=matrix,coverage_is_not_accuracy=True,independent_samples='12 clip-level sources; sparse frames from same clip not independent people',unsupported=['thumb B/F','spread','thumb adduction','exact contact','educational acceptable boundary'],historical_base_fit_cells=861))
print('EXTRA',len(rows),dict(counts),byhand)

