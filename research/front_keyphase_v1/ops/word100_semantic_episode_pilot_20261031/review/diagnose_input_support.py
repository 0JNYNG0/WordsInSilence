"""Read-only diagnosis of frozen observation and ROI support; no model imports."""
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

U=Path(__file__).resolve().parents[1]
OUT=U/'review'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']}
prepared=read(U/'PREPARED.json')
events=prepared['events']
cache={}
for alias in sorted({e['alias'] for e in events}):
 r=rows[alias]
 with np.load(r['input'],allow_pickle=False) as z:
  a={k:np.array(z[k]) for k in ('x','mask','frame_id','timestamp_us')}
 with np.load(r['observation'],allow_pickle=False) as z:
  raw={k:np.array(z[k]) for k in ('xy_px','observed','usable','frame_id','timestamp_us')}
 assert np.array_equal(a['frame_id'],raw['frame_id'])
 assert np.array_equal(a['timestamp_us'],raw['timestamp_us'])
 xy=raw['xy_px'].astype(np.float64)
 finite=np.isfinite(xy).all(2)
 observed=raw['observed'].astype(bool)
 usable=raw['usable'].astype(bool)
 valid=observed&usable&finite
 box=np.asarray(r['box'])
 inside=(xy[:,:,0]>=box[0])&(xy[:,:,0]<box[2])&(xy[:,:,1]>=box[1])&(xy[:,:,1]<box[3])
 roi_valid=valid&inside
 scale=np.linalg.norm(xy[:,11]-xy[:,12],axis=1)
 scale_ok=np.isfinite(scale)&(scale>1e-8)
 body_raw=valid[:,11]&valid[:,12]&scale_ok
 body_roi=roi_valid[:,11]&roi_valid[:,12]&scale_ok
 direct=roi_valid[:,54]&body_roi
 saved=a['mask'][:,[28,29]].all(1)&np.isfinite(a['x'][:,[28,29]]).all(1)
 assert np.array_equal(direct,saved),alias
 cache[alias]=dict(r=r,a=a,raw=raw,xy=xy,finite=finite,observed=observed,
  usable=usable,valid=valid,inside=inside,roi_valid=roi_valid,
  body_raw=body_raw,body_roi=body_roi,scale_ok=scale_ok,saved=saved,
  pins={'input_sha256':sha(r['input']),'observation_sha256':sha(r['observation'])})

records=[]
all_missing=Counter()
by_role=defaultdict(Counter)
for e in events:
 c=cache[e['alias']]
 a=c['a'];f=a['frame_id']; ix=np.flatnonzero((f>=e['core_start_frame'])&(f<=e['core_end_frame']))
 assert len(ix)==e['core_end_frame']-e['core_start_frame']+1
 assert len(ix)==e['frames']
 s=c['saved'][ix]
 n=len(ix);cov=float(s.mean())
 eligible=n>=3 and cov>=.75 and s[:min(3,n)].any() and s[-min(3,n):].any()
 assert bool(eligible)==e['eligible'],e['event_id']
 assert abs(cov-e['input_coverage'])<1e-12,e['event_id']
 causes=Counter();missing=[]
 for j in ix[~s]:
  observed=c['observed'][j]; usable=c['usable'][j]; finite=c['finite'][j]
  valid=c['valid'][j]; inside=c['inside'][j]
  all_causes=[]
  if not observed[54]: all_causes.append('raw_right_hand_wrist_unobserved')
  elif not usable[54]: all_causes.append('raw_right_hand_wrist_unusable')
  elif not finite[54]: all_causes.append('raw_right_hand_wrist_nonfinite')
  elif not inside[54]: all_causes.append('valid_right_hand_wrist_outside_roi')
  for k,name in [(11,'left_shoulder'),(12,'right_shoulder')]:
   if not valid[k]:all_causes.append('body_normalizer_raw_'+name+'_invalid')
   elif not inside[k]:all_causes.append('body_normalizer_'+name+'_outside_roi')
  if not c['scale_ok'][j]:all_causes.append('body_normalizer_scale_invalid')
  assert all_causes,(e['event_id'],int(f[j]))
  primary=all_causes[0]
  causes[primary]+=1
  side_causes=[]
  if valid[54] and not inside[54]:
   x,y=c['xy'][j,54];b=c['r']['box']
   if x<b[0]:side_causes.append('left')
   if x>=b[2]:side_causes.append('right')
   if y<b[1]:side_causes.append('top')
   if y>=b[3]:side_causes.append('bottom')
  missing.append(dict(frame_id=int(f[j]),timestamp_us=int(a['timestamp_us'][j]),
    primary_cause=primary,all_mask_causes=all_causes,
    hand_wrist_roi_sides=side_causes,
    raw_hand_wrist_observed=bool(observed[54]),raw_hand_wrist_usable=bool(usable[54]),raw_hand_wrist_finite=bool(finite[54]),
    raw_right_hand_observed_points=int(observed[54:75].sum()),
    raw_right_hand_valid_points=int(valid[54:75].sum()),
    raw_right_hand_roi_valid_points=int((valid&inside)[54:75].sum()),
    body_normalizer_raw_valid=bool(c['body_raw'][j]),body_normalizer_roi_valid=bool(c['body_roi'][j]),
    body_pose_right_wrist_raw_valid=bool(valid[16]),body_pose_right_wrist_inside_roi_if_valid=bool(inside[16]) if valid[16] else None,
    body_pose_wrist_is_diagnostic_context_only=True))
 raw_eligibility=(c['valid'][:,54]&c['body_raw'])[ix]
 raw_ok=n>=3 and raw_eligibility.mean()>=.75 and raw_eligibility[:min(3,n)].any() and raw_eligibility[-min(3,n):].any()
 reject=[]
 if n<3:reject.append('fewer_than_3_native_frames')
 if cov<.75:reject.append('hand_wrist_support_below_75_percent')
 if not s[:min(3,n)].any():reject.append('no_supported_sample_in_first_3_core_frames')
 if not s[-min(3,n):].any():reject.append('no_supported_sample_in_last_3_core_frames')
 rec=dict(alias=e['alias'],event_id=e['event_id'],role=e['role'],state=e['state'],
  core=[e['core_start_frame'],e['core_end_frame']],core_timestamp_us=[int(a['timestamp_us'][ix[0]]),int(a['timestamp_us'][ix[-1]])],
  native_frames=n,eligible=bool(eligible),supported_frames=int(s.sum()),coverage=cov,rejection_reasons=reject,
  unsupported_frame_primary_causes=dict(causes),
  diagnostic_raw_without_roi_coverage=float(raw_eligibility.mean()),diagnostic_raw_without_roi_eligible=bool(raw_ok),
  roi_alone_prevents_eligibility=bool(raw_ok and not eligible),
  saved_feature_supported_frames={name:int(a['mask'][ix][:,cols].all(1).sum()) for name,cols in [('actual_hand_wrist',[28,29]),('body_pose_wrist',[30,31]),('body_elbow',[32,33]),('hand_palm_center',[36,37])]},
  missing_frame_evidence=missing)
 records.append(rec)
 by_role[e['role']]['events']+=1
 by_role[e['role']]['eligible']+=int(eligible)
 by_role[e['role']]['unsupported_events']+=int(not eligible)
 if not eligible:
  by_role[e['role']]['unsupported_'+e['state']]+=1
  by_role[e['role']]['roi_alone_prevents_eligibility_events']+=int(raw_ok)
  all_missing.update(causes)

report=dict(schema='wis_native_input_support_diagnosis_v1',status='PASS_SUPPORT_REPRODUCTION_AND_CAUSE_ACCOUNTING',
 reviewer='Same-Codex agent rgb_labels_a; independent arithmetic from saved raw arrays, not a detector truth audit',
 inputs={'prepared':str(U/'PREPARED.json'),'prepared_sha256':sha(U/'PREPARED.json'),
  'episode_py_sha256_read_only':sha(U/'episode.py'),
  'read_fields':['raw xy_px/observed/usable/frame_id/timestamp_us','frozen x/mask/frame_id/timestamp_us','AUTHORIZATION alias/role/input/observation/box','PREPARED event support metadata'],
  'candidate_predictions_read':False,'previous_scores_read':False,'model_inference':0,'training':0},
 support_contract={'joint_layout':'Pose33 + LHand21 + RHand21; actual R wrist joint54, body R wrist joint16',
  'formula':'observed & usable & finite for joint54 and both shoulders11/12, all within fixed half-open ROI; shoulder width >1e-8; equals saved feature mask28/29',
  'event_eligible':'at least3 native frames, actual wrist support>=0.75, >=1 supported sample in first3 and last3',
  'other_parts':'body wrist/elbow and palm availability recorded separately; they do not replace a missing actual hand wrist',
  'counterfactual':'raw_without_roi is diagnosis only; it does not authorize removing/changing frozen ROI or training on excluded frames'},
 counts_by_role={k:dict(v) for k,v in by_role.items()},
 unsupported_event_missing_frame_causes=dict(all_missing),
 all_native_support_masks_reproduced=True,all_prepared_event_support_reproduced=True,
 per_alias_input_pins={k:{**c['pins'],'box':c['r']['box'],'native_frames':len(c['a']['frame_id'])} for k,c in cache.items()},
 events=records,
 limitations=['A raw unobserved hand is a detector-observation absence, not proof the hand was physically invisible or the motion wrong.',
 'When actual hand joint is absent, a body wrist outside ROI is contextual proxy evidence only; do not recast hand absence as proven ROI exclusion.',
 'Visible natural rest or hand lowering outside the fixed ROI is input-range exclusion, never action failure.',
 'The RGB labels remain sampled provisional judgments; support arithmetic cannot certify their semantic correctness.',
 'No candidate outputs or saved performance results were opened; existing source/code/labels unchanged.'])
unsupported=[r for r in records if not r['eligible']]
unobs=[f for r in unsupported for f in r['missing_frame_evidence'] if f['primary_cause']=='raw_right_hand_wrist_unobserved']
outroi=[f for r in unsupported for f in r['missing_frame_evidence'] if f['primary_cause']=='valid_right_hand_wrist_outside_roi']
assert len(unobs)==35 and len(outroi)==91
assert all(f['raw_right_hand_observed_points']==0 and f['raw_right_hand_valid_points']==0 for f in unobs)
assert all(f['body_normalizer_roi_valid'] and f['body_pose_right_wrist_inside_roi_if_valid'] for f in unobs)
assert all(f['raw_right_hand_observed_points']==21 and f['raw_right_hand_valid_points']==21 and f['hand_wrist_roi_sides']==['bottom'] for f in outroi)
report['findings']={
 'unsupported_up_events':{'fit':11,'development':3,'direct_cause':'raw right-hand detector wrist unobserved; all21 right-hand points unobserved in these35 missing frames','ROI_removal_would_restore_event_eligibility':False,'body_wrist_and_elbow_supported_through_all_these_cores':True},
 'unsupported_other_events':{'development':5,'direct_cause':'valid raw hand wrist below fixed ROI bottom in all91 core frames','all21_raw_right_hand_points_valid_in_these_frames':True,'without_ROI_diagnostic_coverage':1.0,'ROI_removal_is_not_authorized':True},
 'other_body_normalizer_or_nonfinite_cause_frames_in_unsupported_events':0,
 'wording_correction':{'fit_unsupported_initial_rises':8,'fit_unsupported_second_rises':3,'second_rise_event_ids':['P063_01','P064_01','P065_01'],'note':'The11 unsupported fit rises are not all initial rises.'},
 'rejection_criteria':{'coverage_below_75_percent_events':19,'first3_empty_events':5,'last3_empty_events':5,'shorter_than3_events':0},
 'interpretation':'These are input support limits, not evidence of wrong movement. RGB visibility, detector availability, ROI membership and correctness are separate.'
}
(OUT/'INPUT_SUPPORT_DIAGNOSIS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=[
 '58개 주석창의 입력 지원을 원본 observed/usable/finite와 고정 ROI에서 별도 계산했다. PREPARED와 전부 일치했다: fit 42개 중31개 지원, dev 16개 중8개 지원. 후보 출력이나 이전 점수는 읽지 않았다.',
 '',
 '미지원 상승은 fit11개와 dev3개다. 원인은 ROI가 아니라 원본 오른손 관측 누락이다. 해당35개 프레임에서 오른손21점의 observed가 모두 false였다. ROI를 제거한다고 가정해도 이14개 창은 지원조건을 통과하지 못한다. 몸 관측의 오른손목과 팔꿈치는 이 상승 core 전프레임에서 지원되지만, 별도 추정원이므로 실제 손 관측의 대체 정답으로 채우면 안 된다.',
 '',
 '미지원 other는 dev5개다. 이91개 프레임에는 원본 오른손21점이 모두 관측·사용 가능했으나 실제 손목이 고정 ROI 하단 밖이었다. ROI 없는 계산에서는 지원율100%지만 이는 원인 분류용 가정이며 ROI 변경 승인이 아니다. 몸 옆에서 쉬거나 손을 내린 뒤 ROI 밖에 있는 것은 입력범위 제한이다. 정상적인 손 내림이나 휴식을 동작 실패로 부를 근거가 없다.',
 '',
 '어깨 정규화, 손목 nonfinite, 다른 신체부위 마스크가 원인인 미지원 프레임은0개다. 19개 미지원 창 모두 손목 지원율75% 미만이며, 그중 ROI 밖 other5개는 첫3/마지막3 프레임 지원도0이다. 최소3프레임 조건 때문에 탈락한 창은 없다.',
 '',
 '표현 정정: fit 미지원 상승11개는 초기 상승8개와 두 번째 상승3개(P063_01/P064_01/P065_01)다. 전체 fit 상승15개 중 지원되는 것은4개다. 이는 지원 분모이며 모델 성능 수치가 아니다.',
 '',
 '별칭/창 | 역할/상태 | core(frame, inclusive) | 지원 | 직접 원인 | 미지원 원본 frame_id'
]
for r in unsupported:
 cause='원본 오른손 전체 관측 누락' if r['state']=='up' else '유효 손목이 ROI 하단 밖'
 ids=[f['frame_id'] for f in r['missing_frame_evidence']]
 ids_text=','.join(map(str,ids)) if len(ids)<12 else f'{ids[0]}~{ids[-1]} 전부'
 lines.append(f"{r['event_id']} | {r['role']}/{r['state']} | {r['core'][0]}~{r['core'][1]} | {r['supported_frames']}/{r['native_frames']} | {cause} | {ids_text}")
lines += ['', '이 진단은 같은 Codex의 독립 수치 재계산이다. 원본 손 관측 누락이 실제 가림, 검출 생략, 검출 실패 중 무엇인지는 mask만으로 단정하지 않는다. 손이 RGB에 보이는 것, 원본 관절이 관측된 것, ROI 안에 있는 것, 올바른 수어인 것은 각각 별개다. 주석은 표본 RGB 초안이며 dense GT가 아니다. 기존 코드·라벨·원본은 변경하지 않았고 추론·학습을 실행하지 않았다.']
(OUT/'INPUT_SUPPORT_DIAGNOSIS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'counts':report['counts_by_role'],'missing_causes':dict(all_missing),'unsupported':[{k:r[k] for k in ['event_id','role','state','core','native_frames','supported_frames','unsupported_frame_primary_causes','diagnostic_raw_without_roi_coverage','roi_alone_prevents_eligibility','saved_feature_supported_frames']} for r in records if not r['eligible']]},ensure_ascii=False))
