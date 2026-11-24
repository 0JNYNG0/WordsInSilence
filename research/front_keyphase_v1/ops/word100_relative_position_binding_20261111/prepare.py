from relative_core import *
from collections import Counter
from datetime import datetime,timezone,timedelta

def main():
 assert not (U/'PREPARED.json').exists()
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']:
  d=U/'before'/p.name;d.parent.mkdir(exist_ok=True);d.write_bytes(p.read_bytes())
 protected=read(Q/'PROTECTIONS.json')
 for pat in ['*.py','*.json','runs/*/*.pt']:protected.update({str(p):sha(p) for p in Q.glob(pat)})
 write(U/'PROTECTIONS.json',protected)
 write(U/'AUTHORIZATION.json',dict(user='신체 기준 시작·끝 보존과 설명 구간 연결 계획 모두 진행',public_only=True,private_data=False,C_T3F_preserved=True,camera=False,downloads=False,external_transfer=False,implementation='Codex',independent_human=False))
 labels=read(Q/'LABELS.json');roles=read(R/'ROLE_LABELS.json');fitsha={r['source_sha256'] for r in labels['rows'] if r['role']=='fit'}|{r['source_sha256'] for r in roles['rows']};devsha={r['source_sha256'] for r in labels['rows'] if r['role']!='fit'}|{r['source_sha256'] for r in read(Q/'EXTRA_EVAL_LABELS.json')['rows']};assert not fitsha&devsha
 # Reuse reviewed labels. Generic background means outside the annotated action,
 # not proof that the limb is physically stationary. Mask its quality supervision.
 samples=read(Q/'TRAIN_SAMPLES.json');source={r['clip_id']:r for r in read(P/'PUBLIC_CACHE.json')['rows']};arr=dict(np.load(Q/'train_arrays.npz'));pos=np.zeros((len(arr['quality']),20),np.float32);cache={};manifest=[]
 for r in samples:
  if 'sample' not in r:continue
  clip=r['clip_id'];src=source[clip];assert src['role']=='fit'
  if clip not in cache:assert sha(src['path'])==src['sha256'];cache[clip]=dict(np.load(src['path']))
  z=cache[clip];e=dict(a=r['a'],b=r['b'],axis=r['axis'],moving=r['side']);pos[r['sample']]=position_features(z[r['view']],z['ts']/r['speed'],e)
  if r['kind']=='background':arr['quality_mask'][r['sample']]=0
  manifest.append(dict(r,physical_quality_mask=float(arr['quality_mask'][r['sample']]),physical_label_reason='Outside reviewed target, physical motion not labeled' if r['kind']=='background' else 'Reviewed physical event or derived interval; not semantic sign completion'))
 mask=pos[:,16:].repeat(4,axis=1);mean=np.sum(pos[:,:16]*mask,0)/np.maximum(mask.sum(0),1);std=np.sqrt(np.sum((pos[:,:16]-mean)**2*mask,0)/np.maximum(mask.sum(0),1));norm=dict(mean=mean.tolist(),std=np.maximum(std,.05).tolist())
 arr['position']=pos;np.savez_compressed(U/'completion_train.npz',**arr);write(U/'TRAIN_MANIFEST.json',manifest)
 records=[r for r in read(R/'TRAIN_INPUTS_COMPLETE.json')['rows'] if r['role_frames']>0]
 for r in records:assert 1<=r['person']<=9 and r['source_sha256'] not in devsha and sha(r['path'])==r['sha256']
 write(U/'ROLE_INPUTS.json',records)
 audit=dict(physical_videos=len(cache),physical_event_labels=sum(len(r['events']) for r in labels['rows'] if r['role']=='fit'),physical_samples=len(pos),background_quality_masked=sum(r['kind']=='background' for r in manifest),role_videos=len({r['clip_id'] for r in records}),role_classes=roles['roles'],independent_human=False,label_policy='Physical event quality and guide action role are separate. Generic background is not forced to stationary. Existing reviewed role anchors and weak interpolation kept separate.',missing_left_axis2=True,primary_guards='No expected direction or desired answer enters physical observation or role selection.',public_source_separation=True,review_rechecked=[str(R/'review/NIA_SL_WORD1133_REAL02_F.jpg')],fresh_inputs='Selected caches comprise previously exposed public training/development videos. No new unseen test set claimed.')
 write(U/'LABEL_AUDIT.json',audit)
 cfg=dict(seeds=[42,43,44],completion_modes=['control','position'],completion_epochs=40,completion_lr=.0003,role_epochs=20,role_lr=.0005,completion_threshold=.8,role_threshold=.65,role_coverage=.65,norm=read(Q/'FROZEN.json')['norm'],position_norm=norm,primary='position_and_binding',comparisons=['previous','control_only','position_only','binding_only','position_and_binding'],no_evaluation_tuning=True,common100_frozen=True,gates=['whole_recall_not_lower_per_side','partial_false_not_higher','normal_speed_crossing_no_new_warning','normal_similarity_preserved','existing_shape_direction_warnings_retained','observation_missing_not_normal','all_three_seeds'],pins={str(p):sha(p) for p in [U/'relative_core.py',U/'prepare.py',U/'TRAIN_MANIFEST.json',U/'ROLE_INPUTS.json',Q/'LABELS.json',Q/'TRAIN_SAMPLES.json',R/'ROLE_LABELS.json']})
 write(U/'FROZEN.json',cfg);write(U/'PREPARED.json',audit);write(U/'CURRENT_EXECUTION.json',dict(status='PREPARED',pid=None))
 plan='''1. 物理的な移動と意味上の完了を区別する。'''
 plan='''이번 목표: 정상 이동을 놓치지 않으면서 부분 이동을 전체로 오인하지 않고, 실제 움직임을 올바른 설명 구간에 연결한다.
1. 기존 공개 원본 주석의 뜻을 감사한다. 물리적 전체/부분과 설명의 주요/준비/복귀를 분리하고, 주요 행동 밖이라는 이유로 움직임 없음으로 가르치지 않는다. FIT/DEV 기록ID·원본해시 분리. 실제 왼손 상대팔 축은 현재 부족하다고 명시한다.
2. 기존 완료 판정부의 시간 변화량에 정규화된 몸 기준 시작/끝/앞뒤 관계와 관측마스크를 추가한다. 수정 정답만 같은 횟수 학습한 대조군을 함께 둔다. 공통100단어·기존 모양/방향 모델은 동결한다.
3. 기존 짧은 창 역할 판정에 전체 시간 흐름을 보는 작은 Conv+GRU 잔차 역할부를 학습한다. 기존 직접 검토 정답과 약한 보간 정답의 가중치를 구분한다. 정상 영상의 순서를 바꾼 입력에는 바뀐 실제 행동 정체성을 가르치며 올바른 수어라는 정답을 주지 않는다. 기대 방향으로 후보를 골라 정답을 숨기지 않는다.
4. 이전/대조군/위치만/연결만/둘다를 조건 고정 후 비교. 직접 구간100개·자동78조건·다른6표현24조건·기존왼손 추가22구간을 평가한다. 과거 노출 자료임을 명시하고 새 실제 사용자 성능으로 일반화하지 않는다. 세 초기값 모두 회귀 조건 통과 시에만 채택 검토. 미통과 시 웹 보존,임계값후조정/재추론없음.
'''
 (U/'PLAN_KO.txt').write_text(plan,'utf-8')
 state=read(O/'STATE.json');state.update(current_stage=U.name,last_updated=datetime.now(timezone(timedelta(hours=9))).isoformat(),next_single_action='신체상대위치+설명역할잔차학습→고정비교→검산→통과기능만적용');state[U.name]=dict(status='PREPARED',unit=str(U),private_data=0);write(O/'STATE.json',state)
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계\n\n사용자 승인 후 신체 상대 위치·설명 연결 보완 진행 중.\n현재 '+str(U)+'\nPLAN_KO.txt/FROZEN.json의 조건을 따름. 이전 완료 결과 반복 실행 금지. C·T3-F/공통100단어/기존웹 보호. 사용자·멘토자료 접근/학습/튜닝/선택 금지.\n','utf-8');event('prepared',**audit)

if __name__=='__main__':main()
