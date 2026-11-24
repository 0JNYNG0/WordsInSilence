"""Read-only code/saved-result audit; clarify the user's intended next path."""
import json, hashlib, importlib.util, shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
OLD=O/'word100_body_boundary_20261017'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (U/'AUDIT.json').exists()
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
protected=read(O/'word100_current_image_redetection_20261020/FROZEN.json')['protected']
for p,h in protected.items():assert sha(p)==h,p
sources=[
    OLD/'training_core.py',OLD/'training_driver.py',OLD/'training_COMPARISON.json',
    OLD/'training_scores_INDEPENDENT.json',
    O/'word100_roi_training_20261017/prep.py',
    O/'word100_roi_training_plan_20261017/PROPOSED_EXPERIMENT.json',
    ROOT/'research/front_keyphase_v1/src/wis_lab/service/camera.py',
    ROOT/'research/front_keyphase_v1/src/wis_lab/service/word100_candidate.py']
pins={str(p):sha(p) for p in sources}
comparison=read(OLD/'training_COMPARISON.json')
for p,h in comparison['result_pins'].items():assert sha(p)==h,p
spec=importlib.util.spec_from_file_location('readonly_schedule',OLD/'training_core.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
ids=[r['clip_id'] for r in read(O/'word100_current_image_redetection_20261020/FROZEN.json')['rows']]
schedule={cid:{arm:[list(core.schedule(cid,e,arm)) for e in range(1,5)] for arm in ['B','C']} for cid in ids}
for value in schedule.values():
    assert all(x[0]=='masked' for x in value['B'])
    assert sorted(x[0] for x in value['C'])==['masked','masked','sensor','sensor']
driver=(OLD/'training_driver.py').read_text('utf8')
assert "xs = [r['arrays'][s][t] for r, (s,t) in zip(rows, choices)]" in driver
camera=(sources[-2]).read_text('utf8')
assert 'self._candidate_lm.process(rgb, ts_us, crow, 0)' in camera
assert 'self.raw.process(rgb, ts, out, row)' in sources[-1].read_text('utf8')
metrics={arm:{view:comparison['arms'][arm][view]['top1_percent']
              for view in ['masked','sensor']} for arm in ['B','C']}
result=dict(status='CODE_AND_SAVED_RESULTS_AUDITED_USER_PRIMARY_PATH_CLARIFIED',at=at,
    scope='Code, saved summaries, hashes only; no source pixels/user recordings/model inference.',
    existing_training=dict(B='whole-source detection then fixed ROI mask only',
        C='alternating masked and independently cropped sensor examples across epochs',
        same_frame_cross_view_hand_fusion=False,schedule_four_epochs=schedule),
    metrics=metrics,metric_scope='Existing DEV300 /3 seeds; reused selection set, not new-user or independent final test.',
    causality='TCN training cannot change the earlier cached MediaPipe detections in this pipeline. Cropping pixels before detection can change detection. Mixing training views may affect classification; existing mean scores do not establish harm.',
    current_web='Each detector receives the full available webcam RGB frame; it does not fuse offline masked/sensor coordinate streams. Physically out-of-camera pixels remain unavailable.',
    user_intent='Whole available source image -> continuous detection/assignment/current-image re-detection -> resolve candidates in this same coordinate system -> fixed learning ROI mask -> features. No sensor examples mixed into future training by default.',
    next_merge_status='Not implemented in this clarification audit. Previous sensor-only redetection proposals must not be pasted into native detections.',
    old_models_preserved=True,source_pins=pins,protected_files=len(protected),
    classifier_inference=0,training=0,new_detection=0,user_recording_reads=0,production_changes=0,
    verification='Codex static inspection + saved schedule arithmetic and saved-result pins; no other reviewer.')
write(U/'AUDIT.json',result)
text="""전체 영상 검출과 학습 범위 제한 — 입력 경로 확인

사용자 의도
원본 전체 화면에서 처음부터 끝까지 관절을 검출·추적하고, 학습에 넣을 때 고정된 상반신 범위 밖 관절을 제외한다.
미검출 재탐색과 후보 중복 확인 역시 같은 원본 화면 좌표계에서 먼저 수행한다.
다음 개발의 기본 경로는 이 방식으로 정리한다. 제한 화면 검출 자료를 미래 학습에 자동으로 혼합하지 않는다.

현재 코드에서 확인한 사실
기존 B 조건이 사용자의 설명과 일치한다.
기존 C는 B 자료와 별도로 화면을 먼저 제한해 검출한 sensor 자료를 학습 시 번갈아 제공했다.
한 프레임의 왼손은 한 검출 경로에서, 오른손은 다른 검출 경로에서 가져와 합친 방식은 아니다.
training_core.schedule은 B를 항상masked로, C를4epoch당masked2/sensor2로 선택한다.
training_driver.torch_batch는 영상마다 선택된 하나의 완성 배열만 읽는다.
이는 영상 추출/좌우배정과 그 뒤 TCN 학습을 구분해서 이해해야 한다.

혼합 때문에 오류가 생겼나?
최근 손 미검출/좌우 문제는 TCN 학습 이전에 저장된 관절 검출 자료에 이미 존재했다.
이 코드에서는 TCN 결과가 MediaPipe로 되돌아가 좌우를 바꾸지 않는다.
따라서 혼합학습이 앞단 MediaPipe 오류를 만들었다는 해석은 맞지 않는다.
단, 화면을 먼저 제한하면 보이는 픽셀과 추적 이력이 달라져 검출 문제가 추가될 수 있다.
이 검출 차이와 두 종류 자료로 TCN을 학습한 효과를 분리해야 한다.

이미 있는 비교 결과 재사용 — 학습899/개발300,각3seed평균
전체검출 후 영역제한 평가: B98.11%, C98.44%.
화면먼저제한 후 검출 평가: B97.56%, C98.67%.
같은 개발자료에서 평균Top1은 C가 조금 높았지만 개별 회귀도 있고, 모델 선택에 반복 사용한 자료다.
혼합이 모든 상황에서 이득이라는 보장도, 제거하면 성능이 오른다는 근거도 아니다.
이번에 새 학습·새 분류 평가는 하지 않았다.

웹캠과의 차이
웹 코드는 카메라가 제공한 전체 RGB를 검출기에 전달한다. 두 오프라인 경로를 촬영 중 섞지 않는다.
원본 공개영상에는 카메라 아래쪽 손이 있어도 실제 웹캠에는 그 픽셀이 없을 수 있다.
전체 영상 검출 방식을 기준으로 보완하되, 실제 카메라 화면 밖 손을 관측한 것처럼 처리하면 안 된다.

다음 작업의 경계
기존 C·T3-F/기존B/이전결과를 보존한다.
원본전체 검출의 기존 손과 같은 원본에서 재탐색한 손을 먼저 비교하고, 동일 실제 손의 중복/좌우 충돌을 분리한다.
그 이후에만 고정 학습ROI를 적용한다. 범위밖 관절을 모델입력으로 흘리지 않고 손가락별 미관측을 유지한다.
이전sensor에서 얻은294/303후보를native에그대로끼워넣지않는다.
sensor는 실제웹캠제약의 별도 진단조건으로 분리하며 신규혼합학습은자동진행하지않는다.
추가 학습과 운영교체는 하지 않았다. 이번은 경로감사·설명정정·다음작업기준명확화이며 합치기보완구현완료가아니다.
사용자·멘토 촬영의 학습/튜닝/후보선택 금지 유지.
"""
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
(U/'before').mkdir()
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
    shutil.copyfile(p,U/'before'/p.name)
next_action=('사용자 의도대로 원본 전체 화면 연속 검출·좌우 배정·같은 화면 재검출 후보 통합을 먼저 하고, '
             '그 뒤 고정 상반신 ROI를 적용하는 경로를 다음 보완의 기본으로 삼는다. '
             '기존 C/T3-F 및 과거 B/C 결과는 보존. sensor 혼합을 새 학습에 자동 추가하지 않으며 별도 웹캠제약 진단으로 구분한다. '
             '중복 후보 통합은 아직 미구현; 이전 sensor 후보를 native에 섞지 말고 native 기준 공개 대조 조건부터 고정한다. '
             '정상 교차·기존 정상·영역밖 관절 배제·손가락별 미관측을 검증. 추가 학습은 사용자와 결과 확인 후, 사용자/멘토 자료 학습·튜닝·선택 금지.')
title='### 2026-10-20 사용자 입력 경로 의도 확인 — 전체 검출 후 ROI를 다음 보완 기본으로 명확화'
entry=title+"""
- `ops/word100_input_path_clarification_20261020/AUDIT.json`, `RESULT_KO.txt`. 코드/저장 결과 감사만 완료. 합치기 후보 구현·새검출·학습·분류평가0.
- 사용자는 원본전체검출→범위안관절만학습을 생각했고 혼합사용의도를 부인. 다음보완기본경로를 full/native 검출·같은원본재검출후중복통합→고정ROI→특징으로 명확화. sensor를 미래학습에자동혼합하지않음.
- 기존B=masked전용,C=4epoch당masked2/sensor2. torch_batch는영상별하나의배열을선택,같은프레임의좌우를두경로에서합치는방식아님. 최근미검출/오배정은TCN학습전관절자료문제이며혼합TCN이앞단MediaPipe를변경하는경로없음.
- 화면선제한이관측/추적차이를만들수있다는점과그자료의혼합학습효과는별개. 기존DEV300각3seed평균(masked B98.11/C98.44,sensor B97.56/C98.67)재사용. 반복사용DEV이며혼합의보편적우월/제거효과증명아님.
- 웹은현재카메라전체RGB를검출기에전달.원본공개영상의추가픽셀을실웹캠에서관측한것처럼쓸수는없음. sensor는별도제약진단으로구분.
- C/T3-F/기존B·운영웹·DTW불변,32보호파일해시유지. 사용자/멘토촬영접근0. 다음통합보완미구현상태를성공으로표현하지않음. 이전sensor재검출294/303을native입력에재사용금지.

"""
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';old=p.read_text('utf-8-sig')
for heading,insert in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',entry),
                       ('## 9. 다음 단일 작업',title+'\n'+next_action+'\n\n')]:
    assert old.count(heading)==1;old=old.replace(heading+'\n',heading+'\n\n'+insert,1)
p.write_text(old,encoding='utf8')
state=read(O/'STATE.json');state['current_stage']='WORD100_FULL_DETECTION_PRIMARY_PATH_CLARIFIED'
state['updated_date_kst']='2026-10-20';state['next_single_action']=next_action
state[U.name]={**result,'schedule_note':'36 known TRAIN clip IDs over4epochs only; see AUDIT.json for schedule','next_single_action':next_action}
state[U.name]['existing_training']=dict(result['existing_training'])
state[U.name]['existing_training'].pop('schedule_four_epochs')
write(O/'STATE.json',state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f"""# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 사용자 의도인 전체 영상 검출과 학습ROI 제한을 분리한 경로에서 관측을 보완한다.
마지막 완료: 실제B/C 코드·저장점수·웹입력 경로 감사. 다음 작업의 기본 경로를 명확히 기록했다.
다음 한 작업: {next_action}

- 최신 단위 `{U.as_posix()}`. `AUDIT.json`/`RESULT_KO.txt`.
- 사용자 의도: 원본전체크기로MediaPipe를끝까지돌리고학습에는영역안관절만. 혼합사용을원래생각하지않았다고명시.
- 기존B=이방식,기존C=B와별도sensor자료번갈아학습. 4epoch masked2/sensor2,같은프레임에서두경로손합성아님.
- 학습전저장관절에이미존재하는검출오류를혼합TCN탓으로인과주장금지. 화면선제한검출변화와혼합학습효과를분리.
- 기존공개DEV300각3seed평균: masked B98.11/C98.44, sensor B97.56/C98.67. 저장결과재사용/새평가아님. 반복DEV,새사용자성능아님.
- 현재웹camera와word100_candidate는카메라전체RGB검출. 실제화면밖손은불가시이며공개원본전체의장점을웹캠에보장하지못함.
- 합치기보완은아직미구현. 이전재검출단위는sensor픽셀기준이므로새native경로에그후보를끼워넣지않는다.
- 직전 `word100_current_image_redetection_20261020`:36TRAIN/23단어/9명/4641프레임,추가후보전체294/국소303(본동작76/77),308고유시각검토. target307공항09f41기존오배정과동일손중복1,형상왜곡/화면밖손가락반례로미채택.
- 직전초기감사:원본36모두첫프레임손후보2,제한화면35개0손/1개1손. 알려진오류12는sensor. 좌우완전정답보장아님.
- 직전배속365같은픽셀프레임timestamp1/.75/.5/1,손수/좌우/XY차0. 재생시간변경으로추가정보생기지않음.
- 현재단위검증:일정코드36ID×4epoch,배치선택정적확인,저장학습결과핀·32보호파일확인. 같은Codex,다른검토자아님.
- C·T3-F/기존B/운영DTW/웹유지. 새검출/분류평가/학습/사용자자료접근/다운로드/카메라/Git0. 이번은경로확인과기록이지입력개선완료아님.
- 명령종료,현재단위실행프로세스없음. 원본·과거동결결과보존. 사용자/멘토촬영학습·튜닝·선택금지.
""",encoding='utf8')
for p,h in {**pins,**protected}.items():assert sha(p)==h,p
assert read(O/'STATE.json')['current_stage']=='WORD100_FULL_DETECTION_PRIMARY_PATH_CLARIFIED'
write(U/'VERIFY.json',dict(status='PASS',source_hashes_unchanged=len(pins),protected_files_unchanged=len(protected),
    saved_result_hashes_checked=len(comparison['result_pins']),schedule_clip_ids=len(ids),
    same_frame_cross_view_fusion=False,root_documents_synchronized=True,
    new_training=0,new_detection=0,user_data=0))
print('PATH AUDIT PASS',json.dumps(metrics,ensure_ascii=False))
