from core import *
from sync_progress import sync
from datetime import datetime,timezone,timedelta

ev=read(U/'EVALUATION_COMPLETE.json');v=read(U/'FINAL_CHECK.json');assert v['status']=='PASS'
original=read(S/'scores/original.json');new=read(U/'scores/shared100_42_records.json')
def key(r):return (r['clip_id'],r['frame'])
old_stage={key(r):r for r in original if r['variant']=='normal' and r['task']=='semantic_stage'}
new_stage={key(r):r for r in new if r['variant']=='normal' and r['task']=='semantic_stage'}
assert old_stage.keys()==new_stage.keys()
changed=[]
for k,a in old_stage.items():
 b=new_stage[k];assert a['target']==b['target']
 if a['pred']!=b['pred']:changed.append(dict(clip_id=k[0],frame=k[1],target=a['target'],old=a['pred'],new=b['pred'],old_correct=a['pred']==a['target'],new_correct=b['pred']==b['target']))
primary={name:ev['models'][name] for name in ['labels_only_42','shared100_42']};diag=read(U/'DIAGNOSTICS.json');tail=read(U/'TAIL_ALIGNMENT_CHECK.json')
stage_change=dict(changed=changed,recovered=sum(not r['old_correct'] and r['new_correct'] for r in changed),regressed=sum(r['old_correct'] and not r['new_correct'] for r in changed))
write(U/'STAGE_CHANGE_AUDIT.json',stage_change)
result=dict(status='COMPLETE_PARTIAL_SHAPE_GAIN_REPLAY_FIX_NO_WEB_MODEL_REPLACEMENT',input_scope=dict(words=100,fit=899,development=300,fit_record_ids=list(range(1,10)),dev_record_ids=[11,12,13],person_identity_beyond_record_ids_verified=False,additional_fit_feature_cells=166,additional_reviewed_dev_cells=133,additional_ambiguous_dev_cells=83),primary=primary,all_seeds={k:dict(normal=z['by_variant']['normal'],aux_word=z['aux_word'],latency=z['latency']) for k,z in ev['models'].items()},stored_original=ev['stored_original_baseline']['by_variant']['normal'],stage_changes=stage_change,protected=dict(C=True,T3F=True,current_TCN=True,deployed_feedback_model=True,private_access=False),replay=dict(code_applied=True,logic_checks=8,browser_checks=8,actual_user_camera_check=False,alignment_algorithm_changed=False),tail_diagnostic=dict(public_clips=12,main_changed=0,late_tail_end_mapping_changed=sum(r['all_original_shift_max_frames']>0 for r in tail['rows'] if r['variant']=='late_action'),scope='synthetic cached-feature tail, not a replay of private mouse action'),verification=str(U/'FINAL_CHECK.json'),next='Preserve the improved common handshape learner as a candidate; separately supervise action phases/short events and validate its connection to the existing stage/evidence heads before web adoption. No automatic tail removal or replacement of TCN.')
write(U/'RESULT.json',result)
text='''공개100단어 공통 학습 및 비교 재생 보완 결과

이번 결론
넓은 자료로 실제 신경망 학습을 마쳤고 손모양 판정은 일부 좋아졌다. 동작 단계를 구분하는 성능은 좋아지지 않아 기존 웹 모델을 교체하지 않았다. 비교 영상이 튀어 보일 수 있던 재생 코드 문제는 별도로 수정하고 공개 영상으로 확인했다.

1. 예전 화면과 무엇이 달랐나
구8812는 사용자 영상만 재생하고 참고 영상을 DTW 위치로 이동시켰다. 새8814는 두 영상을 각각 재생하면서 참고 영상에 DTW 이동 명령도 반복해서 보냈다. 참고 영상이 스스로 앞으로 가는 동작과 정렬 위치로 되돌리는 동작이 겹칠 수 있었다. 새 화면은 사용자 영상 하나의 시간을 기준으로 움직이도록 수정했다. 이동 중에는 다음 요청을 모아 최신 위치만 적용하고 관절 그림도 실제 표시된 영상 시점에 맞춘다. DTW 계산이나 판정값을 매끄러워 보이게 바꾸지는 않았다.
로직8항목, 실제 공개 영상 브라우저8항목을 통과했다. 개인 촬영 재현 또는 모든 환경의 매끄러운 재생을 확인한 것은 아니다. 모델·TCN·촬영 종료 조건은 그대로다.

2. 무엇을 학습했나
기존 TCN과 같은 공개100단어 범위를 사용했다. 학습899영상(기록ID01~09), 비교300영상(11~13), 원본SHA 중복 없음. 여기서 ID는 자료의 기록 식별자이며 100단어 전체의 실제 인물 동일성을 얼굴로 확인했다는 뜻이 아니다.
기존4단어의 시간별 특징·단계 정답에, 8개 추가표현의 보이는 손모양166항목을 보강했다. 왼손 펴기 양성 정답은7→28항목, 1→4표현, 기록ID3종→5종으로 늘었다. 각 항목은 영상 시점과 손 특징 하나이며 영상28개라는 뜻이 아니다.
조건B: 이 추가정답으로 현재 시간 합성곱+GRU를 미세조정.
조건C: B와 동일한 정답에 더해100단어899영상의 단어별 움직임을 보조 과제로 함께 학습.
각 조건을 사전에 고정한3시드로24회씩 학습했다. 대표값은 미리 정한42이며 잘 나온 시드를 고르지 않았다. 전체100단어의 세부 행동 설명·모든 프레임 정답을 완성한 것은 아니다. 가림은 확인된 원래 동작에서 센서값2프레임만 숨기는 방식으로 연습했고, 틀린 순서/빠진 동작을 정상으로 가르치거나 추정 관절을 실제 정답으로 쓰지 않았다. CPU 학습이며 기존 RTMW 추출 결과를 재사용했다.

3. 실제 비교 결과 — 사전에 정한 대표 실행
                         기존 웹 몸통모델    B 정답보강    C 공통100단어 추가
기존4단어 손모양 항목      221/228            225/228       226/228
기존4단어 동작 단계         72/84              71/84         71/84
기존4단어 정상 순서         10/12               9/12         10/12
추가8단어 손모양 항목       이번 미측정         119/133       122/133

228은4단어의 개발 영상12개에서 확인한 손모양 항목 수이고,84는 그 영상의 단계 확인 시점 수,12는 순서를 확인한 쌍 수다. 133은 추가8단어·개발24영상·72시점에서 확실히 확인한3종 손 특징 항목 수다. 가능한216항목 중83은 애매하거나 화면 밖이라 정답 확정에서 제외했다. 모델 출력과 무관하게 먼저 구분했고 그83을 맞았다고 세지 않았다. 전문가 독립 정답이 아니라 같은 Codex의 영상 검토다.
3시드 전체에서도 추가8단어는 B119~120/133, C121~122/133이었다. 기존4단어 손모양은 B225~228/228, C225~226/228로 C가 항상 B보다 좋지는 않았다. 단계는 B71~74/84, C70~71/84이므로 넓은 단어 학습이 단계 판단까지 개선했다는 근거는 없다.
100단어 보조 출력은244/300(다른 시드239,251)이지만 이것은 새 피드백 신경망의 보조 과제 점수다. 기존 TCN 인식률이나 피드백 정확도를 뜻하지 않으며 TCN을 교체하지 않았다.

4. 속도·관측 부족·순서
C대표의 관절 준비 후 신경망 계산은 CPU4스레드 중앙값 약8.03ms였다. 촬영·관절 추출·DTW·브라우저를 합친 응답시간이 아니다.
정상속도/1.25배 길이/0.75배 길이에서 손모양226/228,227/228,226/228; 단계71/84,72/84,71/84. 실제로 다시 촬영한 빠른 손의 흐림을 검증한 것은 아니다.
손 관측을 전체 시간에서 숨기면 손모양228항목과 순서12항목은 모두 보류됐다. 단계 신경망 자체는 몸통만 보고도 출력을 내므로 그것을 동작 완수 증거로 쓰면 안 된다. 짧은 누락2프레임에서는 손모양226/228이었지만 확실하게 비교 가능한 항목은210정답·18보류였다. ‘앞뒤로 추정한 모양’과 직접 관측은 합치지 않는다.
주요 구간 역순 시험의12순서 쌍은 원시 출력상 모두 정상 순서가 아니었지만, 확실히 구분한 것은6쌍이고6쌍은 보류였다. 모든 역순 오류를 잡았다는 뜻이 아니다. 실제 사용자 오류/생략된 짧은 동작 전체의 교정 성능은 아직 미완이다.

5. 촬영 끝에 손이 다시 나오는 문제
공개12영상의 저장 특징 뒤에 대기15프레임/동작8프레임/대기15프레임을 덧붙인 진단을 했다. 중요한 기존 동작 구간의 연결은12개 모두 유지됐지만 끝부분 대응이 바뀌고 참고 프레임이 오래 유지될 수 있었다. 실제 마우스 동작 영상은 사용하지 않았다. 따라서 사용자가 본 문제를 그대로 재현했다고 말하지 않는다. 올바른 마지막 동작까지 잘라내지 않으려면 ‘동작이 끝난 근거’를 확인한 뒤 대기/부가 행동만 재생 정렬에서 분리해야 한다.

6. 남은 작업과 적용 상태
손모양 개선 후보는 저장했다. 단계·순서·중요 동작 누락을 구분하는 정답과 연결 출력부를 보강한 뒤 검증해야 한다. 지금 개선한 공통 특징과 기존 단계 판정부를 무작정 합치면, 신경망 내부 표현이 달라져 다시 잘못 판단할 수 있다. 다음은 손모양 이득을 유지하면서 단계/방향/짧은 사건을 별도로 감독하고, 정상·역순·누락·교차를 전체 자동 흐름에서 확인하는 것이다. 새100단어 정답을 전부 완성할 때까지 웹 개선을 막는 방식은 피한다.
웹에 반영한 것은 비교재생 코드뿐이다. C·T3F/현재 TCN/기존 피드백 가중치·DTW 알고리즘은 유지했다. 사용자·멘토 영상 접근/학습/튜닝/후보선택 없음. 새 모델 다운로드 없음. 새 HTML은 만들지 않았다.
구현·시각검토·별도 산술검산은 같은 Codex가 수행했다. 저장 결과와 해시3190항목을 별도 코드로 확인했고 외부 전문가의 독립 검증을 뜻하지 않는다. 처음 발생한 입력키/검산키/브라우저 제한 오류는 로그에 남겼으며 재학습·완료 모델 재평가 없이 해결했다.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
summary='공개100단어899FIT/300DEV를 재사용한 Conv+GRU2조건×3seed 학습·평가·3190저장산술/해시검산 완료. 추가8표현 보이는 손모양166FIT항목; 왼손펴기7→28항목/1→4표현. 대표42 기존4단어 손모양221/228→B225→공통100추가C226, 단계72/84→71/84, 정상순서10/12→10/12. 추가8표현의 사전RGB검토133DEV항목 B119→C122(미확정83별도),3seed B119~120/C121~122. 신규모델은 단계회귀와연결출력부미검증으로웹미적용. 100보조분류244/300은TCN인식률아님; CPU신경망8.03ms는전체응답아님. 구8812와달리신8814는두영상독립재생+반복seek중복이확인되어사용자영상한시계/참고seek대기/실제표시시점그리기로수정,로직8+공개실영상브라우저8통과. DTW/판정값불변. 공개꼬리합성12영상의주요연결변화0,끝부분영향존재;개인마우스재현아님. 개인자료접근0, C·T3F/현재TCN/웹피드백가중치유지. 같은Codex시각검토/별도검산이며전문가정답아님. 모든훈련평가프로세스종료. 근거 '+str(U/'RESULT_KO.txt')
next_action='이번6학습/6고정평가/꼬리진단완료재실행금지. 다음은새공통손특징의이득을보존하면서단계·방향·짧은사건/누락감독을분리하고의존출력부를같은입력표현으로연결검증하는작업. 기존단계정답84중신규손실/개선은STAGE_CHANGE_AUDIT,모든값은scores에보존. 정상/역순/짧은누락/교차/긴관측부족을전체자동흐름에서통과해야연구웹모델적용가능. 사후DEV로문턱/시드반복튜닝금지. 촬영종료후부가동작은검증된종료근거가있을때만재생범위분리검토하며마지막주요동작을정답으로채우거나임의제거금지. 현재8814재생코드만적용했고실제사용자카메라검증없음;서버/카메라를이번에자동기동하지않았다.'
write(U/'CURRENT_EXECUTION.json',dict(status=result['status'],pid=None,training_complete=True,evaluation_complete=True,verification_complete=True,report=str(U/'RESULT_KO.txt'),next=next_action))
sync(result['status'],summary,next_action)
p=O/'STATE.json';st=read(p);st['updated_at']=datetime.now(timezone(timedelta(hours=9))).isoformat();st[U.name].update(result=str(U/'RESULT.json'),verification=str(U/'FINAL_CHECK.json'),new_model_deployed=False,replay_code_changed=True);write(p,st)
print(json.dumps(dict(status=result['status'],stage_changes={k:stage_change[k] for k in ['recovered','regressed']})))
