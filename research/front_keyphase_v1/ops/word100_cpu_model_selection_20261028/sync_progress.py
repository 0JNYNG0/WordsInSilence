from pathlib import Path
import datetime,json,shutil
U=Path(__file__).resolve().parent
R=U.parents[3]
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
before=U/'before'/'progress_sync';before.mkdir(exist_ok=True)
paths=[R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',U.parent/'STATE.json']
for p in paths:
    if not (before/p.name).exists():shutil.copy2(p,before/p.name)
status='CPU_MODEL_COMPARISON_RUNNING'
summary='사용자가 MobileNet을 최선으로 단정하지 말고 실제 CPU 성능을 비교하도록 요청했다. 출처검토·평가절차를 별도 에이전트로 확인하여 기존 MobileNetV3-Small과 RepViT-M1.0, MobileNetV4-Hybrid-Medium을 선택했다. 공식 ImageNet 가중치에서 시작하며 전체 공식 분류 경로를 유지하고 마지막 단계만 미세조정한다. 공개 수동45수형 fit2489(ID01/02/04/05)·개발488(ID03/06/07),3seed×20epoch로9회 비교한다. 이전 최종382는 완전 제외, 사용자·멘토 자료 미접근, 기존C/T3F/운영32핀 보존. 입력2977파일해시·사람/영상 분리와 full/split 경로 동치 통과. 학습 전 metrics호출 인수순서와 출발가중치핀을 별도검토로 보완하고 최초코드·계약을 보존했다. 학습 실행 exec82227 PID12560; 실제진행은 STATUS와log확인, 중복실행금지. 이 실험은 CPU 손모양 모델 선택용 개발비교이며 한국수어 교정 정확도나 최종시험 결과가 아니다.'
nxt='9회 고정학습 완료 뒤 저장DEV점수만 별도검산하고, 학습실행이 종료된 상태에서 전체모델CPU 속도와120/300손입력 묶음을 실측한다. 사람별 균형macroF1 및1%p내 속도우선 규칙으로 후보를 선택하고 한계와함께 보고한다. 약3초는전체피드백의부드러운선호치이며RGB시간만으로보장하지않는다. Holistic 비교는 별도후속이며 이번RGB비교와혼동하지않는다. 이전최종382·사용자자료로선택하지않고운영교체도하지않는다.'
context=paths[0].read_text(encoding='utf-8-sig')
for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','2026-10-28 노트북 CPU 영상모델 3종·9회 비교 진행 중',summary),('## 9. 다음 단일 작업','2026-10-28 CPU 모델의 손모양 정확도·처리속도 실측',nxt)]:
    context=context.replace(marker,marker+'\n\n### '+title+'\n'+now+'\n'+body+'\n근거: research/front_keyphase_v1/ops/'+U.name+'/FROZEN_PROTOCOL.json, CURRENT_EXECUTION.json.\n',1)
paths[0].write_text(context,encoding='utf-8')
paths[1].write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n현재 목표: 노트북 CPU에서 정확도와 속도를 함께 비교하여 손모양 관측 모델을 선정한다.\n\n'+summary+'\n\n다음 한 작업: '+nxt+'\n\n작업단위: research/front_keyphase_v1/ops/'+U.name+'\n\n기존42회결과는동결보존. 현재이전최종test재사용0. 최신실행은CURRENT_EXECUTION/STATUS/RUNNING/TRAIN_RUN.log 확인.\n',encoding='utf-8')
s=json.loads(paths[2].read_text(encoding='utf-8-sig'));s['updated_date_kst']=now;s['updated_at']=now;s['current_stage']=status;s['next_single_action']=nxt
s[U.name]={'status':status,'at':now,'unit':U.name,'summary':summary,'next':nxt,'production_changed':False,'user_mentor_data_used':False,'previous_test382_used':False,'expected_runs':9,'execution_session':82227}
paths[2].write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
(U/'CURRENT_EXECUTION.json').write_text(json.dumps({'session_id':82227,'pid':12560,'command':'compare_cpu.py run','log':str(U/'logs/TRAIN_RUN.log'),'at':now,'no_duplicate_execution':True},indent=2),encoding='utf-8')
print('synced')
