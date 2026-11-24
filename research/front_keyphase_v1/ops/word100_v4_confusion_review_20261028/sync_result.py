"""Record the requested explanation/visual diagnosis without changing the model plan."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,shutil
U=Path(__file__).resolve().parent; R=U.parents[3]; S=U.parent/'word100_cpu_model_selection_20261028'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
s=read(U/'SUMMARY.json');assert read(U/'AUDIT.json')['status']=='PASS' and read(U/'UI_VERIFY.json')['status']=='PASS'
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
summary=f"사용자 요청으로 저장된 9회 예측만 이용해 45종 오차 행렬, 3모델·3실행 전환, 행별 정답률/예측 신뢰도 구분, 실제 공개 오답 사진을 시각화했다. 재학습·재추론 없음. V4 평균59.90%; 같은488장 중 3회 모두정답204/혼재167/모두오답117. 총1464예측 중877정답/587오답이며 서로다른1464장이 아니다. 주요 f→5 48건,3→5 34건,pincet→f 34건; 상위4혼동쌍24.0%. 원자료라벨5 재현율84.85%이나 해당예측정밀도22.76%로 단독교정근거 부적합 예시를 확인. 원인인과관계는 확정하지 않았다. V4는 RGB손사진→45수형점수/후보를 출력하는 개발보조후보이며 관절복원/시계열이해/TCN대체/교육정오판정 모델이 아님을 설명했다. 저장값 별도산술136항목·UI13항목 검산; 동일Codex 별도검산이며 독립사람정답검토가 아님. 운영32핀보존,사용자·멘토·이전최종382미접근."
next_action="오차 행렬·역할 설명을 사용자와 확인한 뒤, 고정 V4 보조후보의 새 공개 한국수어 근거 검증으로 이어간다. 기존9회 개발점수를 최종시험으로 재해석하거나 오답사진만으로 원인을 확정하지 않는다. 정상손/회전/교차/짧은변화/관측부족의 독립근거를 확보하고, 필요한 짧은시간창의 RGB보조가 정상관측을 해치지 않고 손모양 근거와 피드백을 개선하는지 비교한 후 전체지연을 측정한다. C/T3F유지,개인촬영학습·튜닝·선택금지; 운영교체/추가학습을 이번 설명요청만으로 시작하지 않는다."
paths=[R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',U.parent/'STATE.json']
(U/'before').mkdir(exist_ok=True)
for p in paths:
    if not (U/'before'/p.name).exists():shutil.copy2(p,U/'before'/p.name)
context=paths[0].read_text(encoding='utf-8-sig')
title='2026-10-28 V4 오차 행렬·역할 설명 완료'
assert title not in context
for marker,heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',title,summary),('## 9. 다음 단일 작업','2026-10-28 오차 진단 확인 후 선택적 RGB 보조 검증',next_action)]:
    context=context.replace(marker,marker+'\n\n### '+heading+'\n'+now+'\n'+body+'\n근거: research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.html, SUMMARY.json, AUDIT.json, UI_VERIFY.json.\n',1)
paths[0].write_text(context,encoding='utf-8')
paths[1].write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n현재목표: 수어단어인식과 신뢰할수있는 세부학습피드백. V4는 손모양 사진관측 보조후보.\n\n'+summary+'\n\n직전모델선정: fit2489(4명)/DEV488(3명),3모델×3seed. V4가V3대비9.15%p높음.45종학습이나개발30종만포함. CPU300손사진RGB p95 4.34초로전체피드백3초미보장. Holistic시험/도입없음.\n\n다음한작업: '+next_action+'\n\n작업단위: research/front_keyphase_v1/ops/'+U.name+'\n활성학습/추론없음. 이전동결결과재추론불필요. 사용자가현재요청한시각화와설명완료; 모델선택결과불변.\n',encoding='utf-8')
state=read(paths[2]);state.update(current_stage='V4_CONFUSION_REVIEW_COMPLETE_NOT_DEPLOYED',updated_at=now,updated_date_kst=now,next_single_action=next_action)
state[U.name]=dict(status='COMPLETE',at=now,summary=summary,next=next_action,result=str(U/'RESULT_KO.html'),model_selection_changed=False,new_training=False,new_inference=False,operating_changed=False,user_mentor_used=False,verification=['AUDIT.json','UI_VERIFY.json'])
paths[2].write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
(U/'RESULT_KO.txt').write_text(summary+'\n\n'+next_action+'\n\nV4입력: 손영역RGB사진1장. 처리: 전처리 후분류망. 출력:45종수형점수/후보순위. 현재는 독일수어손사진개발비교에서선정. 단어의미/관절좌표/동작순서/교육적오류교정은V4단독출력아님. C/T3F단어인식과DTW위치·움직임비교유지,검증후수형관측근거로연결하는방향.\n',encoding='utf-8')
(U/'SCOPE.json').write_text(json.dumps(dict(at=now,authorization='User requested confusion matrix visualization and plain explanation of V4 role and project goal',inputs='saved public fit/dev manifests and nine stored dev predictions, deterministic public image examples',outputs=['RESULT_KO.html','figures','SUMMARY.json','MATRICES.json'],new_training=False,new_inference=False,production_changed=False,performer='Codex',verification='separate saved-data arithmetic + browser UI + visual inspection; no new human labels'),ensure_ascii=False,indent=2),encoding='utf-8')
assert read(paths[2])[U.name]['status']=='COMPLETE'
print('Documentation synchronized; source/model plan preserved.')
