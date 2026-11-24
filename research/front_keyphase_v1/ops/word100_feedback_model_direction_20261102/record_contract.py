from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
R=Path(r'C:\MyUnityProjects\WordsInSilence');O=R/'research/front_keyphase_v1/ops';U=Path(__file__).resolve().parent
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
summary=('TCN은 단어 인식, 별도 피드백 AI는 해당 단어의 중요한 특징·전체 단계·순서·변이를 공개 다인 자료로 미리 배워 새 사용자 동작을 평가한다는 역할을 재확정했다. '
         '현재 SignRep은 동결 특징 추출 기반이며 작은 판정부는 제한된7특징 과제만 학습했다. 모델 크기 때문에 원천 불가능한 것으로 설명하지 않는다. '
         '동시에 현재 판정부 그대로 정상 자료만 추가하면 수행 평가가 완성된다고 주장하지 않는다. 목표 조건의 평가 학습 및 전체 순서 출력은 아직 구현되지 않았다.')
next_action=('GOAL_CONTRACT_KO.txt의 계약과 순서를 따른다. 기존4표현 공개 자료의 중요한 특징/단계 정답 지원을 먼저 표로 확인하고, '
             '목표 단어 조건의 전체 시계열 평가부를 명세·구현·학습·분리 평가한다. 기존 관측 부품/SignRep을 재사용하며 '
             '예상 정답으로 실제 관측을 바꾸지 않는다. DTW는 준비/설명 보조,9원영상 실시간 대조는 최종 판단의 필수 경로에서 제외한다. '
             '정상 오경고/핵심 차이·누락/관측 부족 오통과/속도를 평가하고 검증된 범위만 데모에 연결한다. '
             '사용자 자료 학습·튜닝·선택 금지, C/T3F·기존 데모 보존. 이번은 목표/설계 계약 작성이며 새 학습·추론0.')
record=dict(status='TARGET_CONDITIONED_FEEDBACK_CONTRACT_DEFINED_NOT_TRAINED',at=now,summary=summary,next_action=next_action,
            contract=str(U/'GOAL_CONTRACT_KO.txt'),contract_sha256=hashlib.sha256((U/'GOAL_CONTRACT_KO.txt').read_bytes()).hexdigest(),
            model_size_claim='Feasibility does not follow from size alone; task/supervision/temporal context differ.',new_training=0,new_inference=0,user_inputs_used=0)
with (U/'TARGET_CONTRACT_RECORD.json').open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig')
for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','TCN과 학습된 수행 평가 AI의 역할 확정',summary),('## 9. 다음 단일 작업','목표 단어 조건 판정부의 학습 정답과 전체 순서 명세',next_action)]:
    assert marker in t;t=t.replace(marker,marker+'\n\n### 2026-11-02 '+title+'\n'+body,1)
p.write_text(t,encoding='utf8')
p=O/'STATE.json';d=json.loads(p.read_text(encoding='utf-8-sig'));d['current_stage']=record['status'];d['next_single_action']=next_action
d[U.name]['target_contract']=record;p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_action+
    '\n\n계약: '+record['contract']+'\n기존8804는개별9참고비교연구데모이며새목표모델완성아님. PID31864/8802 PID26000은재확인필요. '
    '이번원본촬영접근/새추론/학습/다운로드/운영변경0. 기존공개평가재실행하지말고계약에없는무조건관측정밀화/모델검색반복하지않는다.\n',encoding='utf8')
with (R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:
    f.write('\n\n'+now+' TCN과 별도 수행 평가 AI 역할 계약\n'+summary+'\n'+next_action+'\n기술 선택보다 학습 과제와 정답·출력의 불일치를 바로잡는 단계. 새 성능 결과는 아직 없음.\n근거: '+record['contract']+'\n')
assert json.loads((O/'STATE.json').read_text(encoding='utf8'))[U.name]['target_contract']['new_training']==0
print(record['status'])
