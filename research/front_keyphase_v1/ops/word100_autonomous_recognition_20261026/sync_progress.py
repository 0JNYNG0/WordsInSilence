from common import *
import shutil

def main():
 status='FIXED_ENSEMBLE_DEV_GAIN_FRESH_PUBLIC_INPUTS_RUNNING_NOT_DEPLOYED'
 fresh=U/'fresh199';at=now()
 current=dict(at=at,status=status,unit=str(U),prep_session=1215,prep_lock=str(fresh/'PREP_RUNNING.json'),prep_status=str(fresh/'STATUS.json'),records=199,implementation='Codex local; separate scalar verifier same implementer',next='Wait existing preparation, independently verify inputs, evaluate four frozen models once, aggregate fixed E3 and scalar verify; no fresh-score tuning.')
 write(U/'CURRENT_EXECUTION.json',current,new=False)
 description='''사용자가 인식률 개선을 위한 자율 검증·후속 비교를 승인했다. 사소한 단계 재승인 불필요. 기존 C·T3-F/운영 웹·DTW는 보존하며 사용자·멘토 촬영 접근·학습·튜닝·선택금지. 추가 배포는 별도다.
같은 T3-F 세 체크포인트(20261013/14/15)의 확률을 동일1/3로 합치는 E3 하나를 점수 확인 전에 동결했다. 기존 선택 T3F13 대비 DEV300 sensor296→299(+1.00%p), masked295→300(+1.67%p). 이미 노출된 REAL17/18 200자료는200→200 유지; 새사람 검증으로 세지 않는다. 별도scalar1000행 검산/보호32파일 해시 PASS. CPU4thread 순수 모델 median2.30→6.78ms, 검출·DTW·웹 시간 제외. 새 학습 없음.
신규 평가 준비 중 REAL16 영상4개가 기존 DEV REAL11과 SHA256까지 같음을 발견. 최초 freeze 오류·중복감사 보존. fresh 점수 전 REAL16 전체100을 격리; 원래 영상연결 미확정1 제외한 REAL14/15 199개로 수정하고 동결. 실제 사람 신원은 파일 ID로만 표현하며 얼굴로 동일인 판정하지 않는다. 소스/기록·코드 검색상 이전1199/노출200과 겹치지 않는 조건이지 독립 실사용자 검증 아님.
현재 fresh199/prepare_inputs.py exec1215에서2worker 실행 중. PREP_RUNNING/STATUS/EVENTS 확인하고 중복실행하지 않는다. 모델별 새점수 아직없음. 숫자 검산 뒤 네 모델(C13,T3F13/14/15)×2관측경로를 한번 평가하고 E3를 저장점수로 합친다. 실패 입력을 분모에서 제거하지 않는다. 첫오류보존, 규칙·모델·가중치를새점수로변경금지.
손목 과제는 현재 RGB의 팔·손 연결부를 대략적범위/불확실로 구분하는 평가기준을 정리한다. 기존10충돌은 미해결이고 영역근거를 정확한 손가락 정답으로 승격하지 않는다. E3는 손좌표·마스크를 바꾸지 않아 기존 보존계약을 그대로 유지한다. 관측 복원과 단어 인식 성과를 분리한다.'''
 nexttext='fresh199 입력 완료 후 verify_inputs.py → evaluate.py(4모델,1회) → 상위 aggregate_fresh.py → verify_fresh.py. 완료 검산/점수는 재실행하지 않는다. 새199점수는 보고만 하고 튜닝 금지. 결과·개선/회귀·시간비용·평가한계를 보고하고 STATE/§8/§9/인수인계 동기화. 기존C·T3-F·운영 보존, 사용자자료사용0.'
 context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';statepath=O/'STATE.json';handoff=ROOT/'NEW_CHAT_HANDOFF.md'
 before=U/'before';before.mkdir(exist_ok=True)
 for p in [context,statepath,handoff]:
  b=before/p.name
  if not b.exists():shutil.copy2(p,b)
 text=context.read_text('utf-8-sig')
 title='### 2026-10-26 자율 인식 개선 — 고정 E3 개발 개선·신규 공개 검증 진행'
 if title not in text:
  marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';text=text.replace(marker,marker+'\n\n'+title+'\n'+description+'\n',1)
  marker='## 9. 다음 단일 작업';text=text.replace(marker,marker+'\n\n### 2026-10-26 자율 개선 — 새199 공개 입력 및 고정 E3 검증\n'+nexttext+'\n',1)
  context.write_text(text,encoding='utf8')
 state=read(statepath);state['updated_date_kst']=at;state['current_stage']=status;state['next_single_action']=nexttext
 state[U.name]=dict(**current,authorization=str(U/'AUTHORIZATION.json'),frozen=str(U/'FROZEN.json'),development_scores=str(U/'SAVED_COMPARISON.json'),development_verifier=str(U/'SAVED_VERIFICATION.json'),duplicate_audit=str(U/'DUPLICATE_AUDIT.json'),fresh_context=str(fresh/'EVAL_CONTEXT.json'),training=0,production_changed=False)
 write(statepath,state,new=False)
 handoff.write_text('# WordsInSilence 최신 인수인계\n\n갱신 '+at+'\n현재 단위: '+str(U)+'\n\n'+description+'\n\n다음 단일 작업: '+nexttext+'\n\n원본W:/sources/과거 동결 결과 읽기전용. 구현/실행/별도수치검산 모두 Codex 로컬; 독립 전문가 검토 아님. 이전 위치·모양 단위 결과는 before/인수인계 및 원래 RESULT 참조.\n',encoding='utf8')
 print(json.dumps(current,ensure_ascii=False))

if __name__=='__main__':main()
