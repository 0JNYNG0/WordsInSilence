from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
import json,hashlib,html,urllib.request
from evaluate import U,O,PREV,read,sha,METHODS
ROOT=O.parents[2]
NAMES=dict(global_dtw='기존 전체 DTW',mean_posture='평균 자세 검색',combined_context='앞뒤 통합 검색',core_first='핵심 우선 검색')
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 f=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in f['pins'].items())
 checks={n:read(U/'review'/n) for n in ['EVALUATOR_REVIEW.json','SAVED_AUDIT_VERIFICATION.json','LABEL_CONTRACT_CHECK.json','FINAL_CLAIM_REVIEW.json']};assert all(x['status'].startswith('PASS') for x in checks.values())
 ui=read(U/'UI_VERIFICATION.json');assert ui['status']=='PASS' and ui['html_sha256']==sha(U/'index.html') and ui['data_sha256']==sha(U/'REPORT_DATA.json')
 protected=read(O/'word100_semantic_episode_pilot_20261031/FROZEN.json')['protected'];assert all(sha(p)==h for p,h in protected.items())
 assert all(sha(PREV/p)==h for p,h in read(PREV/'COMPLETION.json')['pins'].items())
 health=json.load(urllib.request.urlopen('http://127.0.0.1:8795/health',timeout=4));assert health['read_only'] and not health['inference'] and health['unit']==U.name
 audit=read(U/'ASSESSMENT.json');s=audit['summary'];now=datetime.now(timezone(timedelta(hours=9))).isoformat();lines=[
 '다른 사람 사이 같은 동작 단계가 맞춰지는지 확인한 결과',
 '결과 화면: http://127.0.0.1:8795/  |  이전8794 화면과 모든 모델은 보존',
 '이번 목표: 같은 사람 영상을 바꿔 만든 시험에서 나아가, 서로 다른 공개 수행자의 원본 영상끼리 같은 물리 단계가 연결되는지 확인했다. 새 모델을 고르거나 정렬을 다시 실행하지 않고 이미 저장한 네 조건을 같은 영상 근거에 대조했다.',
 '자료: 공개 4표현·4수행자 ID·16영상. 참고3명×4표현=12영상, 개발1명×4표현=4영상. 참고끼리24쌍/78구간, 개발-참고12쌍/39구간, 전체36쌍/117구간이다. 이미 개발에 사용한 자료라 새로운 미사용 평가 점수라고 부르지 않는다.',
 '먼저 예측 시점을 열지 않고 RGB를 확인했다. 참고12의 핵심과 경계552프레임을 연속 검토하고, 개발4의624프레임을 모두 시트로 확인했다. root도 8영상의12시트를 교차확인했다. 손을 옮기는 과정과 멈추는 자세를 분리하고, 같은 종류의 반복과 몇 번째 반복인지도 따로 기록했다. 가림·경계는 보류했다. 정밀 손가락 3D·접촉·수어 정오의 독립 사람 정답은 아니다.',
 '비교 방법: 참고의 원래 핵심 범위는 바꾸지 않았다. 그 범위의 중간 프레임에 실제 연결된 비교 영상 시점을 평가했다. 경로가 없는 평균 자세 검색만 원래 검색 중심을 사용한다. 대표시점 하나만으로 전체 동작이 맞았다고 하지 않도록, 고정한 참고 핵심 범위 안의 저장 경로를 빠짐없이 확인하고 참고 프레임마다 같은 무게로 집계했다. 개발 비교의 참고 프레임 기회440개가 분모이며, 전체 영상 처음~끝의 모든 프레임을 평가한 비율은 아니다. 미확인도 분모에서 빼지 않았다.',
 '개발12쌍/39개 참고 범위 결과(단어 인식률·교육적 정오 정확도 아님):']
 for k in METHODS:
  z=s['dev'][k];c=z['states'];p=z['path'];same=c['same_kind_core'];diff=c['different_kind_core'];other=z['total']-same-diff;ratio=100*p['fractions'].get('same_kind_core',0) if p else None
  lines.append(f"{NAMES[k]}: 같은 단계 대표시점 {same}/{z['total']}({z['same_kind_percent']:.1f}%), 다른 단계 {diff}, 경계·미확인·기타 {other}. 고정 핵심 범위 전체 경로의 같은 단계 비율 "+(f'{ratio:.1f}%' if ratio is not None else '해당 없음')+'.')
 lines += ['이번 판단: 핵심 우선 검색을 기존 전체 DTW 대신 채택하지 않는다. 대표시점은 둘 다31/39개지만 고정 핵심 범위 전체 경로의 같은 단계 비율은 기존67.2%, 핵심 우선64.1%였다. 핵심 우선은 앞뒤 통합 검색61.7%보다는 높지만 기존 전체 DTW보다3.2%p 낮다. 이전 같은 영상 변형시험에서의 개선이 다른 사람 영상에서도 더 좋은 방법이라는 보장은 아니었다. 공통 알고리즘을 비교했으며 단어별 예외를 추가하지 않았다. 이번 검증은4표현 범위로, 모든 수어에서 확인됐다고 말할 수 없다.',
 '실제 발견1 — 조상: 참고 P063의 두 번째 왼손 상승(s05)에 기존 DTW는 P066의96프레임(두 번째 상승), 핵심 우선은73프레임(첫 번째 상승)을 연결했다. 두 순간은 손 모양과 상승 종류가 비슷해 종류 점수만 보면 통과하지만 회차가 다르다. 이 회차 차이는 정렬 문제이며 필요한 반복 횟수의 수어 규칙이나 사람의 수어 오류를 확정한 것은 아니다.',
 '실제 발견2 — 순종: 참고 P034의77~83프레임은 닫힌 오른손을 어깨 방향으로 옮기는 과정이다. 핵심 우선은 이를 P036의124~127프레임, 이미 멈춘 자세에 연결했다. 기존 DTW는107~113프레임의 이동 과정을 연결했다. 대표시점에서는 둘 다 마지막 유지 자세로 맞아 보이므로, 대표 한 장만 보면 놓칠 문제다.',
 '남은 문제: 기존 DTW도 전체 경로에서 다른 물리 단계와 연결한 비율이1.8%, 핵심 우선은4.0%였다. 두 방법 모두 참고 자체의 경계·미확인18.2%를 분모에 남겼다. 나머지 비일치 전부가 오류는 아니다. 같은 종류 대표시점 중 기대된 관측이 부족한 건 기존9개/핵심 우선4개였지만, 이는 다른 시간을 골랐을 때의 비교 지원 차이이며 손 검출 자체가 좋아졌다는 뜻이 아니다.',
 '검증 결과: 별도 Codex의 저장값 검산265,895항목에서 오류0. 사전 합성 코드64검사와 주석 계약6,172검사를 통과했다. 실제 Chrome에서36쌍×4조건과117개구간의 영상시간·경로 이동·재생·가독성을 확인했다. 같은 Codex의 반복 검토가 독립 사람의 한국수어 정답을 대신하지 않는다.',
 '결과 해석: 위 수치는 실제 수행자 사이의 물리 단계 대응에 대한 이번 주석 감사다. 정상적인 수어 영상을 서로 다른 단계끼리 비교해 놓고 손 모양이 다르다고 교정하면 안 된다. 같은 단계라도 원래 손 관측이 부족한 항목은 그대로 보류해야 한다. 원본 비교 점수, 임계값, 검색 조건은 이번 결과를 보고 바꾸지 않았다.',
 '중요한 제한: reference 구간의 모양·몸 관계·이동 종류를 확인했지만 올바른 수어의 허용 범위를 새로 학습한 것은 아니다. 동일한 상승 종류의 다른 반복 회차는 종류 일치로 세되 회차를 별도로 보여준다. 경계 판단은 같은 Codex가 만든 물리 초안이라 사람 전문가의 수어 정오 정답과 구분한다. 정밀 좌우 손가락·접촉·영상 밖 손은 계속 별도 관측 문제다.',
 '기존 기능 보존: C·T3-F/운영웹/DTW/과거 동결결과와 보호32파일 보존. 개인·멘토 자료0, 새 학습0, 새 정렬 검색0, 모델 추론0, 다운로드0, 외부전송0, Git0. 이번 감사 계산은 실제 게임의 전체 처리시간을 새로 측정한 시험이 아니다.',
 '다음 한 작업: 기존 전체 DTW를 기준으로 유지한다. 먼저 이 경로에 남은 다른 단계 연결을 전부 분류하고, 전체 순서를 유지한 채 이동→유지와 반복 회차를 함께 검증하는 공통 후보를 만든다. 정상적인 속도·반복 횟수 변이는 틀렸다고 강제하지 않고 복수 대응/보류로 남긴다. 조건을 실행 전에 고정하고 기존 정상 대응·손내림·교차·미관측·짧은 단계가 악화되는지 함께 본다. 현재4표현은 개발용으로 명시하고, 다음 일반화 주장은 별도의 미사용 사람·표현으로 검증한다. 교육적 교정이나 새 모델 학습으로 성급히 넘어가지 않는다.']
 report='\n\n'.join(lines);(U/'RESULT_KO.txt').write_text(report,encoding='utf8');(U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>수행자 사이 단계 대응 감사</title><style>body{background:#fff;color:#172d45;font:18px/1.8 system-ui;max-width:1080px;margin:30px}pre{white-space:pre-wrap}a{color:#175b90}</style><h1>다른 사람 사이 단계 대응 결과</h1><p><a href="http://127.0.0.1:8795/">영상·비교 화면 열기</a></p><pre>'+html.escape(report)+'</pre>',encoding='utf8')
 save(U/'review/PROTECTED_CHECK.json',dict(status='PASS',protected_files=len(protected),previous_completion_pins_unchanged=True))
 health['exec_session_id']=50928
 complete=dict(status='COMPLETE_CROSSPERSON_PHYSICAL_STAGE_AUDIT_NOT_EDUCATIONAL_TRUTH',at=now,active=None,summary=s,scope='posthoc development physical correspondence audit only',decision='KEEP_EXISTING_GLOBAL_DTW; core-first did not improve development whole-path physical correspondence',report_url='http://127.0.0.1:8795/',server=health,new_training=0,new_search=0,user_mentor_used=False,production_adopted=False,verification={n:x['status'] for n,x in checks.items()},pins={n:sha(U/n) for n in ['FROZEN.json','PHYSICAL_CORRESPONDENCE.json','ASSESSMENT.json','REPORT_DATA.json','CAUSE_REVIEW.json','review/CAUSE_RGB_VIEW.json','RESULT_KO.txt','RESULT_KO.html','index.html','UI_VERIFICATION.json','review/SAVED_AUDIT_VERIFICATION.json','review/PROTECTED_CHECK.json']})
 save(U/'COMPLETION.json',complete);save(U/'CURRENT_EXECUTION.json',dict(status=complete['status'],active=None,server=health))
 state=read(O/'STATE.json');state.update(current_stage='CROSSPERSON_PHYSICAL_STAGE_AUDIT_COMPLETE',active=None,updated_date_kst=now,next_single_action='Diagnose verified cross-person stage mismatches with whole-sequence order, movement/hold and repetition evidence; preserve missingness and all denominators; freeze any candidate before new execution; no word-specific exceptions or DEV threshold tuning.');state[U.name]=complete;save(O/'STATE.json',state)
 counts=' / '.join(f"{NAMES[k]} {s['dev'][k]['states']['same_kind_core']}/39" for k in METHODS)
 eight=f'''### 2026-11-01 다른 수행자 물리 단계 대응 감사 완료

- `{U.name}` 공개4표현/4수행자ID/16영상. RGB 참고핵심·경계552프레임+개발624프레임 확인 후 물리종류/이동·유지/회차/경계 구분, 저장36쌍117기회 감사. 새 검색·모델·학습0.
- 개발39 대표시점의 같은 물리단계: {counts}. 고정 핵심 범위 경로(DEV440 참고프레임 기회)는 reference 프레임 등가중치로 별도 집계; 전체영상 모든프레임 아님. 원본/초안/이전점수 보존, 새로운 미사용 평가·교육정오 정확도 아님.
- 기존 전체DTW 유지: 전체경로 same-kind67.2% > 핵심우선64.1%, 다른단계1.8% <4.0%. 조상P066/P063 s05의두번째상승을핵심우선이첫번째73프레임으로연결(기존96). 순종P036/P034 s04의이동을핵심우선이유지로연결. 정상수어를틀렸다고판정한것이아닌정렬문제. CAUSE_REVIEW와영상교차근거보존.
- 사전 합성 코드 검토·저장 산술·117분모·보호해시·실제 Chrome 영상 검증 완료. 읽기전용8795. C/T3F/운영/개인자료 보존.

'''
 nine=f'''### 2026-11-01 다음 단일 작업 — 다른 단계 연결의 원인과 전체 순서

`{U.name}`의 ASSESSMENT/PHYSICAL_CORRESPONDENCE/CAUSE_REVIEW/RESULT_KO와8795 화면 기준. 기존전체DTW를유지하며잔여다른단계연결전수분류→전체흐름순서와이동·유지/반복회차를함께지키는공통후보조건동결. 정상반복변이는복수대응/보류로남김. 모든정상·짧은동작·미관측분모유지,단어특례/DEV임계값튜닝/언어정오강제금지. 다음일반화주장은별도미사용사람·표현검증필요. 완료36쌍/117감사와이전162결과중복실행금지. 사용자·멘토자료/기존C·T3F/운영보존.

'''
 cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=cp.read_text(encoding='utf-8-sig')
 for h,v in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',eight),('## 9. 다음 단일 작업',nine)]:
  assert h in text
  if v.splitlines()[0] not in text:text=text.replace(h,h+'\n\n'+v,1)
 cp.write_text(text,encoding='utf8');(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n목표: 한국수어 설명과 여러 공개 참고로 어떤 손·모양·몸 관계·움직임·순서인지 파악하고 중요한 차이와 관측 부족을 구분한다.\n\n'+eight+nine+'\n계산 active=null. 8795는 공개16영상 읽기전용 보고 서버. health/CURRENT_EXECUTION 확인 후 중복실행금지. 기존8794/8793/8792/8791 보존. Python word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. 로컬 HTTP/Chrome은 require_escalated.\n담당 root 구현/통합/영상교차/감사/화면/문서, A DEV RGB 및 저장검산, B FIT RGB, guide 계획/코드 합성검토. 같은 Codex 검토이며 독립 사람의 언어 정답이 아니다.\n',encoding='utf8');print(complete['status']);print(counts)
if __name__=='__main__':main()
