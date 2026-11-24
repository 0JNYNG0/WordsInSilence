from pathlib import Path
import json,hashlib,datetime,html
from collections import Counter
U=Path(__file__).resolve().parent;ROOT=U.parents[3];O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
 s=read(U/'SUMMARY.json');v=read(U/'review/SAVED_VERIFICATION.json');ui=read(U/'UI_VERIFICATION.json');f=read(U/'FROZEN.json')
 assert v['status'].startswith('PASS') and ui['status']=='PASS'
 assert sha(U/'index.html')==ui['html_sha256'] and sha(U/'REPORT_DATA.json')==ui['data_sha256']
 assert all(sha(p)==h for p,h in f['pins'].items()) and all(sha(p)==h for p,h in f['protected'].items())
 for row in read(U/'RESULT_MANIFEST.json')['results']:assert sha(row['file'])==row['sha256']
 d=read(U/'REPORT_DATA.json');partial=[];diff=[]
 for c in d['cases']:
  if c['role']!='development':continue
  for st in c['stages']:
   keys={g for card in st['source_cards'] for g in card['applicable_numeric_groups']}
   for g in sorted(keys):
    x=st['groups'][g];m=x['saved_metric'];fb=x['feedback']
    row=dict(case=c['id'],stage=st['id'],group=g,original_range_state=fb['original_range_state'],
       expected=m['expected'],possible=m['reference_possible'],mean=m['mean_difference'],
       query_missing=m['path_missing_cells'],alignment_flags=st['alignment']['flags'])
    if fb['status']=='comparison_evidence_insufficient':partial.append(row)
    if fb['status']=='observed_difference_to_review':diff.append(row)
 analysis=dict(development_source_linked_groups=84,partial_observation=partial,
   partial_previous_state=dict(Counter(x['original_range_state'] for x in partial)),
   numerical_difference_candidates=diff,not_error_examples=True,
   note='Counts are correlated measurement opportunities, not videos, errors, or accuracy.',
   preservation='All existing values/masks/path retained; confidence wording refined only.')
 write(U/'ANALYSIS.json',analysis)
 next_plan=dict(goal='Named observable features and their temporal order, linked to descriptions; important versus acceptable differences remains unproven.',
  steps=[
   dict(step=1,task='Inventory existing public frame-level feature labels and observable support. Distinguish extended/closed fingers, hand height relation, and direction/hold changes from exact contact/hidden fingers.',gate='Positive target needs visible evidence and a time interval; source prose/context link alone is not a label. Existing source-action flags remain false.'),
   dict(step=2,task='Prepare a small person-separated feature-learning set from already authorized public material, plus short change/occlusion/normal variation checks.',gate='Deduplicate original clips; do not promote consumed four-expression DEV to fresh evaluation; preserve unknown intervals, original sequence, and normal rest/crossing.'),
   dict(step=3,task='After data readiness, compare a small temporal feature head on existing frozen SignRep representations/MediaPipe observations against the existing geometry baseline. No new backbone chosen by assumption.',gate='Predeclare targets, speed measurement, per-person split and missing-data rules. No user footage, no label-driven coordinate filling, no production replacement.'),
   dict(step=4,task='Attach only verified feature outputs to the current source/evidence cards and compare complete feedback on unused public examples.',gate='Assess feature errors, timing, normal false corrections, missing short motions, abstention, and end-to-end latency separately. Do not claim educational correctness without support.')],
  no_new_download=True,production_models_preserved=True,user_data_excluded=True,
  current_learning_readiness=False,reason='The ten connected source items are partial context, not dense semantic targets; exact named shapes/contact/required repetition are not certified.')
 write(U/'NEXT_STEP_PLAN.json',next_plan)
 text=f'''설명과 영상의 비교 근거 연결 — 완료 ({now})

이번 목표
색으로 동작 구간을 나누는 작업에서 한 걸음 나아가, 사전 설명의 어느 항목을 영상의 무엇과 비교했는지 실제로 보여주는 연결부를 만들었다.
아직 설명문을 이해해 수어의 정오를 판단하는 학습 모델을 완성한 것은 아니다.

실행한 순서
1. 기존 사전 설명11항목과 원본 검토 시점을 대조했다.10항목을 직접 검토한 참고4영상의 순간에 연결하고, 다른 표현형1개는 합치지 않았다. 설명 범위와 숫자 비교 구간을 구별했다.
2. 기존 공개16영상(4표현×4기록 수행자),36쌍117구간의 전체DTW·수치·마스크를 그대로 재사용했다. 설명 관련 손가락 각도·손 위치·팔 관계와 시간별 그래프를 연결했다. 어떤 손이 보이지 않아도 참고 범위를 줄이지 않는다.
3. 전체 항목과 각각의 세부 값을 분리했다. 한 각도가 불확실해도 이웃 각도의 좌표와 비교 곡선은 보존한다. 일부 팔채널만 있는 경우 팔 전체가 관측된 것처럼 말하지 않는다.
4. 출처검토, 합성코드검사8,109건, 별도 저장검산{v['checks']:,}건, Chrome36쌍117구간·영상재생·그래프·좁은화면 검증을 통과했다. root 구현, guide 출처/계획검토, B 합성코드검증, A 저장산술검산. 모두 같은 Codex 계열 검증이며 독립 사람의 수어 정답 검증이 아니다.

개발 비교 결과
설명과 관련된84개 측정 기회: 참고 소표본 범위 안42, 더 큰 수치 차이10, 비교 근거 부족26, 참고 범위 없음6.
84는 영상 수나 정답률의 분모가 아니다. 같은 영상·구간의 여러 항목이 포함된다.
기존 전체 개발507항목은 각각212/67/203/25. 별도 정렬 확인에서25/39구간에 국소 비교 이견이 남아 있으며 이 경고도 따로 표시한다.

실제로 보완된 점
26개는 참고 자체의 일부 값이 빠져 있었다. 예를 들어 조상의 오른손 각도는 한 구간에56개가 필요하지만42개만 있었다. 남은42개가 비슷하다고 손 전체가 잘 보였다고 하지 않는다. 보이는42개는 그대로 비교하고, 빠진14개를 명시한다.
이26개의 과거 범위 상태: {json.dumps(analysis['partial_previous_state'],ensure_ascii=False)}.
‘잘하다’에서는 오른손과 왼팔 관계를 시간별로 볼 수 있다. 이 값은 손목과 반대 전완의 화면상 관계이지 손바닥 접촉·압력을 증명하지 않는다. 오래된 설계의4차원 설명을 실제2차원 구현으로 오인하지 않게 문구를 맞췄다.
정상 손내림·교차를 새 규칙으로 삭제하지 않았으며, 동일동작/속도변형/교차/짧은변화/관측누락 합성검사에서 원연결과 차이·누락이 보존됐다. 실제 수어 허용 경계가 입증된 것은 아니다.

남은 문제
10개의 큰 차이는 틀린 수어10개가 아니다. 정상적인 개인차, 검출 오차, 정렬 차이가 섞일 수 있다.42개 범위 안도 정답42개가 아니다.
주먹·엄지 펼침 같은 이름, 정확한 피부 접촉, 꼭 필요한 반복횟수, 허용과 오류의 경계는 아직 자동 판정하지 못한다.
이번에는 새로운 학습·추론·단어 인식률 시험을 하지 않았다. 기존 성능 향상을 새로 주장하지 않는다. 처리1.39초는 저장자료 보고 계층 생성시간이며 사용자의 촬영 후 전체 응답시간이 아니다.
같은4표현의 이미 소모한 개발자료이며 다른 모든 단어에 검증된 성능을 보장하지 않는다. 공통 비교 함수에는 단어별 수치 예외를 넣지 않았다.

다음 핵심 작업
구간 색깔을 더 잘게 나누기보다, 직접 보이는 손 모양과 양손 관계·움직임의 시간별 정답표본을 먼저 준비한다.
기존 설명 연결을 정답라벨로 바꾸지 않고, 직접 확인한 특징·불확실한 특징을 분리한다. 사람별 분리를 지키고, 자료가 갖춰지면 기존 SignRep 동결특징·관측값에 작은 시간별 판정부를 붙이는 비교로 이어간다.
새 평가에는 미사용 공개 표현/수행자가 필요하다. 특징정확도·잘못된 교정·짧은동작 누락·보류율·실제 전체처리시간을 따로 확인한다. 구체 gate는 NEXT_STEP_PLAN.json.

확인 방법
http://127.0.0.1:8797/ 에서 표현→구간→설명→세부각도/팔관계를 선택한다. 슬라이더와 그래프를 누르면 두 영상을 저장된 시간 연결로 이동한다.
파랑=참고값, 주황=연결된 비교값들의 범위, 보라점=설명과 연결한 참고순간, 빨간표시=관측부족. 선이 겹친다고 정답, 떨어진다고 오답으로 해석하지 않는다.

보존
기존 C·T3-F·운영웹·DTW 보호32파일 및 입력16파일 해시 유지. 사용자·멘토 촬영 사용0. 원본·과거동결결과 수정0. 새 외부모델·다운로드·외부전송·Git0.
'''
 (U/'RESULT_KO.txt').write_text(text,encoding='utf8')
 (U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>설명과 관측 연결 결과</title><style>body{background:#f4f7f9;color:#172b3a;font:17px/1.8 system-ui;max-width:1000px;margin:auto;padding:24px}pre{white-space:pre-wrap;font:inherit}a{color:#075c91}img{max-width:100%}</style><h1>설명과 영상의 비교 근거 연결</h1><p><a href="http://127.0.0.1:8797/">두 영상과 그래프가 있는 결과 화면 열기</a></p><pre>'+html.escape(text)+'</pre><img alt="영상 비교 화면" src="VIEWER.png"><img alt="시간별 관측 그래프" src="GRAPH.png"></html>',encoding='utf8')
 server=dict(read(U/'SERVER_STATUS.json'),exec_session_id=77202,read_only=True,inference=False,camera=False)
 completion=dict(status='COMPLETE_DESCRIPTION_EVIDENCE_LAYER',at=now,active=None,
    report_url='http://127.0.0.1:8797/',server=server,summary=s,
    implementation='root',verification='guide/B/A separate Codex checks, not independent human ground truth',
    new_training=0,production_adopted=False,recognition_accuracy_improved=False,
    pins={n:sha(U/n) for n in ['FROZEN.json','RESULT_MANIFEST.json','SUMMARY.json','ANALYSIS.json','REPORT_DATA.json','index.html','UI_VERIFICATION.json','RESULT_KO.txt','RESULT_KO.html','NEXT_STEP_PLAN.json','review/PLAN_REVIEW.json','review/CODE_REVIEW.json','review/SAVED_VERIFICATION.json']})
 write(U/'COMPLETION.json',completion)
 (U/'CURRENT_EXECUTION.json').write_text(json.dumps(dict(active=None,status=completion['status'],server=server),ensure_ascii=False,indent=2),encoding='utf8')
 status='''### 2026-11-01 설명과 시간별 관측 연결 완료

- `word100_description_evidence_feedback_20261101`: 기존 공개4표현/4수행자ID/16영상,36쌍117구간 globalDTW·수치·마스크 보존. 사전11항목 중10항목을 FIT직접4영상의 앵커 문맥에 연결(다른표현형1개 제외); 전체프레임 의미정답 전파 없음.
- DEV 설명관련84측정기회: 작은참고범위42/수치차이10/근거부족26/범위없음6. 정확도/오답개수 아님.26개는 reference 일부채널이 없는 경우; 관측값은 보존하면서 전체손·팔 확인으로 오인하지 않게 구분. 원DEV507수치불변,정렬이견25/39 별도 유지.
- 설명·동영상2개·세부각도/팔관계·전체query연결·누락/곡선 UI8797. 계획/합성8109/저장검산/Chrome36쌍117구간 PASS. 기존32보호/16입력 SHA보존. 새학습·추론·정오판정·운영교체·개인자료사용0.
- root구현/guide출처검토/B합성/A저장검산. 같은Codex검증이며 독립사람수어정답 아님. 기존forearm은 손목2D2성분, 접촉·손바닥4D가 아니라는 실제 구현 한계 명시.
'''
 nxt='''### 2026-11-01 다음 단일 작업 — 이름 붙은 특징의 학습 가능 자료 준비

`word100_description_evidence_feedback_20261101` COMPLETION/RESULT_KO/NEXT_STEP_PLAN 기준. 완료36쌍117구간 재실행/새정렬금지. 이제 구간색 세분화 반복보다 직접 보이는 손모양·양손관계·이동/유지의 시간별 표적을 기존공개자료에서 점검하고 사람별로 분리한 작은 학습/평가묶음을 준비한다. 설명anchor는 문맥이지 프레임정답/필수성라벨이 아님. 미관측은 보존하고 정답으로 채우지 않는다. 충분한 표적/미사용평가가 확보된 후 기존SignRep동결특징·MediaPipe와 작은시간판정부를 비교; 새backbone임의선정/다운로드/DEV임계값튜닝/개인자료사용/운영교체금지. 기존C·T3F보존. 특징정확도/잘못된교정/짧은동작누락/보류/전체처리시간 별도평가. 현재수치차이는 수어정오가 아님.
'''
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';context=p.read_text(encoding='utf8');assert U.name not in context
 for prefix,block in [('## 8.',status),('## 9.',nxt)]:
  start=context.index(prefix);end=context.index('\n',start);context=context[:end+1]+'\n'+block+'\n'+context[end+1:]
 p.write_text(context,encoding='utf8')
 st=O/'STATE.json';state=read(st);state['updated_date_kst']=now;state['current_stage']=completion['status'];state['active']=None;state[U.name]=completion;state['next_single_action']='Audit existing public time-local named feature targets, prepare person-separated feature learning/evaluation readiness; no source-anchor-to-ground-truth promotion. Reuse completed evidence layer and preserve C/T3F/production/private-data exclusion.'
 st.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n목표: 한국수어 설명과 여러 공개 참고로 손·모양·몸관계·움직임·순서를 파악하고 중요한 차이/허용변이/관측부족을 구분한다.\n\n'+status+'\n'+nxt+'\n계산 active=null.8797은 공개16영상 읽기전용 보고서버 PID14256/exec77202. 이전8796이하 보존. 현재 JSON·로그·세션 먼저 확인해 중복실행하지 않는다. Python word100_signrep_feasibility_20261029/runtime/Scripts/python.exe; PYTHONIOENCODING=utf-8. 로컬HTTP/Chrome require_escalated. 현재 source-action/handshape 의미정답 flags는 false이며 새로운 이름특징 모델은 학습하지 않았다. 완료 보호/출처/수치 검산을 새 성능으로 과장하지 않는다.\n',encoding='utf8')
 # Verify authoritative summaries after writes, without touching historical artifacts.
 assert U.name in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
 assert read(st)['current_stage']==completion['status']
 assert all(sha(p)==h for p,h in f['protected'].items())
 print(completion['status'],analysis['partial_previous_state'])
if __name__=='__main__':main()
