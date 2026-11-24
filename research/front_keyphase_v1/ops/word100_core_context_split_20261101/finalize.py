from datetime import datetime,timezone,timedelta
import html,urllib.request
from run import U,OLD,REL,read,sha
import json
ROOT=U.parents[3];OPS=U.parent
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 frozen=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in frozen['pins'].items())
 reviews={n:read(U/'review'/n) for n in ['PLAN_REVIEW.json','SYNTHETIC_CHECK.json','SAVED_VERIFICATION.json','CORRESPONDENCE_VERIFICATION.json']}
 assert all(r['status'].startswith('PASS') for r in reviews.values())
 ui=read(U/'UI_VERIFICATION.json');assert ui['status']=='PASS' and ui['html_sha256']==sha(U/'index.html') and ui['data_sha256']==sha(U/'REPORT_DATA.json')
 protected=read(OPS/'word100_semantic_episode_pilot_20261031/FROZEN.json')['protected'];assert all(sha(p)==h for p,h in protected.items())
 for oldunit in [OLD,REL]:
  assert all(sha(oldunit/n)==h for n,h in read(oldunit/'COMPLETION.json')['pins'].items())
 save(U/'review/PROTECTED_CHECK.json',dict(status='PASS',production_pins=len(protected),previous_units=2))
 health=json.load(urllib.request.urlopen('http://127.0.0.1:8794/health',timeout=4));assert health['read_only'] and not health['inference'] and health['unit']==U.name
 s=read(U/'SUMMARY.json');c=s['controls'];now=datetime.now(timezone(timedelta(hours=9))).isoformat()
 lines=['핵심 동작과 앞뒤 연결 분리 — 네 단계 작업 결과','화면: http://127.0.0.1:8794/ (이전8793 보존)',
 '목표는 특정 단어의 답을 외우는 예외를 만드는 것이 아니라, 어느 표현에서도 같은 방식으로 핵심 움직임과 앞뒤 연결을 비교하는 것이다. 단어 이름·사람 ID·설명 문구는 숫자 검색에 사용하지 않는다. 다만 검증은 공개4표현16영상으로 제한되며, 모든 수어의 효과를 검증한 것은 아니다. 각 표현의 참고 구간과 설명 자료는 별도로 필요하고 자동 전 단어 단계 분할은 아직 미완성이다.',
 '1. 핵심 동작과 앞뒤 문맥을 분리했다. 핵심 전체의 최소비용 창을 먼저 찾고, 앞뒤0.20초는 고정된 핵심 위치 바로 앞뒤에서 별도로 계산한다. core의 유일 최선 위치는 주변 흐름 때문에 바꾸지 않는다. 숫자상 동점인 핵심 후보들만 문맥을 참고할 수 있고, 이때도 모든 후보의 앞뒤 지원과 유일한 문맥 최솟값이 필요하다.',
 '2. 두 비교가 다른 위치를 고르면 두 위치와 충돌을 남겼다. 핵심 지원·앞뒤 지원·반복 회차 모호성·관측 부족을 별도로 표시한다. 경로를 만들 수 없는 경우와 손 관측이 없는 경우도 구분한다. 원본 프레임·좌우·마스크·기존 전체 DTW 경로를 보존했다.',
 '3. 공개 자료에서 새 후보 하나를 동결한 뒤 162회 비교했다. 이전 평균검색/앞뒤통합검색 결과는 재계산하지 않고 보존했다. 아래 수치는 원본 대응을 아는 합성 통제의 중심시점 적중이며 수어 단어 인식률이나 교육적 정오 정확도가 아니다.']
 for name,v in c.items():
  lines.append(f"{name}: {v['cases']}영상/{v['stages']}구간, 평균검색 {v['mean_hits']} → 앞뒤통합 {v['context_hits']} → 핵심원후보 {v['raw_core_hits']} → 분리후 {v['selected_hits']}; 지원충분·모호하지않은적중 {v['supported_hits']}; 앞뒤까지충분한적중 {v['context_complete_hits']}; 문맥선택에따른상실 {v['raw_to_selected_losses']}, 위치충돌 {v['conflicts']}; 상태 {v['states']}.")
 lines += ['검증 기준 결과: '+str(s['gates']),
 '4. 검산된 근거를 별도 연구 피드백 화면에 연결했다. “핵심 대응 후보”, “앞뒤 관측 부족”, “두 비교의 위치 불일치”, “반복 회차 미확정”을 설명한다. 참고 설명은 원래 참고의 anchor 범위로만 보여주고 다른 사람의 동작 정답으로 복사하지 않는다. 수어가 맞다/틀리다 또는 반드시 이렇게 고치라는 판정은 만들지 않는다.',
 '처리시간: 준비된 관절값에서 새 핵심검색·고정된 앞뒤 비교의 개발영상별3참고 합계(각1회)='+str(s['incremental_seconds_by_dev_query'])+'초. 이전 통합검색은 저장결과를 재사용했다. 전체 검출·TCN·UI를 포함한 게임 지연이나 모든경로 동시실행 측정은 아니다.',
 '쉽게 읽는 핵심 결과: 원본의 같은 구간은 45개 중 45개, 느리게 만든 입력도 45개 중 45개를 찾았다. 두 동작 구간을 수치상 바꿔 놓은 시험에서는 이전 34/43에서 43/43으로 대응이 회복됐다. 단, 이것은 수어가 맞는지 판정한 점수가 아니다. 원본에서 핵심 자체의 비교 근거가 충분한 것은 44/45이고 앞뒤까지 충분한 것은 여전히 36/45이다. 나머지 앞뒤 근거가 새로 생긴 것이 아니라, 앞뒤 부족 때문에 핵심의 관측까지 버리지 않도록 분리한 결과다.',
 '남은 문제: 5프레임을 뺀 시험에서는 대응 중심을 45개 중 43개만 찾아 2개를 아직 놓쳤다. 다른 수행자끼리 비교한 36쌍은 근거를 정리했지만 실제 대응 시간의 독립 정답이 없으므로 정확도 향상을 주장하지 않는다. 기존 입력과 좌우·관측 마스크는 보존했지만 모든 손내림·교차·반복 사례를 검증한 것은 아니다.',
 '한계: 현재 16영상은 개발에 이미 사용된 작은 묶음이다. 원본과 그 수치 변형에서 같은 부분을 찾는 능력이 다른 수행자의 실제 수어 오류 판단까지 증명하지 않는다. 5프레임 삭제의 언어적 필수성, 정상 반복 허용 횟수, 얼굴·손 접촉 정답은 별도 근거가 필요하다. 손이 안 보이는 구간은 계속 보류한다. 창 안의 여러 동점 경로는 아직 모두 열거하지 않는다.',
 '검증: 별도 에이전트의 합성 정책·단어 메타 변경 불변성·저장 경로 비용·원본index/고정분모·해시 검산과 실제 Chrome 영상재생/가독성 검사를 수행했다. 같은 Codex의 검산이며 독립 사람의 한국수어 정오 주석은 아니다.',
 '보존: C·T3-F, 운영웹/DTW, 이전 결과와 원본은 그대로다. 사용자·멘토 자료0, 새 신경망 학습0, 다운로드0, 외부전송0, 카메라0, Git0.',
 '다음 목표: 같은 영상 변형의 수치대응 확인에서 더 나아가, 서로 다른 공개 수행자의 대응 구간을 직접 확인할 작은 평가 묶음을 마련한다. 특히 정상 속도·반복·손내림과 짧은 실제 손모양 차이를 함께 확인한다. 지금 분리한 core/context/미관측 근거를 유지하고 기존 결과를 튜닝용 정답으로 바꾸지 않는다. 교육적 중요한 차이 판정과 학습은 실제 의미·시간 근거가 확보된 특징부터 연결한다.']
 report='\n\n'.join(lines);(U/'RESULT_KO.txt').write_text(report,encoding='utf8');(U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="color-scheme" content="only light"><title>핵심·문맥 분리 결과</title><style>body{background:#fff;color:#152738;font:18px/1.8 system-ui;max-width:1050px;margin:32px}pre{white-space:pre-wrap}a{color:#165b99}</style><h1>핵심 동작과 앞뒤 연결 분리</h1><p><a href="http://127.0.0.1:8794/">영상과 근거 화면 열기</a></p><pre>'+html.escape(report)+'</pre></html>',encoding='utf8')
 status='COMPLETE_GENERIC_CORE_CONTEXT_DIAGNOSTIC_NOT_PRODUCTION_ADOPTED'
 health['exec_session']=82350
 completion=dict(status=status,at=now,active=None,summary=s,report_url='http://127.0.0.1:8794/',server=health,protected_unchanged=True,new_neural_training=0,user_mentor_used=False,all_word_performance_verified=False,review_statuses={n:v['status'] for n,v in reviews.items()},pins={n:sha(U/n) for n in ['FROZEN.json','split_match.py','run.py','SUMMARY.json','CONTROL_EVALUATION.json','PAIR_EVALUATION.json','RESULT_MANIFEST.json','REPORT_DATA.json','index.html','check_viewer.py','UI_VERIFICATION.json','RESULT_KO.txt','RESULT_KO.html','review/PLAN_REVIEW.json','review/SYNTHETIC_CHECK.json','review/SAVED_VERIFICATION.json','review/CORRESPONDENCE_VERIFICATION.json','review/PROTECTED_CHECK.json']})
 save(U/'COMPLETION.json',completion);save(U/'CURRENT_EXECUTION.json',dict(status=status,active=None,server=health))
 statep=OPS/'STATE.json';state=read(statep);state['current_stage']='CORE_CONTEXT_SPLIT_DIAGNOSTIC_COMPLETE';state['active']=None;state['updated_date_kst']=now;state['next_single_action']='Validate stage correspondence across different public performers with independently checked temporal evidence; preserve core/context/missing separation, short observed differences and reference scope. No all-word/educational correctness claim or DEV retuning.';state[U.name]=completion;save(statep,state)
 ident=c['identity'];swap=c['swap_adjacent_core_blocks']
 eight=f'''### 2026-11-01 공통 핵심·문맥 분리 네 단계 완료 — 연구 피드백 연결

- `{U.name}` 공개4표현16영상/126통제+36개인쌍. 단어·사람별예외없이core최소비용먼저고정,앞뒤별도비교/정확동점에서만지원된문맥선택. 원본전체/미관측/이전결과보존,새학습0.
- identity중심 {ident['context_hits']}→{ident['selected_hits']}/45,블록교환 {swap['context_hits']}→{swap['selected_hits']}/43. 실제수어정오/모든단어일반화검증아님. 전체gate는SUMMARY참조,raw/지원/문맥충돌별도.
- 숫자·고정분모·메타불변·화면검산완료. 근거설명만연구화면8794에연결,언어정오/교정지시null. C/T3-F/운영/개인보존,이전8793유지.

'''
 nine=f'''### 2026-11-01 다음 단일 작업 — 다른 수행자 대응 근거 확인

`{U.name}`의완료162결과재실행금지. 공통알고리즘과4표현검증범위를구분한다. 다음은서로다른공개수행자의단계대응을직접확인할작은평가묶음과정상속도·반복·손내림·짧은실제차이근거를준비한다. core/context/미관측과전체참고분모유지,기존결과로임계값튜닝/설명anchor다인정답전파/언어필수성강제금지. 기존C/T3F·운영·개인자료유지. 상세 RESULT_KO/PLAN/SEMANTIC_FEEDBACK_CONTRACT_REVIEW를참조한다.

'''
 cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=cp.read_text(encoding='utf-8-sig')
 for h,v in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',eight),('## 9. 다음 단일 작업',nine)]:
  assert h in txt
  if v.splitlines()[0] not in txt:txt=txt.replace(h,h+'\n\n'+v,1)
 cp.write_text(txt,encoding='utf8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n목표: 한국수어 설명과 여러 참고로 손·모양·위치·움직임·순서를 파악하고 중요한 차이와 관측 부족을 구분한다. 공통 구조이며 모든 단어 성능 검증은 아직 아니다.\n\n'+eight+nine+'\n현재 계산active=null. 읽기전용8794열람서버유지,기존8793/8792/8791보존. health확인후중복실행금지. Python word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. 로컬HTTP/Chrome은require_escalated.\n근거 FROZEN/RESULT_MANIFEST/SUMMARY/CONTROL_EVALUATION/REPORT_DATA/review검산/UI_VERIFICATION/COMPLETION. root구현실행통합, A원인·대응검산, B정책·코드·저장검산, guide의미계약·화면. 같은Codex검산을독립사람수어정답으로부르지않는다.\n',encoding='utf8');print(status)
if __name__=='__main__':main()
