from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,html,urllib.request
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];PREV=O/'word100_crossperson_stage_audit_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 f=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in f['pins'].items())
 checks={n:read(U/'review'/n) for n in ['PLAN_REVIEW_DRAFT.json','CODE_REVIEW.json','SAVED_POLICY_VERIFICATION.json']}
 assert all(d['status'].startswith('PASS') for d in checks.values())
 ui=read(U/'UI_VERIFICATION.json');assert ui['status']=='PASS' and ui['html_sha256']==sha(U/'index.html') and ui['data_sha256']==sha(U/'REPORT_DATA.json')
 assert all(sha(PREV/n)==h for n,h in read(PREV/'COMPLETION.json')['pins'].items())
 protected=read(O/'word100_semantic_episode_pilot_20261031/FROZEN.json')['protected'];assert all(sha(p)==h for p,h in protected.items())
 auth=read(U/'AUTHORIZATION.json');assert all(sha(r['input'])==r['input_sha256'] for r in auth['rows'])
 health=json.load(urllib.request.urlopen('http://127.0.0.1:8796/health',timeout=4));assert health['unit']==U.name and health['read_only'] and not health['inference']
 sums=read(U/'SUMMARY.json')['summary'];audit=read(U/'POLICY_EVIDENCE_AUDIT.json');d=audit['counts']['development'];ff=audit['counts']['fit']
 lines=['전체 동작 흐름을 지키는 보완 — 완료 결과',
 '쉽게 말하면: 두 사람의 춤 영상에서 두 번째 손올림은 두 번째끼리, 움직이는 순간은 움직이는 순간끼리 비교해야 한다. 이번에는 기존 전체 흐름의 짝을 지키고, 별도 자세 검색이 다른 시간대로 튀었으면 확실한 근거로 채택하지 않게 했다.',
 '계획 검토에서 확인한 사실: 기존 전체 DTW의 남은 물리 단계 불일치20링크는 모두 같은 활동의 이동↔유지 전환이었다. fit12링크/개발8링크이며 참고 프레임별 같은 무게로는fit11.5/880=1.31%, 개발8/440=1.82%. 모두 적어도 한쪽이 주석상 확실한 내부 범위의 첫/끝 프레임이었다. 해당 링크의13그룹63셀은 모두 관측되어 있었으므로, 이를 손 검출 누락이나 큰 순서 혼동 탓이라고 설명할 근거는 없었다. 이20링크와 경계 주석은 변경하거나 분모에서 제외하지 않았다.',
 '따라서 계획을 좁혔다: 새 가중치로 전체 경로를 다시 계산하지 않고, 원래 경로와 모든 관측을 보존하는 검증 부품 하나를 구현했다. 별도 후보가 전체 흐름의 해당 범위와 떨어지면 위치·회차 충돌, 겹치지만 벗어나면 경계 이견, 안에 들어오면 범위 양립으로 기록한다. 앞뒤 후보의 역순/겹침과 관측 부족은 별도 보류한다. 범위 양립은 정확한 단계·손 모양·수어 정답을 증명하지 않는다.',
 '실행: 공개4표현/4수행자ID/16영상의36쌍117단계와 기존126수치통제에 이 정책을 적용했다. 사람쌍은 기존 semantic_episode 전체 DTW, 수치통제는 relational 특징의 전체 DTW라는 다른 기원이므로 같은 정렬 정확도로 합산하지 않았다. 새 검색·신경망 학습·검출·다운로드는0. 주석은 사후 분석에만 사용했고 정책은 단어명·사람ID·주석정답을 입력받지 않는다.',
 '개발12쌍39단계의 범위 판정: '+json.dumps(sums['dev']['range_relation'],ensure_ascii=False)+'. 참고끼리24쌍78단계: '+json.dumps(sums['fit']['range_relation'],ensure_ascii=False)+'.',
 f"개발 결과의 이득과 비용: 별도 후보가 다른 반복 회차였던 {d.get('different_ordinal',0)}개 중 {d.get('different_ordinal_held',0)}개를 미확정으로 남겼다. 반면 같은 물리종류·같은 기록회차였던 {d.get('same_kind_same_ordinal',0)}개 중 {d.get('same_kind_same_ordinal_held',0)}개도 보류했다. 나머지 경계/미확인은 그대로 남겼다. 이는 보류가 많을수록 좋은 정책이라는 뜻이 아니며, 정상 후보까지 확신을 줄이는 보수성 비용이다. 같은 종류/회차도 교육적 정답 주석은 아니다.",
 f"참고끼리 결과: 같은종류·같은회차 {ff.get('same_kind_same_ordinal',0)}개 중 {ff.get('same_kind_same_ordinal_held',0)}개가 보류됐다. 다른종류 {ff.get('different_kind',0)}개 중 {ff.get('different_kind_held',0)}개, 다른회차 {ff.get('different_ordinal',0)}개 중 {ff.get('different_ordinal_held',0)}개가 보류됐다. 세부 항목은 POLICY_EVIDENCE_AUDIT.json에 전부 남겼다.",
 '구체적인 미해결 반례: P071/P070 s02는 별도 후보와 전체 흐름의 범위가 양립해도 RGB 물리 주석상 다른 단계인1건이다. 두 방법의 범위가 일치한다는 것만으로 정답이라는 결론을 내릴 수 없다. 정책의 부족함을 숨기거나 이1건을 단어별 예외로 막지 않았다.',
 '수치 통제: 원본/0.75속도/1.25속도 재표본의 각45단계는 모두 같은 전체흐름 범위 안에 있었다. 기존 관측 부족에 의해 각각1/1/6단계는계속보류됐다. 오른손전체미관측은45/45보류. 5프레임삭제·손모양값변경·구간교환은원경로와원차이값이그대로남았다. 이는실제사람이빠르게다시촬영한수어정오시험이아니며,범위양립이짧은동작누락까지정상이라는의미도아니다.',
 '직접 볼 장면: 조상 P066/P063 s05의 별도 검색73프레임(첫 상승)은 전체 흐름96프레임(두 번째 상승)과 충돌하므로 채택하지 않는다. 순종 P036/P034 s04의 별도 유지 범위도 전체 흐름의 이동→유지 범위와 경계 이견으로 남긴다. 이는 수행자가 잘못했다는 피드백이 아니다.',
 '성능 해석: 기존 DTW 경로와 관측 수치는 모두 그대로이므로 실제 정렬 점수나 단어 인식률은 올라가지 않았다. 기존 개발 대표31/39, 고정 핵심 범위의 같은 단계 대응67.2%도 그대로다. 바뀐 것은 별도 검색 후보를 확정 근거로 쓰는 방식과 화면의 근거 표시다. 전체게임 지연이나 실제 사용자 정확도는 이번에 측정하지 않았다.',
 '검증: 별도 계획검토, 코드/합성51,625검사, 저장결과627,196검사, 실제 Chrome 영상 재생·72조건·117구간·사례4버튼·가독성 검사를 완료했다. 최초검산의83차이는동결전에추가한source-state방어규칙을준비검산기가누락한것이었고최초실패를보존한뒤검산기만수정했다. 본체정책과결과의재실행은0. 원본16입력/보호32파일/이전 완료보고서 보존. 검증자는 Codex 에이전트들이며 독립 한국수어 교육자의 정오 판단을 대신하지 않는다.',
 '남은 문제와 다음 목표: 움직임에서 멈춤으로 바뀌는 경계는 여전히 애매할 수 있다. 전체 흐름과 별도 후보가 다르다는 사실만으로 어느 쪽이 정확한지 알 수 없고, 정상 후보를 너무 많이 보류하는 비용도 확인됐다. 다음은 경로를 다시 바꾸기보다 기존 경로의 관측 지원되는 손모양·몸 관계·이동 차이를 시간 범위와 함께 설명하는 제한된 피드백을 연결하는 것이다. 정확한 경계 한 프레임이나 필수 반복 횟수를 강제하지 않는다. 이4표현은 이미 소모한 개발 자료이며 모든 단어에 대한 일반화 주장은 별도 미사용 표현·사람 자료에서 검증해야 한다.',
 '확인 화면: http://127.0.0.1:8796/ — 기존8795 보존. C·T3-F/운영웹/DTW 보존, 개인·멘토자료 미사용.']
 text='\n\n'.join(lines);(U/'RESULT_KO.txt').write_text(text,encoding='utf8');(U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>전체 흐름 보존 결과</title><style>body{background:white;color:#17304a;font:18px/1.8 system-ui;max-width:1080px;margin:30px}pre{white-space:pre-wrap}a{color:#175b90}</style><h1>전체 흐름을 지키는 보완 결과</h1><a href="http://127.0.0.1:8796/">영상으로 보기</a><pre>'+html.escape(text)+'</pre>',encoding='utf8')
 now=datetime.now(timezone(timedelta(hours=9))).isoformat()
 health['exec_session_id']=42522
 complete=dict(status='COMPLETE_GLOBAL_PATH_PRESERVED_LOCAL_EVIDENCE_POLICY',at=now,active=None,report_url='http://127.0.0.1:8796/',server=health,alignment_accuracy_improved=False,production_adopted=False,training=0,private_data_used=False,summary=sums,evidence_counts=audit['counts'],checks={n:v['status'] for n,v in checks.items()},pins={n:sha(U/n) for n in ['FROZEN.json','SUMMARY.json','RESULT_MANIFEST.json','POLICY_EVIDENCE_AUDIT.json','REPORT_DATA.json','index.html','UI_VERIFICATION.json','RESULT_KO.txt','RESULT_KO.html','review/SAVED_POLICY_VERIFICATION.json']})
 save(U/'COMPLETION.json',complete);save(U/'CURRENT_EXECUTION.json',dict(status=complete['status'],active=None,server=health))
 state=read(O/'STATE.json');state.update(current_stage='GLOBAL_FLOW_LOCAL_GUARD_COMPLETE',active=None,next_single_action='Keep global path; connect only supported shape/body/movement difference evidence with time ranges, not exact phase or linguistic correctness. Address unnecessary local withholding without DEV threshold tuning. New generalization needs unused public expressions/people.');state[U.name]=complete;save(O/'STATE.json',state)
 latest=f'''### 2026-11-01 전체 흐름 보존·국소 후보 검증 완료

- `{U.name}` 기존wholepath/수치/missing변경0,공개36쌍117단계+126수치통제정책실행. 단어/사람/주석을정책입력으로쓰지않음. local disjoint/범위이견/역순겹침/미관측보류,contained도정확단계증명X.
- 기존global불일치20links는모두move↔hold주석interior단부,link자체63셀관측전부지원;큰회차/반대손오류0. 기존DEV31/39·핵심경로67.2%불변,새정렬정확도개선아님.
- DEV다른회차 {d.get('different_ordinal_held',0)}/{d.get('different_ordinal',0)} 보류,같은종류/같은회차도 {d.get('same_kind_same_ordinal_held',0)}/{d.get('same_kind_same_ordinal',0)} 보류하는비용공개. 계획/합성/저장/Chrome검증완료. 읽기전용8796,8795보존. 개인/멘토·C/T3F·운영/원본보존.
- FIT P071/P070_s02 다른단계인데compatible인1건잔존(기존global도differentkind). 범위합의=정답금지. 속도재표본각45범위충돌0/기존보류1·1·6;오른손미관측45/45보류,실제속도재촬영정오아님.

'''
 nexttext=f'''### 2026-11-01 다음 단일 작업 — 지원되는 차이 근거를 시간 범위로 설명

`{U.name}` 완료정책/117분모/162결과중복실행금지. 원globalDTW를유지하고손모양·몸관계·이동의지원되는차이를시간범위와연결하는제한된피드백을구체화한다. local충돌만으로global정답/사용자오류확정X. 정상후보보류비용을숨기거나DEV로임계값조정X. 이동→유지경계/필수반복수/교육정오강제X. 4표현소모개발과미사용평가분리,새일반화주장은별도공개표현·사람필요. 기존C·T3F/운영/개인자료보존. RESULT/COMPLETION/8796참조.

'''
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig')
 for h,v in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',latest),('## 9. 다음 단일 작업',nexttext)]:
  assert h in t
  if v.splitlines()[0] not in t:t=t.replace(h,h+'\n\n'+v,1)
 p.write_text(t,encoding='utf8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n목표: 한국수어 설명과 여러 공개 참고로 손·모양·몸관계·움직임·순서를 파악하고 중요한차이/허용변이/관측부족을구분한다.\n\n'+latest+nexttext+'계산active=null.8796은16공개영상읽기전용보고서버. health/CURRENT_EXECUTION확인해중복실행금지. 이전8795이하보존. root구현/정책실행/화면/문서,guide계획,A원인·저장검산,B코드합성. 같은Codex검증이며독립사람수어정답아님. Python word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. 로컬HTTP/Chrome require_escalated.\n',encoding='utf8')
 print(complete['status']);print(json.dumps(audit['counts'],ensure_ascii=False))
if __name__=='__main__':main()
