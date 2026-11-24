from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,html,urllib.request
from run import U,OLD,read,sha
ROOT=U.parents[3]; OPS=U.parent
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    f=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in f['pins'].items())
    reviews={name:read(U/'review'/name) for name in ['SYNTHETIC_CHECK.json','SAVED_VERIFICATION.json','CORRESPONDENCE_VERIFICATION.json']}
    assert all(v['status'].startswith('PASS') for v in reviews.values())
    ui=read(U/'UI_VERIFICATION.json');assert ui['status']=='PASS' and ui['html_sha256']==sha(U/'index.html') and ui['data_sha256']==sha(U/'REPORT_DATA.json')
    protected=read(OPS/'word100_semantic_episode_pilot_20261031/FROZEN.json')['protected']
    assert all(sha(p)==h for p,h in protected.items())
    old=read(OLD/'COMPLETION.json');assert all(sha(OLD/n)==h for n,h in old['pins'].items())
    save(U/'review/PROTECTED_CHECK.json',dict(status='PASS',protected_files=len(protected),previous_pins=len(old['pins']),unchanged=True))
    health=json.load(urllib.request.urlopen('http://127.0.0.1:8793/health',timeout=4));assert health['unit']==U.name and health['read_only'] and not health['inference']
    s=read(U/'SUMMARY.json'); c=s['controls']; now=datetime.now(timezone(timedelta(hours=9))).isoformat()
    lines=['앞뒤 움직임으로 반복 회차 찾기 — 완료 결과',
    '목표: 수어 설명과 여러 참고를 이용해 어느 손·모양·관계·움직임이 어떤 순서로 이루어지는지 파악한다. 이번에는 같은 자세가 반복될 때 잘못된 회차를 고르는 문제에 한정해 새 비교 후보를 검증했다.',
    '새 결과 화면 http://127.0.0.1:8793/ / 이전 화면8792는 유지.',
    '', '1. 원인과 계획 검토',
    'P063의 기존 DTW는 원본142프레임 대각선으로 정상이었다. 그 뒤 평균 자세검색에서 s05의13그룹 중6개만 남고 이동방향·오른손 관련정보가 빠져, 실제시점보다후반비슷한자세를골랐다. 새후보는그룹을평균지지율로삭제하지않는다.',
    '짧은 왼손변화는관측없는3개/범위없는4개/평균에묻힌2개/범위초과3개로분리했다(fit12). P046·P047의편집관측칸은핵심구간의26.3%·31.25%여서평균이작아졌다. 최고차이값은기존평균기준과바꿔비교할수없다.',
    '', '2. 실제 구현',
    '각참고핵심전체와앞뒤0.20초를연속된관측으로사용했다. query전체에서각핵심을독립적으로찾으며,다음핵심은반드시뒤에있어야한다는강제규칙은없다. 창길이는참고프레임수0.5~2배,최대연속가로/세로2회,대칭가중DTW로전체창을보존한다.',
    '정상손내림/관측없는참고그룹은오류로채우지않는다. 참고에보이는데query에없는차원은기대분모에남고부족으로표시된다. 핵심·진입·이탈별그룹근거와다른회차후보를따로저장한다. 가까운수치는확률이나정답확정이아니다.',
    '서로다른개별참고3개를유지했다. 공개4표현16영상(fit12/dev4),개인간36쌍/이전126통제의새후보한번계산. 이전점수·원본배열·기존전체경로를보존했다. 창밖영상도정상이라고지우지않는다.',
    '', '3. 수치대응 검증 결과(교육적 정오/단어인식률 아님)']
    for name,v in c.items():
        lines.append(f"{name}: {v['cases']}영상/{v['stages']}핵심; 원본core안중심 기존{v['old_center_hits']}→새{v['new_center_hits']}; 기존관측지원정확후보{v['old_supported_hits']}→새앞뒤까지지원정확후보{v['new_fully_supported_hits']}; 역순관계후보{v['old_reverse_relations']}→{v['new_reverse_relations']}; 새상태{v['new_states']}")
    lines += ['',f"정상원본/느린재표본 raw대응비회귀={s['gates']['raw_same_source_no_regression']}, 앞뒤지원정확응답비회귀={s['gates']['supported_correct_coverage_no_regression']}.",
    '보류만 늘려 역순경고를 줄인 경우는 완전개선으로 판정하지 않는다. 조건별지원기준이달라원시대응과지원된응답을함께보아야한다. 작은동일출처통제결과를새사람정확도/모든수어에확대하지않는다.',
    '빠른재표본은원본프레임삭제를포함하며실제빠른수행검출시험이아니다. 5프레임삭제는언어적필수단계정답이없다. 블록교환은앞뒤문맥전체교환이아니므로새문맥비교에서반드시역순검출해야하는정답으로승격하지않는다.',
    '다른사람개발12쌍은정확한세부대응정답이없어새시점의정확도는아직미확정이다. core는기존물리초안이며설명한사람을다른사람정답으로복사하지않았다.',
    '', '4. 시간·검산·보존',
    '준비된관절값CPU새문맥검색3참고합계(영상별1회)='+str(s['development_pair_seconds_by_query'])+'초. 관절검출/TCN/UI포함전체게임지연아님. 모든후보창검색을검증용으로수행해기존경로보다무거울수있다.',
    '알고리즘작은합성문제검증,별도저장경로/비용/지원/대응/집계검산,해시보존,양쪽영상과HTML가독성검사를완료했다. 같은Codex의별도검산이며독립농인교육자정답검증은아니다.',
    '기존C·T3-F/운영웹/DTW유지. 사용자·멘토자료0,새신경망학습0,다운로드0,카메라0,외부전송0,Git0.',
    '', '다음 작업',
    '우선순위는 핵심 동작 자체의 대응과 앞뒤 연결의 차이를 분리하는 것이다. 핵심 구간을 바꾼 경우 앞뒤 문맥 때문에 원래 흐름과 비슷한 다른 구간으로 끌려가지 않도록, 대응과 연결 불일치 근거를 함께 남기는 계획을 세운다. 그다음 짧은 왼손 변화가 전체 평균에 묻히지 않도록 관측된 짧은 구간별 증거를 보존한다. 과거 점수에 맞춰 허용폭을 줄이거나 새 모델을 무작정 추가하지 않는다.',
    '완료는이번한후보의계획·구현·비교·검산이며전체교육피드백AI완성아님. 운영교체없음.']
    report='\n\n'.join(lines);(U/'RESULT_KO.txt').write_text(report,encoding='utf8')
    (U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="color-scheme" content="only light"><title>앞뒤 움직임 비교 결과</title><style>body{background:#fff;color:#162a3a;font:18px/1.8 system-ui;max-width:1000px;margin:32px}a{color:#065a9c}pre{white-space:pre-wrap}</style><h1>앞뒤 움직임과 반복 회차</h1><p><a href="http://127.0.0.1:8793/">영상 비교 화면 열기</a></p><pre>'+html.escape(report)+'</pre></html>',encoding='utf8')
    status='COMPLETE_CONTEXT_SEQUENCE_DIAGNOSTIC_NOT_PRODUCTION_ADOPTED'
    completion=dict(status=status,at=now,active=None,summary=s,report_url='http://127.0.0.1:8793/',report_server=health,protected_unchanged=True,review_statuses={k:v['status'] for k,v in reviews.items()},pins={n:sha(U/n) for n in ['FROZEN.json','context_match.py','run.py','SUMMARY.json','CONTROL_EVALUATION.json','PAIR_EVALUATION.json','RESULT_MANIFEST.json','REPORT_DATA.json','index.html','RESULT_KO.txt','UI_VERIFICATION.json']})
    save(U/'COMPLETION.json',completion);save(U/'CURRENT_EXECUTION.json',dict(status=status,active=None,report_server=health))
    statep=OPS/'STATE.json';state=read(statep);state['current_stage']='CONTEXT_SEQUENCE_DIAGNOSTIC_COMPLETE';state['next_single_action']='Separate core-trajectory correspondence from entry/exit context mismatch so changed core order is not attracted to an unchanged-context occurrence; retain full-path evidence and explicit missing support. Then address short left-hand evidence dilution without DEV threshold tuning.';state['active']=None;state[U.name]=completion;save(statep,state)
    ident=c['identity'];slow=c['speed_075']
    eight=f'''### 2026-10-31 앞뒤 문맥·반복회차 고정 비교 완료 — 운영 미채택

- `{U.name}` 공개16/36개인쌍/126재사용통제 새후보1회. 평균자세검색의그룹소실원인확인후core±0.20s순서trace,각core전체query독립검색/문맥부족보류. 새신경망학습0.
- 원본중심적중{ident['old_center_hits']}→{ident['new_center_hits']}/{ident['stages']},느린재표본{slow['old_center_hits']}→{slow['new_center_hits']}/{slow['stages']}; 앞뒤지원정확응답비회귀={s['gates']['supported_correct_coverage_no_regression']}. raw대응과지원/보류분리,숫자대응개선은교육정오AI완료아님.
- 기존C/T3F/전체DTW/원본/개인보존. 별도산술·합성문제·화면검증완료. 결과8793읽기전용,이전8792유지. 상세SUMMARY/RESULT_KO/COMPLETION/review참조.

'''
    nine=f'''### 2026-10-31 다음 단일 작업 — 회차 대응과 부족한 문맥·짧은 왼손 근거 분리

`{U.name}`의완료162결과재실행금지. 우선 핵심 동작 대응과 앞뒤 연결 차이를 분리해, 핵심만 바뀐 구간이 정상 문맥과 비슷한 다른 회차로 끌리는 회귀를 다룬다. 원본/느린속도통제의숫자대응,앞뒤문맥부족,다른사람의미확인대응을분리한다. 새창검색을운영전체DTW로교체하지않는다. 그다음짧은왼손차이는원관측과전체경로를유지한구간별증거로보강하되기존평균기준을최대값에잘못대입하거나DEV튜닝하지않는다. 개인/멘토자료·새모델순회·다운로드·운영교체금지. 설명anchor의다인정답전파/언어필수성강제없음.

'''
    contextp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=contextp.read_text(encoding='utf-8-sig')
    for head,addition in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',eight),('## 9. 다음 단일 작업',nine)]:
        assert head in txt
        if addition.splitlines()[0] not in txt:txt=txt.replace(head,head+'\n\n'+addition,1)
    contextp.write_text(txt,encoding='utf8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n목표: 한국수어 설명과 여러 참고의 동작 순서·손모양·신체관계를 파악하고 중요한 차이와 관측부족을 구분한다.\n\n'+eight+nine+'\n계산active=null; 새8793서버읽기전용열람유지,기존8792/8791보존. 실제health확인후중복실행금지. Python word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. 로컬HTTP/Chrome은require_escalated필요.\n\n구현root,계획/저장검산rgb_labels_b,원인/합성/대응검산rgb_labels_a,의미계약/화면guide_stage_map. FROZEN이후고정후보162파일,완료추론재계산금지. 같은Codex검산을독립사람수어정답으로부르지않는다.\n',encoding='utf8')
    print(status)
if __name__=='__main__':main()
