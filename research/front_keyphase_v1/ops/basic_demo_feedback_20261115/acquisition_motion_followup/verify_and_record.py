"""Separate post-run checks and concise handoff; does not rerun model experiments."""
from motion_review import *
import ast,datetime,shutil

def main():
    now=datetime.datetime.now().astimezone().isoformat();O=U.parent;R=O.parents[2]
    assert not(D/'VERIFIED.json').exists()
    for p in D.rglob('*.py'):ast.parse(p.read_text('utf-8'),filename=str(p))
    for p in D.rglob('*.json'):read(p)
    protected=read(U/'PLAN.json')['protected']
    assert all(sha(p)==h for p,h in protected.items())
    assert all(sha(p)==h for p,h in read(D/'MOTION_FROZEN.json')['protections'].items())
    acq=read(D/'GREETING_ACQUISITION_SCOPE.json');assert acq['downloaded_corpus_bytes']==0
    assert acq['both_label_archives_bytes']==95541420
    comparison=read(D/'MOTION_CORRECTED_COMPARISON.json');fit=[r for r in comparison['rows']if r['split']=='fit'];dev=[r for r in comparison['rows']if r['split']!='fit']
    assert {r['person']for r in fit}.isdisjoint({r['person']for r in dev})
    assert len(fit)==48 and len(dev)==16
    diag=read(D/'BRIDGE_RESULTS.json');confirm=read(D/'confirmation/RESULTS.json')
    assert diag['passes']==diag['total']==28
    assert confirm['summary']['gratitude_checks']==confirm['summary']['gratitude_total']==7
    normal=next(r for r in confirm['rows']if r['kind']=='normal');assert normal['arrival_native_match'] is True
    prefix=next(r for r in confirm['rows']if r['kind']=='first_only')
    # Missing a partial event is not a false completion, but also not a correct count.
    limitation=dict(case=prefix['clip_id'],kind='first_only',expected_known_arrivals=1,
        observed=len(prefix['new_events']),previous_observed=len(prefix['old_events']),
        no_false_completion=True,complete_partial_count=False,
        interpretation='Both earlier and new observers miss the single arrival when the clip stops immediately afterwards. Do not call35 diagnostic safety conditions full accuracy.')
    sources=read(D/'confirmation/EXTRACTED.json')['rows']
    for r in sources:
        assert sha(r['path'])==r['sha256'] and sha(r['raw_path'])==r['raw_sha256'] and sha(r['channel_path'])==r['channel_sha256']
    assert all(r['person'] not in range(1,14) for r in sources)
    report=dict(at=now,status='SOURCE_SCOPE_CONFIRMED_AND_OFFLINE_COMPONENT_IMPROVED',actor='Codex implementation and separate checks, not independent external expert',
        network='Official metadata449973bytes only; no corpus download, credentials or private data',
        local_public_clips=len(sources),local_public_frames=sum(r['frames']for r in sources),
        local_video_bytes=sum(Path(r['path']).stat().st_size for r in sources),
        movement_training_runs=9,fit_people=9,fit_intervals=48,dev_people=3,dev_intervals=16,
        development=comparison['summary'],diagnostic_safety_conditions=35,partial_coverage_limit=limitation,
        new_confirmation=confirm['summary'],previous_missed_cycle=dict(person=17,old=1,new=2,role='previously opened diagnostic, not new final test'),
        modified_production_files=[],web_admitted=False,whole_word_completion=False,C_T3F_preserved=True,
        protected=protected,private_recordings_used=False,active_training_processes=[],server_restarted=False)
    write(D/'VERIFIED.json',report)
    summary='''안녕하세요 ZIP·용량 확인과 기초단어 공통 피드백 후속

1. 자료 확인
AIHub 공식 파일목록을 새로 조회했다. 2026-10-11 저장목록과 파일별 용량이 일치한다.
- 확정한 SEN0354 REAL01 영상의 원래 ZIP: [원천]01_real_sen_video.zip, filekey39625, 29,139,269,902바이트(29.14GB).
- 그 안의 01/NIA_SL_SEN0354_REAL01_F.mp4는 4,501,585바이트이며 이미 C:에 있다. 이 대표1개 때문에 ZIP01을 다시 받을 필요는 없다.
- 먼저 확인할 학습 문장주석: [라벨]01_real_sen_morpheme.zip, filekey39584, 84,798,396바이트(84.80MB).
- 필요시 검증 문장주석: 01_real_sen_morpheme.zip, filekey494853, 10,743,024바이트(10.74MB). 두 주석 합95.54MB.
- 전체 문장 영상33ZIP 약1.048TB를 받을 필요가 있다고 판단하지 않았다.

공식 aihubshell의 filekey로 ZIP별 선택 다운로드는 지원한다. ZIP 안의 특정MP4만 원격 선택하는 기능은 확인하지 못했다. 다른 수행자의 SEN0354가 어느 ZIP에 배치됐는지는 아직 미확정이며 ZIP번호=수행자번호로 추정하지 않는다. 주석으로 필요한 클립 목록을 먼저 찾고, 해당 ZIP 매핑을 확인한 다음 필요한 ZIP만 순차 확보하는 계획이다. ZIP 전체를 로컬에서 풀지 않고 해당 정면 파일만 추출하는 것은 가능하지만, 이것이 전송량도 그만큼 줄여준다는 뜻은 아니다.

공식 Shell은 병합·해제를 위해 다운로드의2~3배 여유공간을 권한다. 알려진29.14GB ZIP1개라면 약58~87GB다. 조회 당시 C:약263GB/W:약1.81TB 여유가 있다. 이것은 실제 필요한 다인자료 총량이 확정됐다는 뜻이 아니다. 이번에는 영상·주석 다운로드를 하지 않았다.
공식 방법: https://aihub.or.kr/devsport/apishell/list.do?currMenu=403
공식 데이터: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=103
근거: GREETING_ACQUISITION_SCOPE.json / OFFICIAL_CATALOG_CURRENT.json / 기존 acquisition.json.

2. 감사·공통 피드백
손의 가까움/멀어짐이라는 자세와 별도로, 가까워지는 움직임·벌어지는 움직임·유지를 배우는 작은 공통 판정부를 학습했다. 단어ID/사람ID/정답반복수는 신경망 입력에 넣지 않았다. 기존100단어 Conv+GRU·TCN은 동결했다.
공개9명48구간 학습 / 별도3명16구간 비교. 단순선형, 작은MLP, 손윤곽정보 추가MLP를 총9회 학습했다. 검토 중 개발 주석2개가 방향전환을 한 방향으로 묶은 것을 발견하여 원본 순차프레임으로 구간만 바로잡았다. 학습자료·임계값은 바꾸지 않았다. 이전 결과는 보존했다. 프레임읽기 오류 가능성도12영상에서 확인했으며 OpenCV요청프레임과 순차PyAV원본의 픽셀이 정확히 같았다.
바로잡은 개발16구간에서 기존 자세상태만 이용한 비교는14맞음/1오판/1보류, 작은움직임MLP는16맞음이다. 개발 중 정정한 소규모 자료이므로 전체 정확도 상승으로 일반화하지 않는다. 손윤곽 추가는 이익이 없어 더 단순한 움직임모델을 후보로 유지했다.
학습한 움직임과 실제 가까운 자세의 관측을 연결하자, 이전 공개17번의 두 번 중 한 번만 확인하던 사례를 두 번으로 관측했다. 기존14·15·16번도 두 번으로 유지했다. 현재 후보 선택에 쓰지 않은18번에서도 두 번을 실제 원본의 도착구간 안에 찾았다.
5공개영상의 정상/한번만수행/정지/미검출/타임스탬프속도변환/위치·크기변환35조건에서 없는두반복을 만들어내지 않는 기준을 통과했다. 다만18번을 첫동작 직후 끊은 경우는 이전과 새 방식 모두 첫1회도 보류했다. 따라서35조건을 완벽한반복인식률로 보고하면 안 된다. 속도변환은 프레임을 버리지 않은 타임스탬프 시험이며 실제웹캠고속성능이 아니다.

3. 사과
후보선택에 쓰지 않은 공개14·15번 원본을 HDD에서 로컬확보·RTMW추출·RGB검토했다. 기존100+사과 후보에서2개 모두 사과1순위. 입술부근검지동작의 직접확인9시점×4손가락=36칸 중 기존손가락판정35맞음/0오판/1보류였다. 독립36영상이 아니라2영상내확인값이다. 다른주먹을 닦는 뒷동작은 아직 세부완료 판정이 준비되지 않았다. 단어이름을 맞췄다고 전체동작정답 처리하지 않는다.

4. 남은 일과 실제 적용
이번 수정·학습은 별도연구폴더 안에서만 실행했다. 기존 C·T3-F,100공통학습,정지유지 손모양연결과 현재웹/서버는 변경하지 않았다. 사용자촬영을 열거나 학습·튜닝에 쓰지 않았다.
새3공개영상 총382프레임의관절을 추출했고, 감사움직임후처리는 원본1개 약0.037초였다. 이 값은 관절추출·촬영·DTW·화면표시를 제외하며 전체대기시간이 아니다.
다음은 검증된 움직임관측을 설명필수항목에 연결하고, 짧게끝난첫동작도 관측근거를 보존하는지를 확인한다. 사과는 검지→반대손상호작용 순서와 닦기완료의 정답·판정을 보강한다. 손모양·손방향·관계·반복을 모두 충족하는 실제정상완료생성기와웹연결은 여전히 남아 있다. 원문 설명과 보이지 않는 손가락을 임의로 학습정답으로 채우지 않는다.

담당: Codex 구현·실행 후 별도검산. 독립 수어전문가 검수는 아니다.
'''
    (D/'RESULT_KO.txt').write_text(summary,encoding='utf-8')
    # Preserve the previous handoff before updating canonical status.
    before=D/'before';before.mkdir(exist_ok=True)
    for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json',U/'TASK_GRAPH.json']:
        shutil.copy2(p,before/p.name)
    next_action='검증된 공통 움직임 관측을 설명의 필수항목 완료근거에 연결. 첫동작 직후 종료한 공개18번의 부분관측 누락과 사과 검지→닦기 순서를 보완. 안녕하세요는84.8MB SEN주석을 우선 확보할 범위가 확정됐으며 다른수행자 ZIP매핑/실제다운로드는 아직. C·T3-F/개인촬영학습금지 유지.'
    graph=read(U/'TASK_GRAPH.json');graph['at']=now;graph['next']=next_action
    for n in graph['nodes']:
        if n['id']=='F2a':n['note']='SEN0354 사용자확정. 공식ZIP/용량/선택다운로드방법 확인; 다인ZIP매핑·다운로드 전. 감사·사과 추가공개3원본 검수.'
        if n['id']=='F2b':n['note']='독립움직임학습으로17번 반복1→2,선택미사용18번2회확인. 사과추가2영상인식/검지근거확인. 전체필수항목완료·짧은부분관측은 남음.'
    write(U/'TASK_GRAPH.json',graph)
    state=read(O/'STATE.json');state['next_single_action']=next_action
    entry=state['basic_demo_feedback_20261115'];entry['next_single_action']=next_action
    entry['acquisition_motion_followup']=dict(path=str(D),report=str(D/'RESULT_KO.txt'),verified=str(D/'VERIFIED.json'),
        greeting_download_bytes=0,motion_diagnostic_old_new=[1,2],new_gratitude_person=18,apple_new_correct=2,
        partial_prefix_missed=True,whole_word_completion=False,web_admitted=False)
    entry['learning_runs']=30;entry['active_training_processes']=[]
    write(O/'STATE.json',state)
    note=f'''\n### {now[:10]} 안녕하세요 선택확보 범위 확인 / 공통 움직임 보완 추가검증\n\nAIHub 공식목록 재조회: SEN학습주석84.80MB(file39584), 검증주석10.74MB(file494853). 선택한REAL01대표의원래ZIP01은29.14GB(file39625),대표4.50MB는이미C에있음. ZIP별선택다운로드지원/내부MP4만원격받기는미확인. 다른수행자ZIP매핑미완료,전체1TB재다운로드불필요. 실제다운로드없음.\n공통움직임9명48구간학습·3명16구간개발. 원본검토로방향전환이섞인개발주석2정정(이전결과보존). 작은움직임판정부+가까운자세관측연결로기존17번반복1→2,새18번두번/원본도착구간일치. 사과새14·15두영상인식1위,검지구간손가락36칸35맞음1보류. 단어전체완료/웹채택은아님. 18번첫회직후절단은이전/새방식모두첫회보류이므로35안전진단통과를완벽인식률로보고금지.\n총이번9학습종료·프로세스없음. C·T3-F/100공통/현웹/사용자촬영불변. 구현·별도검산Codex. 근거 `research/front_keyphase_v1/ops/basic_demo_feedback_20261115/acquisition_motion_followup/RESULT_KO.txt`, `VERIFIED.json`.\n\n'''
    ctx=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8')
    for heading in ['## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 9. 다음 단일 작업']:
        at=ctx.index(heading)+len(heading)
        ctx=ctx[:at]+(note if heading.startswith('## 8.') else f'\n\n### {now[:10]} 기초단어 후속\n{next_action}\n\n')+ctx[at:]
    (R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(ctx,encoding='utf-8')
    hand=f'''# 최신 목표 — 기초 단어로 작동하는 피드백 게임 ({now})

시연 표현: 사용자확정 AIHub안녕하세요SEN0354 → 사과WORD0099 → 감사WORD1290. 기존4단어는공통회귀전용. C·T3-F/100공통유지, 사용자촬영 학습·튜닝·후보선택 금지.

## 마지막 완료
`research/front_keyphase_v1/ops/basic_demo_feedback_20261115/acquisition_motion_followup/RESULT_KO.txt`, VERIFIED.json.
안녕하세요 공식파일목록 현재조회 완료: 학습SEN주석84.80MB/검증10.74MB. 대표영상 원래ZIP01은29.14GB,대표4.50MB는이미C에있으므로그파일때문에재다운로드하지않음. 다른수행자ZIP매핑미정. filekey로ZIP선택가능,원격MP4단위는미확인. 영상/주석네트워크다운로드없음. 상세GREETING_ACQUISITION_SCOPE.json.

## 공통 피드백 보완
전체시간문맥GRU와 정지유지손모양연결은보존. 손가까움자세와 움직이는방향을나눠지도: 공개9명48구간FIT,3명16구간DEV. 9작은모델실행종료. 개발주석2가방향전환을가로질러 잘못한방향으로붙어있어순차RGB로수정,학습데이터/임계값변경없음. 옛출력보존. 수정DEV16은MLP16일치,기존자세상태비교14일치1오판1보류;정정한개발자료라독립최종성과로포장금지.
독립움직임+실제가까운자세관측 연결로기존P17감사 두번중한번만잡던것1→2. 기존P14/15/16두번유지. 선택미사용P18도두번/원본도착구간일치. 5영상×7진단35조건은허위완료안전검사이며완벽인식률아님: P18첫동작직후절단은옛/새모두첫1회관측도0(보류)라부분근거회복이남음. timestamp변환은실제프레임드롭/카메라속도검사아님.
사과P14/15새공개2영상준비/RTMW추출/RGB검토:100+사과후보에서둘다1위. 검지구간9시점×4손가락36칸중35맞음0오판1보류. 뒷부분닦기완료는미검증. 전체동작정답이나웹채택아님. 이번추가공개3영상382프레임,네트워크가아닌HDD원본재사용.
새보완경로가운영파일에들어간것은아님. 기본모델/웹/서버변경없음. 임시후처리약.037초는관절추출/촬영/DTW/UI제외.

## 이전 보완과 남은 것
basic_demo_feedback_20261115/shape_consensus.py: 기존개별손가락을보존하고보강전후공통손모델합의만추가. DEV237 208/1/28→227/1/9(맞음/오판/보류);공개16·17의17정답10/0/7→12/0/5. P17오른손보류는아직남음.
web_tutorial_live_integration_20261113/contract.py,feedback_acceptance.py,site/live.js는이전턴에TCN이름일치만으로퀘스트통과하지않게소스수정. **실제정상완료생성기없어정상도완료근거없으면보류한다. 게임완성아님.** 서버재시작안함/8821상태미확인. 카메라는사용자가나중에시작하기로함.

## 다음 한 작업
{next_action}
안녕하세요후보선택을다시묻지말것. 인사법/평안/반갑다/NIKL인사를대신학습시키지말것. 이전WORD HDD는33영상ZIP27만항목SEN0이며문장ZIP이없다. 기존선택영상은runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4.
현재정상완료생성기엔 손날/손등방향·필수모양·반복/순서·단계완료가모두필요. 소규모관측개선을완성으로승격금지. 새P18/사과14·15는이번에열었으므로다음후보의새최종시험으로재사용하지말것.

## 실행 상태
누적30작은학습실행종료(초기정답정정전6포함),진행중프로세스없음. Codex구현/별도검산,독립전문가아님. 보호모델해시검증완료. 사용자촬영접근없음.
Python: word100_signrep_feasibility_20261029/runtime/Scripts/python.exe -X utf8 -B.
root STATE current_stage는basic_demo_feedback_20261115. 연구계획F2는부분완료, F3정상완료/웹연결·F4실제촬영·F5완성시연은남음.
'''
    (R/'NEW_CHAT_HANDOFF.md').write_text(hand,encoding='utf-8')
    assert read(O/'STATE.json')['next_single_action']==next_action
    print(json.dumps({k:report[k] for k in ['status','local_public_clips','local_public_frames','local_video_bytes','modified_production_files','partial_coverage_limit']},ensure_ascii=False))

if __name__=='__main__':main()
