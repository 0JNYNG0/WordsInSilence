from pathlib import Path
import json,hashlib,datetime,ast
E=Path(__file__).resolve().parent;G=E.parent;O=G.parent;R=O.parents[2];D=G/'public_shape_endpoint_learning_20261116'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
 now=datetime.datetime.now().astimezone().isoformat();base=read(E/'palm_checks/RESULT.json');views=read(E/'palm_views/RESULT.json');fresh=read(E/'unselected_12/RESULT.json');rows=base['normal']+views['rows']+fresh['rows'];capture=read(E/'web/CAPTURE_CHECK.json');consumer=read(E/'web/CONSUMER_CHECK.json');cards=read(E/'web/CARD_CHECK.json');http=read(E/'web/HTTP_CHECK.json')
 pins=read(E/'web/ADMISSION.json')['pins'];bad=[p for p,h in pins.items()if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not bad,bad
 parent=read(D/'web/ADMISSION.json')['pins'];assert all(pins[p]==h for p,h in parent.items())
 for p in E.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
 assert base['new_warning']==base['lost_supported']==base['failed_controls']==0 and views['passed']and fresh['passed']
 assert consumer['passed']and capture['passed']and cards['passed']and http['motion_evidence_active']and not http['camera_active']
 assert read(E/'palm_regression/distinctions/RESULT.json')['failed']==0 and read(E/'palm_regression/confirmation/PATTERN_CHECK.json')['passed']
 shape=read(E/'regression/SHAPE.json');assert shape['new_wrong']==shape['lost_correct']==0
 front=[r for r in rows if r['id'].endswith('_F')];assert len(front)==17 and all(all(v=='supported'for v in r['after']['criteria'].values())for r in front)
 stats=dict(public_conditions=len(rows),public_front_original_clips=len(front),all_required_supported_before=sum(all(v=='supported'for v in r['before']['criteria'].values())for r in rows),all_required_supported_after=sum(all(v=='supported'for v in r['after']['criteria'].values())for r in rows),movement_more_observed=sum(r['after']['events']>r['before']['events']for r in rows),zero_to_two_events=sum(r['before']['events']==0 and r['after']['events']==2 for r in rows),new_normal_warning=0,lost_previously_supported=0,shape_anchors=len(shape['rows']),new_shape_errors=0,lost_correct_shape=0,distinction_controls=40,reverse_diagnostic_only=5,pattern_controls=6,speed_scale_partial_missing_controls=len(base['controls'])+len(views['controls'])+len(fresh['controls']),consumer_pairs=len(consumer['rows']))
 result=dict(status='COMPLETE_PUBLIC_REPAIR_WEB_APPLIED',at=now,stats=stats,training=dict(final_relation=read(E/'palm_relation/TRAINED.json'),palm=read(E/'palm/TRAINED.json'),relation_fit_people=9,relation_fit_clips=12,extra_palm_anchors=24,frozen100_backbone_unchanged=True,used_final_models='51-feature body/palm frame MLP and source-trained local temporal palm head, three fixed seeds each',not_claimed='No natural-language end-to-end learning or whole100 retraining in this turn'),failed_iterations=['Full relation replacement lost short reversals and introduced tail inspection','Fallback full timeline preserved cycles but widened finish inspection','33-feature wrist-only body head missed REAL18_L projected arrival','Added palm training did not resolve foreshortened left palm/fist ambiguity; new region retains unknown for that class'],final_policy='Keep prior cycles/bounds. Add independent approach+source-trained body/palm endpoint cycles. Coarse palm support is qualified separately from precise finger extension. Newly recovered curled/palm ambiguity cannot create a new fist accusation.',checks=dict(shape='regression/SHAPE.json',normal='palm_checks/RESULT.json',known_new_views='palm_views/RESULT.json',final_unselected_views='unselected_12/RESULT.json',distinctions='palm_regression/distinctions/RESULT.json',patterns='palm_regression/confirmation/PATTERN_CHECK.json',consumer='web/CONSUMER_CHECK.json',cards='web/CARD_CHECK.json',capture='web/CAPTURE_CHECK.json',http='web/HTTP_CHECK.json'),web=dict(url='http://127.0.0.1:8821/',pid=http['server']['pid'],entrypoint=str(E/'serve.py'),launcher='START_WEB_DEMO.cmd -> motion_demo_launcher.py',camera_active=False),capture_timing=[{k:r[k]for k in['phase','word','frames','observed','auto_end','redetection','stop_to_result_seconds']}for r in capture['rows']],preserved_pins=len(pins),prior_pins=len(parent),private_data_read_or_trained=False,external_downloads=0,git_commit=False,verification_actor='Codex implementation and separate source/serialized-output/hash checks; no independent agent or KSL expert',remaining=['Some side-view orientation/handshape/target observations remain uncertain; three prior-zero side clips show one observed repetition, not full completion.','Known side-view orientation warnings and prior 4-frame synthetic apple-form miss remain; not changed to a pass.','Real user normal/error/correction/retry and actual browser-visible latency need user-started camera verification.','No claim of all-word generalization or physical hand contact; current consumer integration is apple/thanks.','Full physically unplugged-HDD demo not verified this turn.'])
 write(E/'RESULT.json',result);write(E/'PRESERVATION_VERIFIED.json',dict(at=now,passed=True,pins=len(pins),parent_pins=len(parent),all_hashes_match=True,private_data_used=False))
 graph=[dict(id='A',work='공개 원본과 관측값으로 원인 분리',status='complete'),dict(id='B',work='몸·손바닥 관계와 손 전체 모양을 나눠 추가 학습',status='complete'),dict(id='C',work='마무리 혼입·짧은 반복 소실 회귀 보완 후 정상/오류 비교',status='complete'),dict(id='D',work='새 공개 시점과 부분 수행·정지 조건 확인',status='complete'),dict(id='E',work='웹 문장·강조·자동 종료·속도 연결 확인',status='complete'),dict(id='USER',work='실제 웹캠에서 정상→오류→교정→재시도',status='pending_user_capture_not_training')]
 write(E/'GRAPH.json',dict(at=now,nodes=graph,loop='실제 문제가 확인된 원인 단계로 복귀; 초기 후보의 회귀를 보완한 뒤 통과한 기능만 적용',scope='이번 공통 관측/반복 보완 작업. 프로젝트 전체 완성도 비율이 아님.'))
 report='''이번 보완 결과
1. 손가락 굽힘과 움직임 끝 판정을 분리했습니다. 손목만으로는 부족해 손바닥의 위치·방향도 함께 보도록 작은 판정부를 학습했습니다.
2. 기존에 확인한 동작 구간은 그대로 두고 실제 접근/도착 근거가 추가로 있는 구간만 연결했습니다. 손을 내리는 마무리까지 무작정 검사하지 않습니다.
3. 공개 학습 사람 3명의 좌시점에서 양손 전체 모양 24정답을 추가해 기존142정답과 함께 손모양 판정부를 미세조정했습니다. '약간 굽힌 편 손'과 '정확히 어느 손가락을 폈는가'는 구분합니다. 100단어 TCN/공통GRU 본체와 C·T3-F는 그대로입니다.

이전 → 현재
- 46공개영상 조건 중 모든 요구 항목 확인: 26 → 31조건. 원본 정면17개는 현재 모두 요구 항목 확인. 서로 다른 사람46명이라는 뜻이 아니며, 조건에는 옆시점과 해상도 변형도 포함됩니다. 전체 사용자 정확도 수치가 아닙니다.
- 동작을 전혀 못 찾던 조건10개에서 두 번 동작을 확인. 추가3조건은 한 번만 확인했으므로 완료로 통과시키지 않습니다.
- 새 정상 오답 경고/기존 정상 항목 소실0. 기존15단어237손모양 기준 추가오판·정답소실0. 기존오판1을 해결했다는 뜻은 아닙니다.
- 기존 오류구분40조건, 다른 손가락모양6합성조건, 속도·크기·부분수행·누락56조건 통과. 역재생5조건은 수어가 여전히 성립할 수 있어 진단용입니다.
- 마지막 새12번 수행자 좌우 영상도 확인했습니다. 후보 선택에 쓰지 않은 시점이지만, 그 사람의 정면은 이미 알고 있으므로 새 사람 일반화 검증은 아닙니다.
- 실제 웹 소비부12쌍에서 TCN 후보/DTW경로 유지. 합성 손모양 차이를1.83~2.30초 손 강조와 교정 문장으로 연결; 불확실한 왼손은 회색 확인부족으로 유지.
- 공개카메라 재생: 사과 종료→결과계산0.309→0.307초, 감사0.381→0.481초. 전123/125프레임 관측, 재검출0,3초시작·손내림자동종료·정상통과 유지. 실제웹캠/브라우저 표시시간은 아닙니다.

웹 적용 완료: http://127.0.0.1:8821/ 새로고침. START_WEB_DEMO.cmd로도 최신 버전 시작. 카메라는 꺼져 있습니다.
남은 한계: 일부 옆시점은 손모양·손 방향·도착 위치가 여전히 확인부족입니다. 기존 옆시점 방향 경고와 사과4프레임 합성오류 누락도 이번에 해결했다고 주장하지 않습니다. 3D 접촉이나 모든 단어 성능을 보장하지 않습니다. 실제 새 촬영의 정상→오류→교정→재시도 확인은 사용자 시작이 필요합니다.
사용자 촬영 읽기/학습/튜닝/후보선택0. 기존142핀/그 안의 이전123핀 보존. 새 다운로드/외부전송/커밋0. Codex 자체 검산·원본 검토이며 수어전문가의 독립 검증은 아닙니다.
'''
 (E/'RESULT_KO.txt').write_text(report,encoding='utf-8')
 next_action='웹8821에 적용한 사과·감사를 사용자가 정상→명확한 손모양/방향 오류→교정 재시도로 확인한다. 사용자 촬영은 진단만 하고 학습·튜닝·선택하지 않는다. 새 문제가 재현되면 원인별 공개자료 보완→영향받는 정상/오류 검증→통과한 기능만 적용하는 루프를 계속한다. 현재 46공개조건/237손모양/40구분/6패턴/56보호조건 및 실제공개캡처 검증 완료이므로 근거 없이 반복·학습하지 않는다. 남은 옆시점 불확실을 정답으로 강제하지 않는다. 실제 브라우저 대기시간·교정 재시도·HDD분리 시연은 미검증.'
 state=read(O/'STATE.json');state['current_stage']='PUBLIC_BODY_PALM_MOTION_REPAIR_WEB_APPLIED';state['next_single_action']=next_action;state['last_updated']=now
 state['capture_feedback_completion_20261116']['public_motion_decoupling_20261116']=dict(status=result['status'],at=now,report=str(E/'RESULT_KO.txt'),result=str(E/'RESULT.json'),graph=str(E/'GRAPH.json'),stats=stats,pid=http['server']['pid'],session_id=None,training_private=False,next_single_action=next_action)
 write(O/'STATE.json',state)
 p=R/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf-8-sig')
 heading='### 2026-11-16 몸·손바닥 끝 판정 분리 / 공개 반복 보완 웹 적용'
 assert heading not in text
 update=f'\n\n{heading}\n공개FIT9명12시점영상146앵커로 51개 몸·손바닥 특징 관계head3seed 학습, 추가 양손24정답+기존142정답으로 손모양head3seed 미세조정. 기존 동작구간 보존·추가 접근/도착만 연결. 단순 전체구간교체/손목만 후보의 회귀·누락을 원인별로 보완하고 최종후보 적용.46공개조건 모든항목확인26→31, 원본정면17개 현재모두확인.0→2동작10조건/0→1추가3조건(후자는완료아님).237손모양·40오류구분·6패턴·56속도/부분/누락조건유지. 새로운12좌우도확인(미사용사람검증아님).웹소비부12쌍TCN/DTW유지,오류손강조/불확실구분.공개캡처사과.309→.307/감사.381→.481초,123/125전프레임·자동종료유지(실제웹캠표시시간아님).8821 PID{http["server"]["pid"]} ready/cameraoff.142핀/기존123핀,C·T3-F/100본체/이전결과보존,사용자자료0.근거 {E.relative_to(R).as_posix()}/RESULT.json·RESULT_KO.txt·GRAPH.json.남은옆시점불확실/기존방향경고/실제촬영확인은정확히구분.\n'
 marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';assert marker in text;text=text.replace(marker,marker+update,1)
 marker='## 9. 다음 단일 작업';assert marker in text;text=text.replace(marker,marker+'\n\n### 2026-11-16 공개 보완 완료 후 실제 교정 흐름 확인\n'+next_action+'\n',1);p.write_text(text,encoding='utf-8')
 handoff=f'''# 최신 인수인계 — {now}

## 현재 결과
공개 원인분리→학습→회귀 보완→미사용 시점→웹 연결까지 이번 루프 완료. 사용자 지시: 보완 중 의미있는 새 문제는 멈추지 말고 원인 단계로 돌아가 해결, 사소한 광범위 반복 금지.
근거: {E}
RESULT.json / RESULT_KO.txt / GRAPH.json / web / palm_checks / palm_regression / palm_views / unselected_12.

## 실제 변경과 학습
최종 PalmMotionLesson: 기존 동작/도착 경계 보존. 손가락끝을 제외한 몸·손바닥root/MCP의51특징 관계head + 기존 접근/떼기 관측으로 추가 cycle 연결. 서로 겹치는 cycle은 기존것 유지. 손목33특징 후보는18_L에서근접누락, 손바닥기하추가후회복. 이전전체시간축교체의마무리혼입/짧은반복손실도해소.
관계학습146FIT앵커,9명12영상(9F+1/3/7L),3seed고정. 손모양기존142+새24양손전체모양(FIT1/3/7L)166행3seed미세조정. 100단어TCN/공통Conv+GRU본체 고정. 손전체편모양만검증된upright에서unknown회복; 알려진다른모양/개별손가락값/가린손을정답으로바꾸지않음. 새로회복한검사구간의class1주먹/굽힌편손혼동은unknown,기존주먹차이와class2다른손가락패턴차이는유지.

## 검증/한계
46조건 모든요구항목확인26→31; 정면원본17모두확인. 반복0→2조건10개,추가3개는0→1만확인.새오판/기존정상소실0.237손모양추가오판/정답소실0,기존오판1해결아님.40오류구분+6패턴+56속도/크기/부분/누락통과;역재생5진단만.18좌우는개발로편입,최종12좌우는이번후보선택전미사용시점(사람정면knownDEV).모든단어/새사용자/3D접촉성능보장아님.
웹소비부12쌍TCN순위/DTW경로유지.오류손강조1.83~2.30초/문장,불확실왼손unknown표시확인.공개캡처사과.309→.307/감사.381→.481초,123/125모든수신프레임관측·재검출0·카운트3초/자동종료/통과유지.실제웹캠·브라우저시간아님.
일부옆시점방향경고/손모양·목표위치불확실,기존사과4프레임합성모양누락은남음.이번범위를넘는확실한추가수정근거없이정답으로강제하지않음.
Codex직접구현+별도원본/저장결과/해시검산,독립에이전트/전문가검증없음.개인촬영읽기/학습/튜닝/선택0.

## 실행
http://127.0.0.1:8821/ PID{http['server']['pid']},launcher21860,ready/cameraoff.
entrypoint: {E/'serve.py'}
START_WEB_DEMO.cmd -> motion_demo_launcher.py. 이전demo_launcher.py/WEB서버/이전D모듈/모델그대로,새별도adapter/runtime만삽입.새142핀/기존123핀보존.시작후HTTP사과감사정상통과·motion_evidence_active확인.
학습·검증프로세스종료,웹서버만유지.이전34316과그프로젝트모델자식만중지.새실험중복실행금지.

## 다음 한 작업
{next_action}
'''
 (R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
 write(E/'PROGRESS.json',dict(status=result['status'],at=now,graph=graph,running_sessions=[],server_pid=http['server']['pid'],next_single_action=next_action))
 print('FINALIZED',stats,'pid',http['server']['pid'],'pins',len(pins),flush=True)
if __name__=='__main__':main()
