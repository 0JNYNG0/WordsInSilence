from pathlib import Path
import json,hashlib,datetime
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;R=O.parents[2]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 assert R.name=='WordsInSilence'
 normal=read(D/'qualified_checks/RESULT.json');dist=read(D/'qualified_checks/distinctions/RESULT.json');pattern=read(D/'qualified_confirmation/confirmation/PATTERN_CHECK.json');safe=read(D/'safety/RESULT.json');consumer=read(D/'web/CONSUMER_CHECK.json');capture=read(D/'web/CAPTURE_CHECK.json');http=read(D/'web/HTTP_CHECK.json');error=read(D/'web/ERROR_CONSUMER_CHECK.json');server=read(O/'web_tutorial_live_integration_20261113/SERVER.json');trained=read(D/'relation/TRAINED.json')
 assert normal['new_wrong']==normal['lost_correct']==0 and all(r['passed']for r in normal['normal'])
 assert dist['failed']==0 and all(x['passed']for x in[pattern,safe,consumer,capture,error,read(D/'final_confirmation/RESULT.json')])
 assert http['hand_form_active']and not http['camera_active']
 admission=read(D/'web/ADMISSION.json');bad=[p for p,h in admission['pins'].items()if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not bad,bad
 # Preserve all previous scientific pins. Add the reviewed launcher entrypoint.
 admission.update(runtime_verified=True,scope='Public scientific/runtime/HTTP gates passed; physical user confirmation pending.')
 admission['pins'][str(R/'demo_launcher.py')]=hashlib.sha256((R/'demo_launcher.py').read_bytes()).hexdigest();write(D/'web/ADMISSION.json',admission)
 write(D/'PRESERVATION_VERIFIED.json',dict(pins=len(admission['pins']),mismatched=[],C_T3F_unchanged=True,frozen100_unchanged=True,physical_camera=False,private_data_read=False))
 now=datetime.datetime.now().astimezone().isoformat();status='PUBLIC_SHAPE_AND_LEARNED_INSPECTION_FIX_WEB_APPLIED'
 nxt='반복 동작의 끝을 놓치는 원인을 손가락 모양 분류와 분리해 보완한다. 현재 학습한 관계판정부는 검사 구간 제안에만 쓰고 완료 판정 교체는 하지 않는다. 손목·팔·몸 기준 위치와 접근/떼기 흐름을 이용한 후보를 공개 FIT 정답으로 준비하고, 정상 속도·부분 수행·가림·교차·손모양 변경을 함께 검증한다. 사과·감사 정상/자동종료/속도와 새 손모양 오류 검출을 유지한다. 사용자 촬영은 진단 전용이며 학습·튜닝·선택 금지. 실제 오류→교정→재시도 및 HDD분리 시연은 아직 남음.'
 result=dict(status=status,updated_at=now,implementation='Codex local implementation, source review, fixed FIT training and separate regression/runtime checks; no separate expert/agent verification',
  change=['Separate source-compatible open/curved palm from exact individual-finger flexion; positive recovery only in validated upright view, no hidden-joint fill.','Fine-tune three small relation heads on reviewed public FIT evidence; use only for proposing inspection intervals. Completion/events remain unchanged.','Keep persistent distinct-finger-pattern evidence inside the proposed interval; two hand observers must agree. New curled/fist accusations from added intervals are not admitted because side-view false positives remain.','Map the actual difference interval to hand highlighting and a short source-grounded correction; keep exact-finger identity unspecified when unknown.'],
  training=dict(new_source_anchors=32,new_source_clips=3,total_rows=212,rows_include_observed_prefixes=True,seeds=[42,43,44],backbone_changed=False,word_recognizer_changed=False,private_data_used=False,model_hashes=trained['runs'],role='Shared relation observer for inspection proposals; not completion replacement'),
  validation=dict(public_variants=len(normal['normal']),normal_regressions=0,changed_normal=['NIA_SL_WORD1290_REAL07_F'],shape_anchors=len(normal['anchors']),anchor_words=len({r['clip'].split('_')[2]for r in normal['anchors']}),shape_new_wrong=0,shape_lost_correct=0,distinction_passed=dist['passed'],distinction_diagnostic=dist['diagnostic'],pattern_cases=len(pattern['rows']),pattern_before_missed=3,pattern_after_missed=0,pattern_scope='Synthetic hand-pattern replacements from public digits; NOT actual learner accuracy.',normal_speed_partial_missing_controls=len(safe['rows']),control_failures=0,additional_camera_views=4,additional_camera_limit='Persons16/17 fronts already known. 17_L/R were first checked after final code; both remain unknown/zero events, no improvement claim.',consumer_pairs=len(consumer['rows']),tcn_and_dtw_identical=True),
  rejected=[dict(candidate='initial relation replacement',reason='Known DEV anchor lost correct1/new wrong2 and new side-view warnings; never used as completion replacement.'),dict(candidate='unrestricted palm recovery',reason='New coarse-form labelled-anchor error1; positive recovery restricted to the projection model validated upright domain.'),dict(candidate='old relation envelope alone',reason='Missed all3 distinct-pattern controls and introduced1 side-view shape warning.'),dict(candidate='learned inspection with curled/fist class',reason='Found3 missed patterns but introduced2 side-view shape warnings. New intervals admit distinct-pattern class only; original fist checks retained.')],
  runtime=dict(url='http://127.0.0.1:8821/',pid=server['pid'],launcher_pid=3320,entrypoint=server['entrypoint'],camera_active=False,stop_to_compute=[dict(word=w,before=next(r['stop_to_result_seconds']for r in capture['rows']if r['word']==w and r['phase']=='before'),after=next(r['stop_to_result_seconds']for r in capture['rows']if r['word']==w and r['phase']=='after'))for w in['WORD0099','WORD1290']],timing_scope='Single paired public camera fixture; actual RTMW+encoding+TCN+feedback, no physical webcam/browser paint.',all_received_frames_used=True,automatic_end=True,second_detection_passes=0),
  limits=['Front public11 handshape still uncertain; side-view repetition ends are still missed in some sources.','New inspection recovery does not cover every clenched/curved-hand or brief finger change. Existing apple4-frame synthetic fist miss remains unchanged.','Shared components have no word/person input, but full description-linked web integration is currently apple/thanks; other words need semantic contract and source validation.','No user recording was read, trained on, tuned on, or used for selection in this turn. No new actual webcam confirmation.'],next_single_action=nxt)
 write(D/'RESULT.json',result)
 text=f'''이번에 바뀐 점
1. 정상적인 약간의 손가락 굽힘과 손 전체 모양을 나누어 판단합니다. 공개 감사 정상 영상1건에서 손모양 확인불가가 해소됐습니다.
2. 다른 손모양을 알아봤는데도 그 순간을 검사에서 빼고 뒤의 정상 장면만 보던 문제를 보완했습니다. 공개 손모양을 바꿔 만든 시험3건이 통과에서 해당 구간의 교정 안내로 바뀌었습니다. 실제 사용자 오류 정확도 수치는 아닙니다.
3. 실제 추가 학습도 했습니다. 공개3영상의32시점 정답을 보강하고 기존 정답/짧게 끝난 입력을 포함한212학습행으로 작은 관계판정부3개(seed42/43/44)를 미세조정했습니다. 100단어 TCN과 공통 GRU 본체는 고정했습니다. 새 학습값은 검사할 구간을 제안하는 데만 사용합니다. 완료 판정부 전체 교체 후보는 정답 소실과 새 경고가 생겨 채택하지 않았습니다.

확인한 결과
- 기존38공개조건에 새 정상 오판0. 15단어237손모양 기준에서 추가 오판/기존 정답 소실0.
- 기존 오류 구분40조건 유지, 별도 역재생5조건은 정답/오답으로 단정하지 않는 진단용입니다.
- 다른 손가락 모양 합성6조건 모두 차이 확인(이전3조건을 놓침). 정상 속도/위치차이/부분수행/누락48조건 악화0.
- 새 옆시점4개도 확인했습니다. 수행자17의 양옆영상은 기존/새모델 모두 동작을 찾지 못했습니다. 이 두 사례가 정상 통과했다는 뜻은 아닙니다.
- 실제웹 소비부10쌍에서 TCN순위/DTW경로 보존. 합성 손모양 오류도 최종 카드의1.83~2.30초 손 강조와 교정문장으로 연결됐습니다.
- 공개카메라 재생 시험 종료→계산: 사과0.287→0.287초, 감사0.355→0.451초. 재검출0, 받은프레임전부사용,3초카운트다운/손내림자동종료/정상통과유지. 실제웹캠 화면표시시간은 아닙니다.

웹: http://127.0.0.1:8821/ 새로고침. 서버 PID{server['pid']}, 카메라꺼짐. 기존 서버/UI 파일은 그대로 두고 검증된 부품을 연결하는 별도 실행진입점을 사용합니다. START_WEB_DEMO.cmd의 launcher도 이 진입점으로 연결했습니다.
남은 일: 반복 끝 누락, 일부 손모양 불확실, 실제 오류→교정→재시도 확인. 모든 수어 단어에 대한 완성이나 정확도 보장은 아닙니다.
사용자 촬영 읽기/학습/튜닝/후보선택0. C·T3-F와 이전모델/동결결과 보존. 새다운로드/외부전송/커밋0. Codex 자체 검증이며 수어전문가 검증은 아닙니다.
다음: {nxt}
'''
 (D/'RESULT_KO.txt').write_text(text,encoding='utf-8')
 statepath=O/'STATE.json';s=read(statepath);s['current_stage']=status;s['next_single_action']=nxt
 parent=s['capture_feedback_completion_20261116'];parent.update(status=status,updated_at=now,report=str(D/'RESULT_KO.txt'),next_single_action=nxt);parent['public_shape_endpoint_learning_20261116']=result;write(statepath,s)
 context=R/'WordsInSilence_PROJECT_CONTEXT.md';ct=context.read_text(encoding='utf-8-sig')
 section=f'### 2026-11-16 손모양 의미 분리·학습한 검사구간 제안 웹 적용\n공개3영상32시점 보강,212행/3seed 작은 관계판정부 미세조정. 완료 판정 교체는 DEV 정답소실1/새오판2와 옆시점 경고로 미채택. 대신 검사구간 제안에만 사용하고 두 관측부가 동의한 서로 다른 손가락 모양을 빠뜨리지 않게 연결. 정상 손모양1영상 확인회복, 공개합성 모양변경3건 통과→차이경고.38기존공개조건/237손모양기준 추가오판0,40구분+48속도/부분/누락조건 유지. 새옆시점4개 확인하되17양옆은 여전히 동작0/unknown. 웹소비부10쌍TCN·DTW유지,최종오류카드1.83~2.30초손강조/문장확인. 공개캡처자동종료 유지,사과.287/감사.451초(실제웹캠표시시간아님). 8821 PID{server["pid"]} ready/cameraoff. 사용자자료읽기·학습·튜닝·선택0,C·T3-F/기존100공통본체보존. 근거 capture_feedback_completion_20261116/public_shape_endpoint_learning_20261116/RESULT.json·RESULT_KO.txt.\n\n'
 anchor='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';assert anchor in ct;ct=ct.replace(anchor,anchor+'\n'+section,1);anchor='## 9. 다음 단일 작업\n';ct=ct.replace(anchor,anchor+'\n### 2026-11-16 손모양 검사구간 보완 후 반복 끝 판정부 분리\n'+nxt+'\n\n',1);context.write_text(ct,encoding='utf-8')
 (R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {now}

## 현재 결과
손모양 의미 분리+학습한 검사구간 제안+공개 비교/웹 연결 완료. 새가중치학습3seed(작은 관계head),100단어TCN/GRU본체는고정. 검사구간제안에만사용,반복완료교체는미채택.
근거: {D}
RESULT.json / RESULT_KO.txt / qualified_checks / qualified_confirmation / safety / final_confirmation / web.

## 정확히 무엇을 바꿨나
정상 약간굽힌 손과 손전체모양을 분리. 긍정복구는 기존projection모델검증범위 upright만. 원래개별손가락관측값변경/숨은손복원없음. 공개07_F 손모양unknown→supported.
손가락모양을 바꾸면 relation구간이0으로떨어져 차이를검사에서빼고 뒤정상손만보는 원인확인. 공개FIT3영상32시점과기존정답/실제prefix총212행으로 작은관계판정부3seed고정학습. 새모델은검사할구간만제안하고 원래완료·반복events·DTW는유지. 새구간의오류는기존coarse+surface두관측부의지속class2다른손가락패턴동의시에만추가. class1굽힘/주먹은옆시점오판남아새구간경고에미채택. 기존주먹검사유지.

## 검증과 한계
기존38공개조건 새오판0,15단어237기준추가오판/정답소실0. 기존40오류구분통과+5역재생진단. 공개숫자2/6손모양합성6건중기존3건통과잘못표시→새6건모두차이. 실제사용자정확도아님. 속도/크기/부분/누락48조건악화0. 새16/17좌우4영상중17양옆동작0unknown은그대로남음. 미사용새시점이라해도미사용사람검증아님.
초기완료관계교체는DEV정답소실1/새오판2;단순구간확장/전체굽힘경고는옆시점가짜경고로폐기. form_v5.QualifiedFormLesson이최종. 최종raw검사후UI카드는form_presentation이손강조로연결. 합성오류최종소비부1.83~2.30초차이카드확인. 새손가락별세부이름은추정하지않음.
자료라벨/공개소스 검토는Codex 자체.별도에이전트/외부전문가없음.개인자료읽기·학습·튜닝·선택0.

## 웹과 실행
8821 actual PID{server['pid']},launcher3320,ready/cameraoff,HTTP사과감사정상통과.
entrypoint: {server['entrypoint']}
demo_launcher.py만새진입점연결,기존WEB/server.py는수정하지않아이전108핀admission유효.새admission {len(admission['pins'])}핀.
공개카메라fixture 사과122/감사125수신프레임전부관측,재검출0/3초카운트/손내림자동종료유지.종료→계산사과.287→.287/감사.355→.451초.실제웹캠·브라우저시간아님.사용자촬영확인대기.
실험프로세스모두종료.웹서버만유지.원래C·T3-F/동결모델/결과유지.

## 다음 한 작업
{nxt}
''',encoding='utf-8')
 print('FINALIZED',status,'pins',len(admission['pins']),'pid',server['pid'])
if __name__=='__main__':main()
