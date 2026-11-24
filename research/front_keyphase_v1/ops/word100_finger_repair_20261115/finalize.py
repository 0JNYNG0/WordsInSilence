from common import *
import ast,collections,subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
R=O.parents[2];now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();checks=[]
def ck(name,value):
 checks.append(dict(name=name,passed=bool(value)));assert value,name
connection=read(U/'CONNECTION.json');approved=set(connection['files'])
expected={str(O/'web_tutorial_live_integration_20261113/site/feedback.js'),str(O/'web_tutorial_live_integration_20261113/server.py'),str(O/'word100_live_g6_20261114/engine.py'),str(O/'word100_live_g6_20261114/g6_runtime_adapter.py')}
ck('exact_authorized_four_existing_files',approved==expected)
for p,h in read(U/'BEFORE_PINS.json').items():
 if p not in approved:ck('protected:'+p,sha(p)==h)
 else:ck('original_before_hash:'+p,connection['files'][p]['before']==h)
for file in ['INPUTS.json','TRAIN_FROZEN.json','SUPPLEMENT_FROZEN.json']:
 for p,h in read(U/file).get('pins',{}).items():ck('provenance:'+p,sha(p)==h)
for r in read(U/'TRAINED.json')['runs']:ck('trained_weight:'+r['kind']+str(r['seed']),sha(r['model'])==r['sha256'])
rows=read(U/'INPUTS.json')['rows'];a=np.load(U/'INPUTS.npz');p=np.load(U/'SUPPLEMENT_PREDICTIONS.npz')['pred'];y=a['y'];mask=(y>=0)&a['availability']
fit={r['person']for r in rows if r['role']=='fit'};dev={r['person']for r in rows if r['role']=='development'}
ck('people_disjoint',not fit&dev)
ck('new_fit24_dev18',sum(r['new']and r['role']=='fit'for r in rows)==24 and sum(r['new']and r['role']=='development'for r in rows)==18)
ck('public_manifest_only',all(r['clip_id'].startswith('NIA_SL_WORD')and r['role']in['fit','development']for r in rows))
di=[i for i,r in enumerate(rows)if r['role']=='development'];old=[i for i in di if not rows[i]['new']];new=[i for i in di if rows[i]['new']];ring=[i for i in new if y[i,2]>=0 and y[i,3]>=0 and y[i,2]!=y[i,3]]
metrics={}
for name,ix in [('old',old),('added',new),('ring_little',ring)]:
 mm=mask[ix];yy=y[ix];pp=p[ix];base=a['baseline'][ix]
 metrics[name]=dict(known=int(mm.sum()),correct=int(((pp==yy)&mm).sum()),wrong=int(((pp>=0)&(pp!=yy)&mm).sum()),answered=int(((pp>=0)&mm).sum()),unknown=int(((pp<0)&mm).sum()),baseline_correct=int(((base==yy)&mm).sum()),old_correct_to_wrong=int(((base==yy)&(pp>=0)&(pp!=yy)&mm).sum()),old_correct_to_unknown=int(((base==yy)&(pp<0)&mm).sum()))
 for k,v in metrics[name].items():
  saved=read(U/'SUPPLEMENT_POINT_CHECK.json')['metrics'][name]
  if k in saved:ck('recount:'+name+':'+k,saved[k]==v)
ring_fingers={}
for j in [2,3]:
 mm=mask[ring,j];yy=y[ring,j];pp=p[ring,j];bb=a['baseline'][ring,j]
 ring_fingers[['index','middle','ring','little'][j]]=dict(known=int(mm.sum()),correct=int(((pp==yy)&mm).sum()),wrong=int(((pp>=0)&(pp!=yy)&mm).sum()),unknown=int(((pp<0)&mm).sum()),baseline_correct=int(((bb==yy)&mm).sum()))
rr=read(U/'REPLAY_RESULTS.json');ck('122_case_recount',len(rr['rows'])==122);ck('normal64',sum(r['normal']for r in rr['rows'])==64)
for r in rr['rows']:
 d=read(r['output']);ck('source_cache:'+r['output'],sha(d['source'])==d['source_sha256'])
 ck('old_action_selection:'+r['output'],d['before']['original_candidates']==d['after']['original_candidates']and d['before']['original_orders']==d['after']['original_orders'])
 if r['normal']:ck('normal_unchanged_detail:'+r['output'],not r['details'])
 for e in r['details']:ck('same_observed_shape_span',e['coarse_evidence']['span'][0]<=e['span'][0]<e['span'][1]<=e['coarse_evidence']['span'][1])
ck('preconnect_validation_passed',read(U/'CONNECT_GATE.json')['passed'])
ck('worker_and_http_passed',read(U/'WORKER_CHECK.json')['passed']and read(U/'WEB_CHECK.json')['passed'])
ck('physical_camera_not_started',not read(U/'WEB_CHECK.json')['camera_started'])
for f in list(U.glob('*.py'))+[Path(p)for p in approved if p.endswith('.py')]:ast.parse(f.read_text('utf-8-sig'))
subprocess.run(['node','--check',str(O/'web_tutorial_live_integration_20261113/site/feedback.js')],check=True)
ck('finger_overlay_screenshot', (U/'WEB_FINGER_CHECK.jpg').stat().st_size>10000)
for p in approved:connection['files'][p]['after']=sha(p)
connection['repairs_after_first_connection']=['Prevent duplicate Windows servers by exclusive port binding; old and duplicate versions stopped, one latest server retained','Reference seek callback preserves selected hand/finger highlight, matching query side']
connection['verified']=now;write(U/'CONNECTION.json',connection)
fixture=read(U/'UI_FIXTURE_MANIFEST.json');fixture['files']={str(p):sha(p)for p in(U/'ui_check').iterdir()if p.is_file()};write(U/'UI_FIXTURE_MANIFEST.json',fixture)
write(U/'UI_CHECK.json',dict(passed=True,actor='Codex CUA visual/AX verification',normal_public_result='PUBLIC_WEB_RESULT.json',normal_result='Actual public videos both visible; similar, no correction cues',finger_result='Actual stored synthetic-joint verdict; 1.70s seek and named finger cue; selected right hand long fingers highlighted on both panels; short replay works',finger_renderer_fixture_not_DTW_test=True,screenshot=str(U/'WEB_FINGER_CHECK.jpg'),camera=False,temporary_servers_stopped=True))
verification=dict(passed=True,at=now,checks=checks,metrics=metrics,ring_little_only=ring_fingers,protected_unchanged=sum(c['name'].startswith('protected:')for c in checks),actor='Codex separate saved-output/hash/source/visual audit; same implementer, not independent expert',unrestricted_observer_release=False,conditional_web_refinement=True,scope='Regression and supplementary development observation; no fresh blind learner-error accuracy')
write(U/'VERIFIED.json',verification)
write(U/'FAILURES_AND_REPAIRS.json',dict(items=[
 dict(issue='Standalone expanded whole-hand and shared-finger candidates caused old-source regressions',resolution='Kept both results; did not replace old coarse observer. Admitted compatible supplemental evidence only inside existing attributed handshape-difference spans',remaining='One point-level index error remains; supplement not sufficient for unrestricted correction'),
 dict(issue='Two local Windows servers shared port8821 and old feedback_version responded',resolution='Stopped only identified project servers/workers; exclusive port bind; repeated same public HTTP check passed latest version'),
 dict(issue='Supplemental reference overlay selected both hands after video seek',resolution='Preserve focused cue hand/finger selection through seek callback; visually checked both panels'),
 dict(issue='Temporary renderer fixture server lacked Range support, seek returned0',resolution='Added Range handling to isolated test server. Production server already supports Range; exact1.70s seek verified'),
 dict(issue='Fixture helper imported conflicting legacy common module',resolution='Executed production pure presentation function via AST without model/camera initialization; model runtime itself passed isolated worker and web checks'),
 dict(issue='Initial source-preparation syntax typo',resolution='Corrected before labels/data freeze and training'),
 dict(issue='Sandbox refused localhost HTTPsocket',resolution='Local-only read/public replay repeated through approved escalated execution')]))
report=f'''손가락 조합 보강과 제한된 웹 연결 — 2026-11-15

이번에 끝낸 일
A. 기존 오판 3장면의 원본 RGB와 관절을 앞뒤로 확인했다. 정답 주석을 바꾸어 점수를 올리지 않았다. 손이 돌아가거나 겹치는 장면에서 오판이 나타났으며, 기존 학습 정답에 약지와 새끼가 서로 다른 조합이 없던 점을 확인했다. 모든 오판의 원인이 이것 하나라고 입증한 것은 아니다.
B. 공개 원본 42시점을 재검토했다. 24시점은 학습, 18시점은 별도 사람의 개발 비교에 사용했다. 그중 약지는 펴고 새끼는 굽힌 조합은 학습 3명·6시점, 개발 3명·6시점이다. 굽힌 손가락을 무조건 주먹이라 부르지 않으며 엄지는 이번 교정 대상에서 제외했다.
C. 손 전체를 보는 시간 합성곱+GRU와 손가락별 관절 관계를 보는 시간 합성곱+GRU를 각각 3개 고정 초기값으로 학습했다. 총6회, 각600단계이며 검증 결과를 보고 가장 잘 나온 회차만 고르지 않았다. 새 보조 판정부는 이번에 초기화하여 학습했고, 기존100단어 공통 모델과 C·T3-F는 재학습/교체하지 않았다. 최종 학습 범위는 19단어92공개영상의444손 시점, 개발 비교는15단어42영상의244손 시점이다. 100단어 전부에 새 손가락 정답이 생긴 것은 아니다.
D. 통과한 방향 보완과 조건부 손가락 설명을 실제 웹에 연결하고, 카메라 없이 공개 관절값을 TCN→DTW→최신 판정부→화면 표시까지 통과시켰다. 새 촬영도 같은 경로를 사용하지만 실제 웹캠 확인은 사용자가 미룬 상태를 유지한다.

이전과 현재
- 약지/새끼 자세가 다른 6시점에서 검지·중지·약지·새끼를 각각 평가: 총24항목. 기존 공통 판정6항목 확인→새 보조22항목 확인, 2항목 판단 보류, 오답0. 이24는 영상24개나 단어24개가 아니다.
- 위24항목 중 약지는 기존0/6→새 보조6/6 확인. 새끼손가락은 기존6/6→새 보조4/6 확인, 나머지2개 보류. 모든 손가락이 똑같이 개선된 것은 아니다.
- 기존 개발 정답895손가락 항목에서는 기존 공통 관찰807정답→보조840정답/1오판/54보류. 기존에 맞던 항목을 새 오답으로 바꾼 것은0이나, 단독 사용시 기존 정답 중 {metrics['old']['old_correct_to_unknown']}항목은 보류로 바뀐다. 따라서 새 보조를 기존 판정부 전체와 맞바꾸지 않았다.
- 별도 공개122조건(원본20영상에서 정상·속도·방향·순서·누락·손모양 변형) 비교: 정상/속도/완료64조건의 새 손가락 경고0. 원래 행동 구간 후보·순서 판정과 기존 손모양 근거 유지. 손모양 변형5조건에 구체적인 손가락 안내가 추가됨. 실제 사용자가 틀리게 촬영한 영상의 검출률은 아님.
- 이전에 검증한 방향 보완도 함께 연결. 손모양+반대방향 대조는 기존 방향 보완 결과를 유지했다. 이번42시점 손가락 학습으로 방향까지 배웠다는 뜻은 아니다.

웹에서 달라진 점
예전: '손모양이 다르게 관측됩니다.'
현재: 이미 손모양 차이가 확인된 같은 구간에서 보조 판정도 일치하고 관측이 이어질 때, '1.7초 부근 · 오른손의 엄지를 제외한 네 손가락을 펴 보세요'처럼 시점과 손가락을 표시한다. 네 손가락 전체가 아니라 일부에서 근거가 있으면 검지/중지/약지/새끼손가락을 구분한다.
좌우 손과 단어에 같은 코드를 사용한다. 단어명 분기로 오답을 만들지 않았다. 문장 요구사항과 행동 구간을 잇는 기존 구조를 유지하며, 임의 자연어 문장을 이해하는 LLM은 아니다. 모델이 관측 특징을 내고, 검증된 근거를 정해진 짧은 문장으로 표시한다.
서로 충돌하거나 가려진 경우 보류한다. 기존 행동 구간을 바꾸거나 불확실한 구간을 오류로 승격시키지 않는다. 이번 손가락 보조만으로 새로운 행동 오류를 독립적으로 찾아내는 기능은 열지 않았다.

속도 / 연결
122조건에서 추가 방향+손가락 보완만 중앙값 약0.036초,95백분위0.051초. 이것은 전체 분석 시간이 아니다.
실제 웹 분석 경로에 공개 잘하다1영상의 저장 관절값을 보낸1회는 약{read(U/'WEB_CHECK.json')['seconds']:.2f}초. 촬영·RTMW검출·화면 표시 대기는 제외. 기존 전체와 동일 조건 속도 개선 실험이 아니므로 빨라졌다고 주장하지 않는다.
중복 서버가 이전 판정을 응답하던 문제를 고쳤다. 현재 최신 서버 하나만 http://127.0.0.1:8821/ 에서 준비. 카메라는 켜지지 않았다.

남은 문제와 다음 한 작업
검지가 펴진 한 시점(WORD2551 REAL12 f35)을 굽힌 것으로 읽는 오판1건이 남았다. 정면 축소·가림 상태에서는 관절만으로 판단을 잘못할 수 있다. 그래서 개별 손가락 판정의 전면 교체/무조건 교정은 하지 않는다.
다음은 기존에 후보 선택에 쓴 개발자료와 분리한 공개 연속 구간에서 왼손·약지/새끼 조합과 손이 정면을 향하는 검지를 확인하는 일이다. 정답과 관찰 가능성을 먼저 고정하고, 정상 전체 구간에 불필요한 경고가 생기는지와 오류의 정확한 시점을 평가한다. 이 검증 전에는 교정 범위를 넓히지 않는다. 엄지·벌림·손바닥 방향·접촉까지 모두 해결했다고 말할 수 없다.
사용자 촬영·멘토 촬영은 학습/튜닝/후보 선택에 사용하지 않았다. 사용자 실제 촬영 확인은 보류. C·T3-F,100단어 공통 모델,원본 동결 산출물 보존 해시 확인.

근거
SOURCE_REVIEW.json / COVERAGE.json / TRAIN_FROZEN.json / TRAINED.json / ENSEMBLE_COMPARISON.json
SUPPLEMENT_POINT_CHECK.json / REPLAY_RESULTS.json / CONNECT_GATE.json / WORKER_CHECK.json / WEB_CHECK.json
UI_CHECK.json / CONNECTION.json / VERIFIED.json / FAILURES_AND_REPAIRS.json
작성·실행·별도 검산: Codex. 독립 수어 전문가 검수가 아니며, 개발자료 재사용을 포함한 제한된 연구 결과이다.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')
next_action='기존 후보 선택에 쓰지 않은 공개 연속 구간에서 왼손·약지/새끼 조합과 정면 검지의 정답/관찰가능성을 먼저 고정하고 정상 경고·손가락 오류 시점을 비교. 검증 전 교정 범위 확대 금지. 사용자 촬영 확인은 보류 유지.'
summary='공개42시점 재검토(FIT24/DEV18), 손가락 시간 합성곱+GRU 2구조×3seed 학습 완료. 단독 후보는 기존 오판 증가로 교체하지 않고, 이미 확인된 손모양 차이의 같은 구간에 한해 합의한 손가락 보조만 웹 연결. 분리된6시점의24항목6→22정답/2보류, 기존895항목840정답/1오판/54보류; 미사용 블라인드 성능 아님. 공개122조건 중 정상64새경고0. C·T3-F/공통100동결. 실제 웹 공개 관절 경로 최신버전 확인, 카메라 미사용.'
server=read(O/'web_tutorial_live_integration_20261113/SERVER.json')
write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_LIMITED_SUPPLEMENT_CONNECTED',at=now,training_running=False,pid=None,web=server,temporary_ui_servers_stopped=True,physical_camera_confirmation='USER_DEFERRED',next_single_action=next_action))
state=read(O/'STATE.json');state['current_followup']=U.name;state['next_single_action']=next_action;state[U.name]=dict(status='COMPLETE_LIMITED_SUPPLEMENT_CONNECTED',at=now,summary=summary,report=str(U/'RESULT_KO.txt'),verification=str(U/'VERIFIED.json'),web=server,feedback_version='G6-finger-20261115',physical_camera_confirmation='USER_DEFERRED',training_runs=6,common100_changed=False,C_T3F_changed=False,private_training=False,next_single_action=next_action);write(O/'STATE.json',state)
doc=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
 marker=heading+'\n\n';pos=doc.index(marker)+len(marker);end=doc.find('\n### ',pos+1)
 if end<0:end=doc.find('\n## ',pos+1)
 assert end>pos
 entry='### 2026-11-15 손가락 조합 보강 완료 · 제한 보조 웹 연결\n\n'+body+'\n\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`, `VERIFIED.json`. 구현/실행/별도 검산 Codex, 독립 전문가 검수 아님. 최신 웹8821은 준비 상태이며 사용자 촬영 자동 시작 금지.\n'
 doc=doc[:pos]+entry+doc[end:]
(R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(doc,'utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# 손가락 조합 보강 / 조건부 웹 연결 완료\n\n'+summary+'\n\n현재 근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`, `VERIFIED.json`, `CONNECTION.json`, `CURRENT_EXECUTION.json`.\n\n다음 한 작업: '+next_action+'\n\n웹 http://127.0.0.1:8821/ PID '+str(server['pid'])+' (재개 시 생존/포트 확인). 최신 G6-finger-20261115. 임시8830/8831 표시 점검 서버 종료, 학습/실험 프로세스 없음. 카메라는 사용자 촬영 시작 전까지 꺼짐. W: 미연결 상태에서 C:공개 캐시만 사용.\n\n보호: C·T3-F/공통100 가중치/동결 원본 변경 없음. 사용자 자료 학습·튜닝·후보 선택 금지. 이번 구현·실행·별도 검산은 Codex, 독립 전문가 검수 아님. 새 모델 다운로드/LLM/3D/Unity 작업 없음.\n','utf-8')
for p in [O/'STATE.json',U/'VERIFIED.json',U/'CONNECTION.json',U/'CURRENT_EXECUTION.json']:read(p)
print(json.dumps(dict(passed=True,checks=len(checks),protected=verification['protected_unchanged'],metrics=metrics,ring_little_only=ring_fingers,web=server),ensure_ascii=False))
