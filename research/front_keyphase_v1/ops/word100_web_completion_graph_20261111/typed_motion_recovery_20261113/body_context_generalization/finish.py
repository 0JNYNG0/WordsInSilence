"""Close the bounded C6 diagnostic and synchronize the single resume point."""
from pathlib import Path
import datetime,hashlib,json
U=Path(__file__).resolve().parent;H=U.parent;G=H.parent;O=G.parent;ROOT=O.parents[2]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8');t.replace(p)
at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
cfg=read(U/'FROZEN.json');v=read(U/'SAVED_VERIFICATION.json');c=read(U/'COMPARISON.json');r=read(U/'RESIDUAL_AUDIT.json');demo=read(U/'DEMO_WORD_READINESS.json')
assert v['status']=='PASS'and v['comparison_sha256']==sha(U/'COMPARISON.json')
for p,h in cfg['pins'].items():assert sha(Path(p))==h,p
for p,h in cfg['protected_pins'].items():assert sha(Path(p))==h,p
assert c['status']=='DIAGNOSTIC_REWORK' and not c['gate']['motion_family_preserved']
assert not(U/'DECISION.json').exists()
summary=('C6 공개 FIT6기록ID·24영상의 사람제외 6분할에서 같은150값을 현재 자세+앞뒤변화로 재표현한 판정부를 고정3초기값으로 비교했다. '
 '움직임822항목은690/676/686→773/769/777, 자세유지522항목은455/454/456→518/518/518로 함께 개선됐다. '
 '다만 팔따라쓸기 가족은seed42에서174/180→169/180으로 감소해 진단 전체통과/웹채택은 아니다. '
 '3뷰·정역의 반복 관측수이며 독립 새사용자/전체 피드백 정확도가 아니다. 기존 가중치 복사 없이 새 작은 판정부36개를 비교했고 TCN·공통100·기존 웹은 보존했다.')
nxt=('C7: C6의 새 움직임 손실15개 고유 원본시점(경계범위 안8개 포함)을 FIT RGB·몸기준 경로와 다시 대조한다. '
 '기존 직접 움직임 정답은 유지하고, 실제 작은 움직임/관측 기준축 문제/모델 일반화 문제를 분리한 뒤 공통 보완 하나를 설계한다. '
 '현재 입력에도 시간정보가 있었고 새 재표현은 같은 정보다. 36개 완료모델 재학습·재추론, 특정단어 예외, DEV/probe 문턱조정 금지. '
 '이 개선을 G2 통과나 새 웹성능으로 올리지 않는다. G2/G3 이후의 최종8·웹교체·Unity병합은 보류한다. '
 '시연은 잘하다 기존범위를 우선 검증하고 감사/즐겁다는 뜻·영상변이·시간주석 준비를 통과해야 후보에서 채택할 수 있다.')
decision=dict(at=at,status='C6_AGGREGATE_GAIN_FAMILY_REGRESSION_NOT_ADOPTED',summary=summary,
    diagnostic_gate=c['gate'],G2_pass=False,web_changed=False,common100_changed=False,C_T3F_changed=False,
    scope='24 historically exposed public source videos /6 recorded IDs; 6 leave-one-ID-out folds,20train+4test sources each;3views×forward/reverse.',
    naming='absolute_context is before/current/after BODY-RELATIVE geometry, not absolute pixel coordinates. relative_context is the same information reorganized as current and differences; all masks retained.',
    normalization='Upstream affine geometry statistics inverted (including availability-weighted contexts), then normalization fitted only on each five-ID training split. No parent learned weight initialization or hidden embeddings.',
    supervision='Existing sparse body movement/hold observations; neither correct/incorrect-sign labels nor completion labels. Unknown and uncertain boundaries retained.',
    runs=36,seeds=[42,43,44],steps=1200,architecture='150->96 GELU->5 scratch heads; paired exact same trainable initialization.',
    comparison=c['aggregate'],source_observation_losses=r['lost_motion_unique_source_points'],
    saved_checks=v['checks'],independent_human_review=False,
    no_private=True,no_probe_DEV_final8=True,no_old_forward=True,no_game_integration=True,next=nxt,
    evidence=[dict(path=str(p),sha256=sha(p))for p in [U/'FROZEN.json',U/'INPUT_AUDIT.json',U/'PRETRAIN_VERIFICATION.json',U/'TRAINING_COMPLETE.json',U/'COMPARISON.json',U/'SAVED_VERIFICATION.json',U/'RESIDUAL_AUDIT.json',U/'DEMO_WORD_READINESS.json']])
write(U/'DECISION.json',decision)
(U/'RESULT_KO.txt').write_text('\n'.join([
 '움직임과 자세 유지: 같은 정보를 다르게 제공한 비교', '',
 '결론: 전체 움직임과 자세 유지가 함께 개선되는 근거를 얻었다. 하지만 일부 팔 쓸기 동작 손실이 남아 아직 웹에 적용하지 않았다.', '',
 '1. 비교한 것',
 '기존 비교입력: 몸 기준 앞·현재·뒤의 위치. 새 비교입력: 현재 위치와 앞뒤 변화량. 양쪽 모두 신체 기준이며 화면 절대좌표가 아니다. 새 센서나 추가 시간정보를 넣은 것은 아니다.',
 '기존 부모 가중치는 복사하지 않고 같은 작은 판정부를 새로 학습했다. 공개6기록ID·24영상에서 한 ID의4영상을 빼고 나머지5ID·20영상으로 학습하는6분할. 세초기값과두입력으로36회 고정비교했다. 이는 서비스에서36개 모델을 함께 쓰겠다는 뜻이 아니다.',
 '정규화는 기존 전체 통계를 역변환한 뒤 각 학습5ID로 다시 계산. 모든 조건의 정답·학습횟수·가중비중·판정문턱을 먼저 고정했다.', '',
 '2. 결과(초기값42/43/44 각각)',
 '실제 움직임: 690/676/686 -> 773/769/777 (각822 관측항목)',
 '자세 유지: 455/454/456 -> 518/518/518 (각522 관측항목)',
 '관측항목은 원본시점·동작종류·3화면조건·정역 반복을 포함한다. 822명/822영상이 아니며 독립 최종시험이나 사용자 정확도가 아니다.',
 '팔을 따라 쓸기 가족은 초기값42에서174/180 ->169/180으로 감소했다. 다른두초기값은164/180 ->168/180이다. 세초기값의 새로운 움직임 손실은33/33/27 반복항목, 고유 원본시점으로 합치면15개다.',
 '그중8시점은 기존 시작/끝 경계 불확실 범위에 있다. 그렇다고 직접 확인한 움직임 정답을 지우거나 유지로 바꾸지 않았다. 단순 경계 근처라는 사실만으로 원인을 단정하지 않는다.', '',
 '3. 이번에 확인하지 않은 것',
 '손모양·방향·전체 동작 순서·교차·완료·교육적 정오의 종합 성능과 실제 웹 지연은 새로 평가하지 않았다. 이 몸동작 부품의 향상이 전체 피드백의 향상으로 확정된 것은 아니다.',
 '모든 학습자료는 공개 자료이며 사용자·멘토 촬영은 사용하지 않았다. C·T3-F·공통100·기존 웹을 유지했다. 보호477파일 해시가 일치한다.',
 '입력1828항목, 저장결과2066항목을 별도 코드로 검산했다. 같은 Codex의 검산이며 한국수어 전문가의 독립 정답 검증이 아니다.', '',
 '4. 시연 후보 준비',
 '잘하다·감사·즐겁다 모두 기존100단어에 포함되며 각각공개학습9/개발3영상이 있다. 현재 세부 피드백 단계 연결은 잘하다2단계가 있고 감사/즐겁다는 이 연결자료에서0단계다. 다른 모든 과거 주석이 없다는 뜻은 아니다.',
 '감사는 기존 원본검토에서 사전과 영상의 손바닥 방향이 미확정이다. 따라서 게임 이야기에 어울린다는 이유만으로 해당 방향을 교정 정답으로 사용할 수 없다. 새단어 학습·게임병합은 하지 않았다.', '',
 '다음: '+nxt,'']),encoding='utf-8')
status=decision['status']
cur=read(G/'CURRENT_EXECUTION.json');cur.update(at=at,status=status,training_running=False,process_running=False,
    active_unit=str(U),session_id=None,other_session_id=None,jobs=[],current_command=None,last_completed=str(U/'DECISION.json'),next=nxt)
write(G/'CURRENT_EXECUTION.json',cur)
sg=read(H/'SUBGRAPH.json');sg.update(at=at,status=status)
node=next(n for n in sg['nodes']if n['id']=='C6');node.update(status='COMPARED_AGGREGATE_GAIN_FAMILY_REWORK',evidence=[str(U/'DECISION.json')],task=summary)
assert not any(n['id']=='C7'for n in sg['nodes'])
sg['nodes'].append(dict(id='C7',title='새 움직임 손실의 관측·정답·일반화 원인 분리',status='NEXT',task=nxt,evidence=[str(U/'RESIDUAL_AUDIT.json')]))
write(H/'SUBGRAPH.json',sg)
graph=read(G/'TASK_GRAPH.json');node=next(n for n in graph['nodes']if n['id']=='G2');node['status']='REWORK'
node['attempts'].append(dict(at=at,experiment_id='C6_FIT_LEAVE_ID_OUT_CONTEXT_PAIR',status='REWORK',
    checks=dict(diagnostic_overall_motion_hold_improved=True,diagnostic_family_preserved=False,G2_runtime_not_evaluated=True),
    evidence=[dict(path=str(U/'DECISION.json'),sha256=sha(U/'DECISION.json'))],note=summary))
graph['updated']=at;write(G/'TASK_GRAPH.json',graph)
state=read(O/'STATE.json');state.update(updated_at=at,last_updated=at,next_single_action=nxt,next_single_task=nxt)
state['word100_web_completion_graph_20261111'].update(at=at,status=status,active_unit=str(U),training_running=False,progress=summary,remaining=nxt,latest_evidence=str(U/'DECISION.json'))
write(O/'STATE.json',state)
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text('utf-8-sig');title='### 2026-11-13 자세·변화 분리 비교 완료 · 움직임과 유지 동시 개선, 일부 동작 회귀'
assert title not in s
for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:
    assert heading in s;s=s.replace(heading,heading+'\n\n'+title+'\n'+body+'\n근거: `'+U.relative_to(ROOT).as_posix()+'/DECISION.json`.\n',1)
p.write_text(s,'utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — 2026-11-13 C6 비교 완료

{summary}

다음 단일 작업: {nxt}

U={U.relative_to(ROOT).as_posix()}
PY=research/front_keyphase_v1/ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe -X utf8

C6는 기존BP/head 재추론이 아니라 새로운동등초기화36개 진단 비교다. 6leave-ID-out×3seed×2표현,각1200단계,seq144/24영상/6ID02..07. 학습완료session28471exit0. PRETRAIN1828/SAVED2066검산PASS. 완료결과재학습/재추론금지. 별도산술검산은같은Codex,독립사람/KSL정답아님.

absolute_context라는 내부이름은몸기준앞현재뒤값이지화면절대좌표가아니다. relative_context는현재+before-current+after-current좌표,기존mask를보존하는가역재표현. upstream평균/표준편차는checkpoint의qm/qs통계만읽고역변환했으며기존학습가중치/latent사용없음. 각fold훈련5ID만정규화. 과거연구에노출된FIT진단으로독립최종새사람성능아님.

세seed움직임822=690/676/686→773/769/777,ownhold522=455/454/456→518/518/518. family4seed42=174→169/180손실. 추가실패33/33/27반복항목,고유15시점(기존경계불확실범위8). 정답삭제/hold재라벨금지. FIT훈련움직임은양조건전부맞춤. aggregate개선만으로웹채택하지않음. COMPARISON/RESIDUAL_AUDIT에상세.

DEMO_WORD_READINESS:잘하다/감사/즐겁다각공개fit9dev3. 현재GUIDE_SUPPORT상세단계2/0/0. 감사의사전-영상면방향기존미확정. 시연단어후보만검토;단어추가학습/게임병합없음.

기존BP는유지개선대신실제움직임손실로미채택이며결과보존. G0/G1PASS,G2REWORK,G3이후보류. 최종8·DEV/probe추가평가0,사용자/멘토/카메라/외부전송/다운로드/Git/웹운영교체0. 보호477파일불변. 실행프로세스없음. C7다음. 같은학습반복이나경계폭완화로해결하지않음.
''','utf-8')
for p in [U/'DECISION.json',G/'CURRENT_EXECUTION.json',G/'TASK_GRAPH.json',H/'SUBGRAPH.json',O/'STATE.json']:read(p)
for p,h in cfg['protected_pins'].items():assert sha(Path(p))==h,p
write(U/'CLOSEOUT.json',dict(at=at,status='COMPLETE',protected_files=477,all_protected_hashes_match=True,
    synchronized=[str(p)for p in [G/'CURRENT_EXECUTION.json',G/'TASK_GRAPH.json',H/'SUBGRAPH.json',O/'STATE.json',ROOT/'NEW_CHAT_HANDOFF.md',ROOT/'WordsInSilence_PROJECT_CONTEXT.md']],
    next=nxt,training_running=False))
print('C6 complete; aggregate gain, family regression, no deployment. Documents synchronized.')
