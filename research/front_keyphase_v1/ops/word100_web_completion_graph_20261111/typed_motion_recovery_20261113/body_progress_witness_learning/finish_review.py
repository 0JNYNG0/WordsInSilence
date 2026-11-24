"""Close this bounded learning comparison without promoting a failing model."""
from pathlib import Path
import json,datetime,hashlib
U=Path(__file__).resolve().parent;H=U.parent;G=H.parent;O=G.parent;ROOT=O.parents[2]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8');q.replace(p)
at=datetime.datetime.now().astimezone().isoformat()
assert not(U/'DECISION.json').exists()
fit=read(U/'SAVED_FIT_VERIFICATION.json'); ev=read(U/'evaluation/INDEPENDENT_SAVED_VERIFICATION.json'); pr=read(U/'probe_review/INDEPENDENT_SAVED_VERIFICATION.json')
cause=read(U/'evaluation/CAUSE_AUDIT.json')
assert fit['protected_hashes_verified']==ev['protected_files_verified']==pr['protected_hashes_verified']==477
assert ev['new_model_forward_calls_by_verifier']==pr['model_forward_calls_by_verifier']==0
assert not ev['web_changed'] and not pr['web_changed']
protected=read(G/'PROTECTIONS.json');assert len(protected)==477
for p,h in protected.items():assert sha(p)==h,p
training=read(U/'TRAINING_COMPLETE.json'); probe=read(U/'probe_review/RESULTS.json'); dev=read(U/'evaluation/COMPARISON.json')
probe_summary=[dict(seed=r['seed'],by_provenance=r['by_provenance'],equal_scope_weighted=r['equal_scope_weighted'])for r in probe['runs']]
assert [r['by_provenance']['own_terminal_body_hold']['new_correct']for r in probe_summary]==[126,127,118]
assert [r['by_provenance']['original_direct_motion']['new_correct']for r in probe_summary]==[216,215,213]
summary='追加8公開영상 주석을 확인하고 기존16과 합친6명24영상으로 움직임/자세유지 판정부를 고정3초기값으로 학습·검산했다. 학습에 넣지 않은2명8영상의 유지 관측144개는22/28/28→126/127/118로 개선됐지만 실제 움직임240개는228/224/231→216/215/213으로 감소했다. 반복 뷰·정역 포함 관측수이며 새 사용자 정확도가 아니다. 정상·속도 순서연결도 통과하지 못해 웹 미채택이다.'
summary=summary.replace('追加8公開영상','추가8 공개 영상')
nxt='같은 학습이나 구간 이동을 반복하지 않는다. C6에서 FIT02..07만으로 몸 기준 현재 자세와 앞뒤 변화량을 분리한 입력/지도 계약을 검토한다. 이미6명을 본 부모를 복사한 사람제외 검증은 금지하고, 실제 분리된 초기화·정규화에서 정상 움직임과 유지를 함께 확인할 비교를 먼저 설계한다. 기존150 입력의 시간정보와 과거1160 구간판정부를 보존·구분하며, 단어별 예외·DEV/probe 문턱조정 없이 근거가 확인된 변경만 학습한다. G2 통과 전 웹/최종8 보류.'
evidence=[U/'SOURCE_ADMISSION.json',U/'ADDITIONAL_RGB_REVIEW.json',U/'PRETRAIN_VERIFICATION.json',U/'TRAINING_COMPLETE.json',U/'SAVED_FIT_VERIFICATION.json',U/'evaluation/INDEPENDENT_SAVED_VERIFICATION.json',U/'probe_review/INDEPENDENT_SAVED_VERIFICATION.json',U/'evaluation/CAUSE_AUDIT.json']
decision=dict(at=at,status='COMPLETED_HOLD_GAIN_MOVEMENT_TRADEOFF_NOT_ADOPTED',G2_pass=False,G3_pass=None,
    annotation=dict(additional_fit_clips=8,total_fit_clips=24,fit_people=6,original_flags_preserved=True,scope='sparse body motion vs maintained body pose',human_expert_validation=False),
    trained=dict(seeds=[42,43,44],steps=1200,models=3,scope='Separate five-family body-progress readout; not retraining TCN/common100/whole feedback network'),
    FIT=[dict(seed=r['seed'],by_provenance=r['by_provenance'])for r in fit['runs']],
    probe=dict(sources=8,people=2,sequences=48,previously_exposed=True,not_fresh_final=True,rows=probe_summary),
    development=dict(scenarios=82,rows=246,normal=[4,4],slow=[4,4],fast=[6,6],denominator=9,
        conditions=['full_bag_rate_rank','old_local_full_plus_fullbag_half'],all_failed_order_is_unknown_not_wrong_order=True,
        limits='Old selected-extent order and new point-progression order are different measures. Cause audit compares same-point old/new output separately. Preserve original L failed extent decision.'),
    protected_files=477,web_changed=False,no_private=True,no_final8=True,no_old_model_inference=True,
    completion=None,why_not_adopted='Improved stationary-pose recognition does not justify losing real movements or normal action-order support.',
    next=nxt,evidence=[dict(path=str(p),sha256=sha(p))for p in evidence],
    review='Root actual additional8 RGB review and execution; separate Codex input/FIT/DEV/probe arithmetic review. Not independent human/KSL ground truth.',
    first_failures_preserved=True)
write(U/'DECISION.json',decision)
lines=['이번 작업: 움직임과 자세 유지 구분','',
'1. 추가8 공개 영상의 실제 RGB를 확인했다. 기존16과 합쳐6명24영상으로 작은 움직임 판정부를 추가 학습했다. 기존100단어 학습과 C·T3-F는 변경하지 않았다.',
'2. 학습에 넣지 않은 수행자2명의8영상을 추가 비교했다. 멈춘 자세를 계속 움직이는 것으로 착각하던 부분은 크게 줄었다. 그러나 실제 움직임 일부까지 놓쳤다.',
'', '같은 저장 관측에 대한 이전 → 현재(세 초기값을 모두 보고)',
'자세 유지 /144: 22→126, 28→127, 28→118',
'실제 움직임 /240: 228→216, 224→215, 231→213',
'144·240은8영상에서 검토한 시점에3개 화면조건과정역조건을 적용한 관측수다. 144명이나240영상이 아니며, 새로운 사용자 정확도도 아니다.',
'',
'3. 기존82 공개 개발조건에서도 움직임 근거를 순서 연결에 사용해봤다. 두 고정 조건 모두 정상4/9·느림4/9·빠름6/9만 순서를 확인했다. 여기서9는3개 공개 사례×3개 학습 초기값이며, 미확인은 오답 판정이 아니다.',
'4. 저장값을 다시 검산했다. 정상·속도 실패13건은 선택 구간에 몸 관측6~15개가 있지만 움직임 지지가0개였다. 전체 영상에서도 지지0~1개뿐이어서 구간만 옮겨 해결할 문제가 아니다. 같은 시간별 판정부끼리 비교하면 정상·느림3/9→4/9, 빠름6/9→6/9였다. 이번 학습 전부터 있던 약점과 실제 새 회귀를 구분했다. 다만 관절이 관측됐다는 사실이 그 좌표의 정확성을 보장하지는 않는다.',
'5. 웹에는 적용하지 않았다. 현재 개선은 자세 유지 구분에 한정된다. 정상 움직임을 놓치면서 교체하면 사용자가 제대로 해도 피드백을 못 받을 수 있기 때문이다.',
'', '다음 작업',nxt,'',
'보호: 기존C·T3-F/공통100/웹·보호477파일 해시 유지. 사용자·멘토 촬영의 학습/튜닝/선택 없음. 최종8평가 사용 없음. 정답주석은 Codex 영상검토로 독립 한국수어 전문가 정답과 구분한다.',
'HTML은 만들지 않았다. 현재 단계는 전체 성능 개선이나 적용 완료로 주장할 수 없기 때문이다.']
(U/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n','utf-8')
status='BP_COMPLETE_NOT_ADOPTED_NEXT_FIT_GENERALIZATION_AUDIT'
current=read(G/'CURRENT_EXECUTION.json');current.update(at=at,status=status,training_running=False,process_running=False,jobs=[],session_id=None,other_session_id=None,active_unit=str(U),last_completed=str(U/'DECISION.json'),current_command=None,next=nxt);write(G/'CURRENT_EXECUTION.json',current)
s=read(H/'SUBGRAPH.json');s.update(at=at,status=status)
for n in s['nodes']:
    if n['id']=='C5':n.update(status='BODY_HOLD_GAIN_GENERALIZATION_REWORK',evidence=[str(U/'DECISION.json')],task='追加8主석 검토·학습·FIT/probe/DEV검산 완료'.replace('追加8主석','추가8 주석'))
assert not any(n['id']=='C6'for n in s['nodes'])
s['nodes'].append(dict(id='C6',title='유지 개선과 실제 움직임 보존의 일반화 검증',status='NEXT_DESIGN_BASED_ON_SAVED_CAUSE_AUDIT',task=nxt,evidence=[str(U/'evaluation/CAUSE_AUDIT.json')]))
write(H/'SUBGRAPH.json',s)
g=read(G/'TASK_GRAPH.json');g2=next(n for n in g['nodes']if n['id']=='G2');g2['status']='REWORK'
g2['attempts'].append(dict(at=at,experiment_id='BP_BODY_PROGRESS_VS_HOLD',status='REWORK',checks=dict(normal_preserved=False,false_support_reduced=None,not_abstention_only=False,three_seeds=True),limited_gain='maintained-body discrimination improved; this does not meet the combined-order gate',evidence=[dict(path=str(U/'DECISION.json'),sha256=sha(U/'DECISION.json'))],note=summary));write(G/'TASK_GRAPH.json',g)
state=read(O/'STATE.json');state.update(updated_at=at,last_updated=at,next_single_action=nxt,next_single_task=nxt)
state['word100_web_completion_graph_20261111'].update(at=at,status=status,active_unit=str(U),training_running=False,progress=summary,remaining=nxt,latest_evidence=str(U/'DECISION.json'));write(O/'STATE.json',state)
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8-sig');title='### 2026-11-13 추가8 학습·검증 완료 · 유지 개선과 실제 움직임 손실 확인'
assert title not in text
for head,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:
    assert head in text;text=text.replace(head,head+'\n\n'+title+'\n'+body+'\n근거: `'+str(U.relative_to(ROOT)).replace('\\','/')+'/DECISION.json`.\n',1)
p.write_text(text,'utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — 2026-11-13 움직임/유지 비교 완료

{summary}

다음 단일 작업: {nxt}

BP=research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/body_progress_witness_learning. 원래 다른 M과 구분. PY=ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe -X utf8.

추가8FIT03/07 주석은 원래false보존+SOURCE_ADMISSION로 좁은몸진행 범위허용. 기존16 RGB검토계승.144 FIT시퀀스/24영상/6명02..07/3뷰×정역. 고정3seed1200단계 학습exec26206exit0. FIT motion822/822모두유지, ownhold522/522모두맞춤, 새FIT회귀0. 원래남은비대상실패seed44두개보존. TRAINING_COMPLETE 이후재학습/재추론금지.

PROBE08/09는BP학습에는미포함이나과거노출과upstream공통100에포함된자료로freshfinal이아님.8영상48시퀀스,RGB16sheet검토24holdpoint적격/8경계보류.144newforward완료. ownhold22/28/28→126/127/118, motion228/224/231→216/215/213. 양손교대/전완세움양성손실. weighted전체상승만으로채택금지.

DEV82*3새forward246완료. 이전L선택범위/10semantic조건exact. 두조건모두observed order정상4/9·느림4/9·빠름6/9,역순6/9·dynamic역순4/9. 모르는것을오답으로변환하지않음. 옛extent순서와새progress순서는다른지표이므로이전→이후정확도로합치지않음. 같은point부모비교는정상3→4/9,느림3→4/9,빠름6/9유지. 실패13건은옛point도지원0,전체seq0/1. 원래L extent33ms실패도보존. 관측mask존재가좌표정확성정답은아님.

별도검산 FIT/DEV20658/probe2668항목 완료; 이는산술검사수다. 원본477해시일치. 같은Codex별도검토≠독립사람/KSL정답. 처음falseflag중단/평가raw키오류보존,후자READY기반PTS계약으로고친후freeze완료.

G0/G1PASS,G2REWORK,G3이후보류,웹/TCN/100공통/손모양방향변경없음. 최종8·사용자·멘토·카메라·다운로드·외부전송·Git·운영교체금지. 실행프로세스없음. C5완료,C6설계다음. 완료BP/과거R/A/PD/H/N/P/S/B/M/J/K/L재학습/재추론금지. 자세유지판정개선은실제움직임보존을충족해야통합가능하다.
''','utf-8')
for p in[U/'DECISION.json',G/'CURRENT_EXECUTION.json',G/'TASK_GRAPH.json',H/'SUBGRAPH.json',O/'STATE.json']:read(p)
for p,h in protected.items():assert sha(p)==h,p
print('BP bounded unit complete, not adopted; C6 next. All477 protections unchanged.')
