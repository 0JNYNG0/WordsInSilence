from pathlib import Path
import json,html
D=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(name,x):
 p=D/name;assert not p.exists(),p;p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=read(D/'ANALYSIS.json');v=read(D/'INDEPENDENT.json');assert v['status']=='PASS'
plan={
 'status':'PROPOSAL_ONLY_REVIEW_BEFORE_IMPLEMENTATION_AND_TRAINING','training_authorized_now':False,
 'diagnosis':'Frozen auxiliary GRU is substantially more sensitive to the final feature block than to swapping earlier blocks. Word supervision learned without providing corrections on seven existing sensor error events. This is not proof of the sole cause or semantic stage understanding.',
 'next_single_action':'Review this diagnosis, then prepare and verify an isolated multi-position GRU readout before any training. Keep the input and loss unchanged to test one structural change first.',
 'proposed_candidate':{
  'name':'F-G16-M3 (working name, not implemented)',
  'base':'Same seed-matched T3-F entirely frozen, eval including BN. Original C/T3-F files and web untouched.',
  'input':'Same public C280 channels, ROI, assignment, missing masks, real dt, whole/context schedule; no added/removed/interpolated hand or frame.',
  'change':'Read the GRU hidden outputs at all real frames; apply the existing elapsed-time triangular early/middle/late weights separately, concatenate 3x16, then one zero-initialized bias-free Linear48->100 residual. No per-word boundaries. Three temporal bins are NOT claimed to be linguistic stages.',
  'difference_from_T3F':'T3-F pools frozen TCN features in three bins. This proposal reads three summaries of order-dependent GRU hidden states, instead of only its final state.',
  'parameter_budget':{'frozen':47236,'GRU':2400,'new_head':4800,'new_trainable_total':7200,'old_aux_trainable':4000,'increase':3200,'total':54436},
  'causal_limit':'Any future gain would establish the effect of the complete multi-readout candidate. The additional3200 parameters are a confound; do not attribute all gain solely to reading order.',
  'loss':'Keep CE(base+aux,y)+0.1*CE(aux,y), inference base+aux. No new coefficient, temperature, word bonuses or post-hoc scaling.',
  'initialization':'Fresh same-seed GRU initialization as previous runs; new output zero. Never resume a trained auxiliary checkpoint; verify initial combined logits equal frozen T3-F.',
  'proposed_training_after_review':'Only after input/gradient/freeze/padding tests and approval: three seeds20261013/14/15, TRAIN899/DEV300, old batch32/AdamW.001/wd.001/clip1/max80/patience12, fixed final combined two-view DEV CE selection. Stored C/T3-F/oldaux results reused.',
  'preflight':['Expose per-frame hidden state while preserving packing order and hidden-state reset.','Only trailing batch padding excluded; real missingness stays in chronology.','Original masks and dt bitwise unchanged.','Triangular phase weights sum correctly, zero-mass fallback declared and tested.','Zero residual reproduces T3-F; frozen weights/BN no gradients or updates.','Class permutation and synthetic variable-class math checks; no word ID branches.'],
  'comparison':'Primary public sensor mean/range and seed-paired new correct/new wrong; secondary masked, per100class/perperson/missingness groups, CE/top5. Preserve original primary adoption gate: primary mean improvement and at least2/3 seed improvement, plus frozen regression safeguards. Sensitivity alone is not a success criterion.',
  'stop':'If no repeatable improvement, report no gain. No automatic extra epochs, coefficient searches, vocabulary-specific patches or heldout reuse.'},
 'evaluation_preparation':{
  'reuse':'Existing all100 public reference selection and input/visual audit. Confirm early/middle/late observable evidence, ambiguity and genuine gaps. Do not inspect personal recordings.',
  'distinguish':['Same valid movement at different speeds.','Different observable event order with reliable joint evidence.','Real sensor loss or out-of-frame hand; not a labeled wrong sign.','Static/repeated movements where temporal reversal does not establish an error.'],
  'semantic_ground_truth':'The current word labels do not say which submovement is mandatory. Claims about omitted/incorrect linguistic stages require competent sign-language review; uncertain pairs stay unscored. Do not turn arbitrary reversals/crops into wrong-sign labels.',
  'future_order_supervision':'If multi-readout still fails, separately propose chronological relation supervision (which observed interval came earlier), with permutation metadata and timing/position-shortcut controls. It is a separate hypothesis, not added in the first candidate and not equivalent to correct-sign supervision.',
  'future_order_limits':'Contextual TCN features can leak neighboring chronology; repeated/static/occluded segments are ambiguous. Establish valid tasks and metrics before selecting any order-loss weight.'},
 'data_boundary':{'training':899,'development':300,'raw_ID14_to18_used':False,'user_and_mentor_data':'Never read/train/tune/select. User is an end-user tester only.','unseen_words':'Shared algorithm goal, but current head and evidence100classes only; other words need labeled public examples and separate signer-disjoint evaluation.'},
 'expected_benefit':'A testable way to preserve earlier evidence at the output; no numerical accuracy gain is established or promised.',
 'references':[{'title':'Temporal Relational Reasoning in Videos (ECCV2018)','url':'https://arxiv.org/abs/1711.08496','use':'Supports investigating relationships across multiple times; not evidence of benefit on our KSL data.'},
  {'title':'Shuffle and Learn (ECCV2016)','url':'https://arxiv.org/abs/1603.08561','use':'Demonstrates chronological-order pretext learning in other tasks; not proof that shuffled KSL equals an incorrect sign.'}]
}
write('NEXT_PLAN.json',plan)
sensor=[r for r in a['probe_rows'] if r['view']=='sensor'];errs=[r for r in a['base_errors'] if r['view']=='sensor']
assert len(errs)==7 and len({r['clip_id'] for r in errs})==6
assert sum(r['aux_target_minus_old_rival']<=0 for r in errs)==5
def rng(cond):
 x=[r['same_aux_class_pct'] for r in sensor if r['condition']==cond];return f'{min(x):.1f}~{max(x):.1f}%'
def esc(x):return html.escape(str(x))
probe_names={'swap_first_two_thirds':'앞 두 구간 교환 · 마지막 구간 유지','reverse_thirds':'세 구간 뒤집기 · 마지막 구간도 변경','suffix_third':'마지막 1/3 특징만 남김'}
probehtml=''
for cond,label in probe_names.items():
 rr=[r for r in sensor if r['condition']==cond];avg=sum(r['same_aux_class_pct'] for r in rr)/3
 probehtml+=f'<div class="barrow"><div>{label}</div><div class="track"><span style="width:{avg}%"></span></div><strong>{rng(cond)}</strong></div>'
errhtml=''.join(f'<tr><td>{esc(e["target"])}</td><td>{e["seed"]%100}</td><td>{esc(e["base_prediction"])}</td><td>{esc(e["aux_prediction"])}</td><td>{e["aux_target_rank"]}위</td><td>{e["old_rival_gap"]:.3f}</td><td>{e["aux_target_minus_old_rival"]:+.3f}</td></tr>' for e in errs)
histrows=''.join(f'<tr><td>{h["seed"]%100}</td><td>{h["selected_epoch"]} / {h["last_epoch"]}</td><td>{h["selected"]["aux_sensor_top1"]/3:.2f}% → {h["last"]["aux_sensor_top1"]/3:.2f}%</td><td>{h["selected"]["sensor_top1"]} → {h["last"]["sensor_top1"]}</td><td>{h["selected"]["sensor_ce"]:.4f} → {h["last"]["sensor_ce"]:.4f}</td></tr>' for h in a['history'])
text=f'''공개 자료 보조 부분 원인 진단 / 추가 학습 없음

결론: 보조 부분은 배웠지만 기존 오답을 고칠 다른 근거가 부족했다. 끝부분의 특징에 크게 의존하는 경향을 새 진단에서 확인했다. 순서 문제의 유일한 원인이라고 확정하지는 않는다.

1. 검증 범위
공개 TRAIN899·100단어·기존 선택된3seed·2관측조건. 저장 DEV300 결과와 학습 이력 재사용. 새 DEV/보류자료 추론0, 추가학습0, 사용자촬영접근0. 기존 C/T3-F/웹/DTW 변경0.

2. 기존 오답을 못 고친 직접 이유
sensor조건의 남은 고유6영상/3seed합계7오답 사건에서 보조 단독도7건모두오답.5건은 정답보다 기존오답에 더 점수를 더했고,2건은 정답방향이지만 기존격차를 뒤집기에 부족했다. 작은보조점수만의 문제가 아니며 단순비중확대 근거가 없다. 모든100단어를 분석했고 오류단어만을 위한 규칙을 만들지 않았다.

3. 새로 확인한 후반 의존
보조GRU가 읽는 가공된 특징의 앞/중간을 교환하되 끝을 유지: 같은 보조답 {rng('swap_first_two_thirds')}.
끝부분까지바뀌는 역순: {rng('reverse_thirds')}.
끝1/3만남김: {rng('suffix_third')}.
정확도/올바른수어비율이 아니다. 기존TCN특징에는 앞뒤 최대61입력프레임 문맥이 포함되어있어 '앞동작을 완전히안본다'는 뜻도 아니다. masked조건에서도 같은경향.

4. 학습목표와 데이터의 한계
단어정답만 제공하므로 필수동작/순서/생략을 직접 가르친것이 아니다. T3-F는 이미 앞·중간·뒤 요약을 쓰므로 순서를 전혀모르는모델도 아니다. 이번보조는 순서의존GRU 마지막기억하나만분류에사용. 선택된시점의단독보조sensor정확도21/27.67/45.33%. 선택이후더학습해단독분류가늘어도합산검증오차는나빠져, 단순더오래학습을해법으로확정할수없다.

5. 다음 공통계획 (승인 전, 미구현)
먼저 마지막기억하나 대신 GRU가 지나온앞·중간·뒤기억을 각각보존해 읽는 후보 F-G16-M3의 사전검증을 준비한다. 같은C280입력/관측마스크/시간/학습손실 유지. T3-F전체고정. 새학습부품4000→7200으로3200개증가하므로효과를순서만의효과라과장하지않는다. 검토·승인 후3seed만동일조건비교, 기존C/T3-F/직전보조저장결과재사용. 실패시자동재탐색없음.
별도로 공개참고자료에서 관측된단계/가림/반복구간을 구분할평가표를준비한다. 임의역순을틀린수어라단정하지않는다. 필수수어단계판정은수어검토가필요하다. 별도순서학습신호는 다음가설로분리하고 첫실험에동시추가하지않는다.

목표는어휘공통방법이지만검증범위100단어. 모든수어성능향상이나기대정확도%는보장할수없다. 사용자는실사용테스터이며학습·튜닝·후보선택대상이아니다.

검산: 별도NumPy계산360출력, 최대차이{v['max_logit_absolute_error']:.2g}; 저장DEV1800 base/aux/final3종정합; 보호2494파일불변. 같은Codex별도검산이며독립agent/수어전문가검증아님.
'''
(D/'RESULT_KO.txt').write_text(text,encoding='utf8')
css='''*{box-sizing:border-box}body{margin:0;background:#eef3f5;color:#19313f;font:17px/1.75 "Malgun Gothic",sans-serif}main{max-width:1060px;margin:38px auto;padding:0 24px}header,section{background:white;border:1px solid #dce5e9;border-radius:20px;padding:32px;margin:20px 0}header{background:#123547;color:white;border:0}h1{font-size:34px;line-height:1.4;margin:8px 0 18px}h2{font-size:24px;line-height:1.5;margin:0 0 20px}h3{font-size:19px;margin:20px 0 6px}p{margin:10px 0}small,.muted{color:#5d6c76}header small{color:#b5d8e2}.badge{display:inline-block;background:#d9f1e9;color:#14624b;border-radius:20px;padding:4px 14px;font-size:14px;font-weight:bold}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:24px 0}.card{padding:18px;background:#f2f7f8;border-radius:12px}.num{font-size:29px;font-weight:bold;color:#087767}.callout{border-left:5px solid #13a083;background:#edf8f3;padding:16px 20px;margin:18px 0}.warn{border-color:#e3a337;background:#fff7e8}.flow{display:flex;gap:12px;align-items:stretch;flex-wrap:wrap;margin:16px 0}.flow div{flex:1;min-width:190px;border:1px solid #b6d9df;background:#f1f8fa;padding:16px;border-radius:12px}.barrow{display:grid;grid-template-columns:310px 1fr 125px;gap:16px;align-items:center;margin:22px 0;font-size:16px}.track{height:20px;background:#e7edef;border-radius:8px;overflow:hidden}.track span{display:block;height:100%;background:#13a083}.barrow:nth-child(2) .track span{background:#d9903b}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:12px 10px;text-align:left;border-bottom:1px solid #dfe6ea;white-space:normal}th{background:#eef5f7}.steps{counter-reset:s}.step{padding:18px 0;border-top:1px solid #e0e8eb}.step h3:before{counter-increment:s;content:counter(s)". ";color:#06846e}a{color:#006f97;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:bold}li{margin:8px 0}.limit{font-size:15px;background:#f5f7f9;padding:16px;border-radius:10px}@media(max-width:700px){main{margin:14px auto;padding:0 12px}header,section{padding:22px 18px;border-radius:14px}h1{font-size:27px}.cards{grid-template-columns:1fr}.barrow{grid-template-columns:1fr 100px;gap:8px}.barrow>div:first-child{grid-column:1/-1}.flow div{min-width:100%}.scroll table{min-width:640px}body{font-size:16px}}'''
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>보조 부분은 배웠는데, 왜 오답은 그대로일까?</title><style>{css}</style><main>
<header><small>WORDS IN SILENCE · 공개 자료 원인 진단 · 2026.10.19</small><h1>보조 부분은 배웠는데,<br>왜 오답은 그대로일까?</h1><p>새 보조 부분도 같은 어려운 사례에서 헷갈렸고,<br>앞의 변화보다 끝부분의 특징에 크게 반응했습니다.</p><span class="badge">추가 학습 없음 · 기존 C / T3-F 유지</span></header>
<section><h2>이번에 무엇을 확인했나</h2><div class="cards"><div class="card"><div class="num">899개</div>공개 학습 영상 전체<br>100단어 · 3개 고정 모델</div><div class="card"><div class="num">0 / 7건</div>주검증의 기존 오답 중<br>보조 추가로 고친 사건</div><div class="card"><div class="num">0회</div>새 학습 · 새 DEV 채점<br>사용자 촬영 접근</div></div><p>기존 개발 검증 300개의 <b>저장 점수</b>를 분석하고, 공개 학습 영상의 가공된 특징열로 보조 부분만 진단했습니다. 주검증 조건은 공개 영상을 상반신 범위로 제한한 뒤 관절을 다시 검출한 <b>sensor</b> 조건입니다. 실제 웹캠 시험은 아닙니다.</p><p class="limit">7건은 3개 모델을 비교하며 반복 집계한 오답 사건입니다. 서로 다른 영상은 6개입니다. 새 사람 7명을 시험했다는 뜻이 아닙니다.</p></section>
<section><h2>1. 보조 부분이 기존 오답에 정답 근거를 더하지 못했다</h2><p>보조만 따로 보면 선택된 세 모델의 단어 정확도는 <b>21.0% · 27.7% · 45.3%</b>입니다. 학습은 되었지만, 기존 모델이 틀린 7건에서는 보조도 모두 틀렸습니다.</p><div class="flow"><div><b>5건</b><br>정답보다 기존 오답에 점수를 더 보탰습니다.</div><div><b>2건</b><br>정답 방향으로 보탰지만 순위를 뒤집을 만큼 크지는 않았습니다.</div></div><div class="callout">“보조 목소리가 너무 작아서”만의 문제가 아닙니다. 방향이 잘못된 경우도 있어, 보조 점수를 무조건 키우는 방안은 근거가 부족합니다.</div><p>전체 주검증 정확도는 T3-F와 동일한 <b>평균 99.22%</b>입니다. 별도 masked 조건에서는 1개 영상·1회차가 개선돼 평균 <b>+0.11%p</b>였습니다. 주검증 추가 개선은 <b>+0.00%p</b>입니다.</p></section>
<section><h2>2. 앞의 순서보다 끝부분에 더 크게 반응했다</h2><p>동영상을 세 부분으로 생각해 보겠습니다. <b>A → B → C</b>를 읽던 보조 부분에, 이미 추출된 특징의 배열만 바꿔 주었습니다.</p><div class="flow"><div><b>B → A → C</b><br>앞의 순서는 달라짐<br>끝부분은 그대로</div><div><b>C → B → A</b><br>앞의 순서도 달라짐<br>끝부분도 달라짐</div><div><b>C만 읽기</b><br>마지막 1/3의 특징만<br>보조 부분에 제공</div></div><h3>바꾸기 전과 같은 보조 답을 낸 비율</h3>{probehtml}<p>막대는 3개 모델의 평균, 오른쪽은 범위입니다. 같은 답을 낸 비율이지, 정답률이 아닙니다. 첫 조건에서 전체 점수 벡터의 상대 변화도 중앙값 <b>0.95~4.30%</b>로 작았습니다.</p><div class="callout">끝부분을 그대로 두면 앞의 순서를 바꿔도 대부분 답이 유지됩니다. 마지막 기억 하나만 읽는 현재 보조 구조에서 <b>후반 특징 의존</b>이 나타난다는 근거입니다.</div><div class="callout warn"><b>이 해석은 여기까지입니다.</b> TCN 특징에는 앞뒤 문맥이 이미 섞여 있습니다. “앞 동작을 전혀 안 본다”, “뒤집은 영상은 틀린 수어다”라는 뜻은 아닙니다. 이번 실험만으로 모든 오답의 원인을 순서 하나로 확정할 수도 없습니다.</div></section>
<section><h2>3. 무엇을 배워야 하는지가 아직 충분하지 않다</h2><div class="flow"><div><b>현재 주어진 정답</b><br>“이 전체 영상의 단어는 ○○이다.”</div><div><b>직접 주어지지 않은 정답</b><br>“이 단계가 먼저다.”<br>“이 필수 동작이 빠졌다.”</div></div><p>단어 이름만 맞추도록 배우면, 일부 구간의 특징만으로 맞히는 방식도 가능합니다. 기존 T3-F도 앞·중간·뒤 정보를 쓰므로 순서를 전혀 모르는 모델은 아닙니다. 다만 그것만으로 <b>필수 단계와 순서를 올바르게 수행했는지</b>까지 보장하지 못합니다.</p><p>새 보조 부분은 TCN이 정리한 특징만 받습니다. 원래 손 관절이 누락돼 구별 정보가 사라진 구간을 보조 학습만으로 되살릴 수 있다는 증거도 없습니다. 기존 입력 감사에서는 내부 시간 프레임 삭제는 없었지만, 관절 관측 빈틈은 남아 있습니다.</p><details><summary>더 오래 학습하면 해결될까? 저장 이력 확인</summary><div class="scroll"><table><tr><th>seed 끝번호</th><th>선택 / 종료 epoch</th><th>보조 단독 정답률<br>선택 → 종료</th><th>합친 정답 / 300<br>선택 → 종료</th><th>합친 오차 CE<br>낮을수록 좋음</th></tr>{histrows}</table></div><p>보조 자체의 분류는 더 배웠지만, 선택 이후 합친 점수의 오차는 나빠졌습니다. 세 번째는 마지막 정답 수가 늘어도 고정된 선택 기준인 두 관측 조건 평균 CE가 더 나빠 미선택입니다. 뒤늦게 최고 정답 시점으로 바꿔 고르지 않았습니다.</p></details></section>
<section><h2>다음 보완 계획: 앞·중간·뒤의 기억을 함께 읽기</h2><p><b>첫 비교에서는 출력 구조 하나만 바꾸겠습니다.</b> 순서용 새 학습 신호까지 동시에 더하면 무엇이 효과를 냈는지 구분하기 어렵습니다.</p><div class="steps"><div class="step"><h3>여러 시점의 기억을 보존하는 후보 준비</h3><p>현재는 GRU의 마지막 기억 16개 숫자만 읽습니다. 제안 후보는 지나온 기억을 앞·중간·뒤로 각각 요약한 <b>48개 숫자</b>를 읽습니다. “마지막 장만 보고 답하기”에서 “앞·중간·뒤 메모를 함께 보고 답하기”로 바꾸는 것입니다.</p><p>모든 단어에 같은 시간 비율을 씁니다. 아직 언어학적 동작 단계의 정답을 얻은 것은 아닙니다. 기존 T3-F의 TCN 구간 요약과 달리, 새 후보는 <b>순서를 거쳐 누적된 GRU 기억</b>을 구간별로 읽습니다.</p></div>
<div class="step"><h3>학습 전 입력·동결·비교 기준 검증</h3><p>현재 280개 입력, 손 미관측 표시, 실제 시간 간격, 학습 구간과 단어 학습 신호는 그대로 둡니다. 화면 밖 손을 새로 채우거나, 미검출 프레임을 삭제하지 않습니다. 초기 출력이 T3-F와 같은지, 기존 모델의 가중치와 정규화 상태가 고정되는지 확인합니다.</p><p>새 학습 부품은 <b>4,000 → 7,200개</b>로 3,200개 증가합니다. 나중에 좋아지더라도 “순서 개선만의 효과”라고 단정하지 않고 이 용량 차이를 함께 보고합니다.</p></div>
<div class="step"><h3>결과 확인·승인 후 3회만 비교</h3><p>공개 TRAIN 899개 / DEV 300개와 기존 세 seed를 유지합니다. 손실 계수, 학습 일정, 선택 기준도 유지합니다. 기존 C·T3-F·직전 보조 후보의 저장 결과와 비교하고, 전체 100단어·사람별·미관측 조건별 개선과 회귀를 모두 확인합니다.</p><p>주검증 평균 개선과 3회 중 2회 이상 개선 등 기존 채택 기준을 유지합니다. 순서를 섞었을 때 민감해졌다는 이유만으로 채택하지 않습니다. 실패하면 결과를 보고하고 자동 재탐색하지 않습니다.</p></div>
<div class="step"><h3>단계·순서의 정답을 평가할 근거 준비</h3><p>공개 참고 영상에서 관측 가능한 동작 변화, 정상 반복, 화면 밖 손, 실제 가림을 나눠 점검합니다. 수어상 필수 단계·생략 오류는 수어를 아는 사람의 검토가 필요합니다. 불확실한 사례는 오류 정답으로 만들지 않습니다.</p><p>여러 시점 읽기만으로 부족하면, “어느 구간이 먼저였는가”를 별도로 배우는 실험을 다음 가설로 검토합니다. 이때도 촬영 순서를 아는 것과 올바른 수어를 아는 것은 구분하며, 임의 역순 영상을 모두 틀린 수어로 학습시키지 않습니다.</p></div></div><div class="callout">이번에는 진단과 계획까지 완료했습니다. 제안 후보의 구현·추가 학습은 결과를 함께 확인한 다음 진행합니다.</div></section>
<section><h2>기대할 수 있는 것과 아직 모르는 것</h2><p><b>기대 근거:</b> 이번 진단에서 실제로 약했던 앞 구간 정보가 출력까지 남도록 구조를 바꾸는 계획입니다. 특정 단어에만 적용하는 예외 규칙은 아닙니다.</p><p><b>미확인:</b> 이 변경이 인식률을 올릴지, 관절 누락 구간까지 해결할지, 새로운 모든 수어와 사람에게 통할지는 아직 모릅니다. 기대 정확도 숫자를 제시할 근거는 없습니다. 현재 출력과 검증 범위는 100단어이며 다른 단어는 자료·출력 범위·별도 검증이 필요합니다.</p><p><b>유지:</b> C·T3-F·웹·DTW, 사용자 촬영 자료는 그대로입니다. 사용자는 실제 이용자 시험을 위한 테스터이며 학습·튜닝·후보 선택에 쓰지 않습니다.</p></section>
<section><h2>근거와 검산</h2><details><summary>남은 오답 7건의 점수 방향</summary><p>점수 차이는 확률·수어 정답률이 아닌 원시 모델 점수입니다. 마지막 열이 +이면 기존 1위보다 정답 쪽으로 보탠 것입니다.</p><div class="scroll"><table><tr><th>정답</th><th>seed</th><th>T3-F 1위</th><th>보조 1위</th><th>보조 정답 순위</th><th>기존 격차</th><th>보조의 격차 보정</th></tr>{errhtml}</table></div></details>
<p>전체 공개 학습 899개 × 2조건 × 3개 고정 모델 × 6가지 특징열 조건을 진단했습니다. 기존 개발 검증의 최종 점수는 다시 추론하지 않고 저장된 base·aux·final 1,800묶음을 검산했습니다. 별도 NumPy 계산으로 360출력을 확인했고 최대 오차는 {v['max_logit_absolute_error']:.2g}입니다. 보호 파일 2,494개 해시가 유지됐습니다.</p><p>구현과 별도 검산 모두 Codex가 수행했습니다. 독립 에이전트나 수어 전문가가 검증했다는 뜻은 아닙니다.</p><p><a href="ANALYSIS.json">전체 수치</a> · <a href="INDEPENDENT.json">별도 수치 검산</a> · <a href="NEXT_PLAN.json">다음 계획</a> · <a href="SCOPE.json">진단 범위</a> · <a href="RESULT_KO.txt">텍스트 보고</a></p>
<p class="muted">관련 원리: 여러 시점의 관계를 다루는 <a href="https://arxiv.org/abs/1711.08496">Temporal Relational Reasoning in Videos</a>, 시간 순서 학습 과제를 연구한 <a href="https://arxiv.org/abs/1603.08561">Shuffle and Learn</a>. 다른 데이터에서의 연구이며, 이번 프로젝트의 성능 향상을 입증하는 자료는 아닙니다.</p></section></main></html>'''
(D/'RESULT_KO.html').write_text(page,encoding='utf8')
write('CONCLUSION.json',{'status':'DIAGNOSIS_COMPLETE_PLAN_PENDING_USER_REVIEW','sensor_error_events':7,'distinct_sensor_error_clips':6,
 'sensor_aux_correct_on_base_errors':0,'negative_correction_events':5,'positive_insufficient_events':2,
 'swap_early_aux_label_retention_range_pct':[min(r['same_aux_class_pct'] for r in sensor if r['condition']=='swap_first_two_thirds'),max(r['same_aux_class_pct'] for r in sensor if r['condition']=='swap_first_two_thirds')],
 'reversed_blocks_aux_label_retention_range_pct':[min(r['same_aux_class_pct'] for r in sensor if r['condition']=='reverse_thirds'),max(r['same_aux_class_pct'] for r in sensor if r['condition']=='reverse_thirds')],
 'evidence':'Auxiliary feature-stream output has stronger dependence on final block; class-only supervision not semantic order ground truth. Not sole-cause proof.',
 'training_runs':0,'new_DEV_inference':0,'user_data_read':0,'C_T3F_web_DTW_preserved':True,'next':'Review multi-position GRU readout proposal and evaluation ground truth before new implementation/training.'})
print('REPORT AND PLAN WRITTEN')
