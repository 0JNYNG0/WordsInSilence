from pathlib import Path
import json,html
U=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(name,v):
    p=U/name;assert not p.exists()
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
r=read(U/'RESULT.json');v=read(U/'INDEPENDENT.json');assert v['status']=='PASS'
plan={
 'status':'IMPLEMENTED_PREFLIGHT_PASS_WAIT_USER_REVIEW_BEFORE_TRAINING',
 'training_authorized_this_turn':False,'candidate':'F-G16: frozen T3-F + GRU16 residual',
 'input_policy':'Original C 280-channel input unchanged; all real missing frames and dt retained. Only trailing batch padding excluded. No detector, ROI, hand assignment or normalization changes.',
 'frozen':'Existing T3-F47236 parameters, including phase_weight, and all BN buffers. Default/web C/T3F preserved.',
 'new_trainable':'GRU16 2400 + bias-free residual head1600 =4000. Head starts at zero. No optimizer steps in this unit.',
 'next_single_action':'이번 검증 결과를 사용자와 확인한 뒤, 같은 공개 TRAIN899/DEV300·3seed로 보조 부분만 학습하는 실행 조건을 동결하고 C/T3-F 저장 결과와 비교한다.',
 'seeds':[20261013,20261014,20261015],
 'base_per_seed':'Each matching frozen T3F_<seed>_best.pt, not a newly selected base.',
 'future_training':{'batch_size':32,'epochs_max':80,'early_stopping_patience':12,'optimizer':'AdamW auxiliary4000 only','lr':.001,'weight_decay':.001,'gradient_clip_norm':1.,'selection':'mean DEV cross entropy over masked+sensor, improvement >1e-6','sampling':'inherit C per-clip/epoch masked-sensor and whole-context schedule, same original899 IDs; no new augmentation'},
 'comparison':'Reuse saved C/T3-F scores. Report per-seed and mean/range of normal top1/top5/CE, percentage-point changes, signer/missingness groups and late-epoch variation. Saved checkpoint scores independently recounted, no repeated final re-inference.',
 'adoption_gate_proposal':'Sensor mean top1 improvement with >=2/3 seeds improved, no masked/top5/signer/missingness group regression; all results reported, no silent seed selection. Even a pass is not automatic deployment.',
 'sequence_sensitivity':'Order/partial/repetition transformations are secondary sensitivity checks; not incorrect-sign ground truth and not an adoption target. If needed, fix that check scope before inference.',
 'remaining':['No improved accuracy measured before learning; initialization is intentionally identical.', 'Frozen old weights do not prevent a trained residual from worsening final predictions.', 'GRU may help order discrimination but does not reconstruct missing fingers or guarantee required-step completion.', 'TCN contextual features are noncausal; this is whole-clip classification, not a validated streaming decoder.', 'New capacity and chronology both change; order-only causal claims need a matched-capacity control in a separate proposal.'],
 'data_separation':'No user/mentor footage read or used. DEV reused for development, not a fresh generalization claim. Unused ID14-16 reserved; no access/evaluation this unit.',
 'execution':'Preflight session54321 exit0; verify exit0; no training process running.',
 'verification':'Codex local implementation plus separate NumPy gate/saved-score verification, no independent agent or expert.'}
write('NEXT_PLAN.json',plan)
write('IMPLEMENTATION_READINESS.json',{'status':'PASS_IMPLEMENTATION_ONLY','training_allowed':False,'requires':'User review before additional training; create separately frozen training context afterward','result':'RESULT.json','independent':'INDEPENDENT.json'})
write('SETUP_NOTE.json',{'first_read_attempt':'Read-only metadata inspection exited1 due to Windows cp949 default decoding. No source/result changed. Repeated with explicit UTF-8-sig exited0.', 'preflight_failure':False,'recovery_changed_model_or_conditions':False})
text='''순서 기억 보조 후보 F-G16 — 구현 및 학습 전 검증 완료

기존 T3-F는 그대로 고정하고, 시간순으로 앞 동작의 정보를 이어받는 GRU16 보조 부분을 추가했다. 예를 들면 앞에서 손을 올렸다는 정보를 다음 접촉·이동 구간을 읽을 때도 참고하는 구조다. 사람이 이해하는 동작 이름이나 필수 단계 규칙을 직접 가르친 상태는 아니다.

이번 결과는 안전하게 연결됐는지 확인한 결과다. 학습과 인식률 향상 비교는 아직 하지 않았다. 사용자가 추가 학습은 결과 확인 후라고 지정했기 때문이다.

검증 범위: 공개 100단어, TRAIN899영상, 촬영자9명. masked/sensor 두 관측 방식 × 전체/핵심 주변 두 구간 =3596입력. 기존 3seed 체크포인트에 각각 연결하여 10788출력을 비교했다. 새로운 영상10788개나 학습3회를 뜻하지 않는다.
초기 출력10788/10788이 기존 T3-F와 완전히 일치했다. 저장된 기존 출력1798건도 최대오차0. 모든 기존 가중치/BN와 새 초기 가중치가 검증 전후 불변이었다. 기존47236개 고정, 새학습가능4000개, 총51236개.
3596입력 기준 실제236617프레임을 유지했다. 그중 양손관절이 모두 미관측인6252프레임도 시간과 함께 유지했다. 같은 원영상이 관측방식·구간별로 중복 계수되므로 고유 촬영 프레임 수는 아니다. 배치 길이를 맞추는 빈칸만 GRU에서 제외한다.
별도 NumPy GRU 계산과 PyTorch 결과 최대오차2.22e-16. 패딩·배치순서·영상간 기억 초기화·잘못된 마스크 거절·보조부분 기울기·저장/재읽기 검증 통과. 처음 출력층이0이라 GRU기울기가0인 것은 의도한 초기상태이며, 별도 비학습 진단 복사본에서 출력층을 고정된 비영 값으로 두면 GRU까지 기울기가 전달됨을 확인했다. 최적화/가중치갱신0회.

남은 문제: 학습 전이므로 믿다/사육장 등 기존 혼동은 그대로다. 보조부 학습 후에는 기존부를 동결해도 최종 판단이 나빠질 수 있다. 손이 가려져 관절이 없는 순간의 손 모양을 복원하지 못한다. 단어 분류만 학습하면 필수 동작을 모두 했는지 판정하는 능력도 보장되지 않는다. 실제 의미 구별과 새 사용자 성능은 아직 미검증이다.

다음 제안: 같은899개 학습·300개 개발 검증, 같은3seed로 새4000개만 학습. 기존 C/T3-F 저장 점수와 정확도·상위5위·사람별/미검출별 변화 및 후반 흔들림을 비교한다. 재사용한 개발 검증이며 독립 최종 성능으로 부르지 않는다. ID14~16 미사용 자료는 후속 최종평가용으로 유지한다. 사용자의 확인 전 추가 학습 없음.

보호: 기존C/T3-F·웹·DTW 유지, 사용자/멘토촬영 읽기0·학습0·튜닝0. DEV/보류자료 신규평가0. 별도 코드 검산은 같은 Codex가 수행했으며 독립 에이전트나 수어 전문가 검증은 아니다.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>순서 기억 보조 — 학습 전 검증</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f3f5f9;color:#17253c;font:17px/1.7 'Malgun Gothic',sans-serif}main{max-width:1080px;margin:auto;padding:48px 28px 80px}h1{font-size:38px;line-height:1.3;margin:14px 0 20px}h2{font-size:26px;line-height:1.4;margin:0 0 16px}p{margin:12px 0}.tag{color:#156c64;font-weight:700}.lead{font-size:20px}.notice{background:#fff3d9;border-left:5px solid #dca548;padding:18px 22px;margin:26px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card,section{background:white;border:1px solid #dce2ed;border-radius:18px;padding:26px;margin:20px 0}.card{margin:0}.num{font-size:30px;font-weight:800;color:#2058b5}.muted{color:#526078;font-size:15px}.flow{display:grid;grid-template-columns:1fr 40px 1fr;align-items:center;gap:14px;background:#edf3fc;border-radius:14px;padding:22px;text-align:center}.arrow{font-size:26px;color:#2058b5}.box{background:white;padding:20px;border-radius:10px;border:1px solid #ccdaef}.box.new{background:#e7f7f1;border-color:#8bc7b4}table{width:100%;border-collapse:collapse;font-size:16px}th,td{padding:15px 12px;border-bottom:1px solid #dce2ed;text-align:left;vertical-align:top}th{background:#edf3fc}td:last-child{width:52%}li{margin:10px 0}a{color:#2058b5}details{padding:16px;background:#f3f5f9;border-radius:10px;margin-top:20px}summary{cursor:pointer;font-weight:700}.state{font-weight:700;color:#156c64}@media(max-width:700px){main{padding:24px 16px}h1{font-size:29px}.grid{grid-template-columns:1fr}.flow{grid-template-columns:1fr}.arrow{transform:rotate(90deg)}section,.card{padding:20px}table{font-size:14px}th,td{padding:10px 6px}}@media print{body{background:white}main{padding:0}section{break-inside:avoid}}</style>
<main><div class="tag">WordsInSilence · 2026.10.19 · 별도 연구 후보 F-G16</div><h1>앞에서 한 동작을<br>차례대로 기억하는 보조 부분</h1><p class="lead">구현과 학습 전 검증을 마쳤습니다.<br>기존 T3-F의 판단을 보존한 채, 다음 학습을 준비한 단계입니다.</p>
<div class="notice"><strong>이번에는 추가 학습을 하지 않았습니다.</strong><br>따라서 인식률 개선 수치는 아직 없습니다. 결과를 함께 확인한 뒤 학습한다는 요청을 따랐습니다.</div>
<div class="grid"><div class="card"><div class="num">899개</div>공개 학습 영상<br><span class="muted">100단어 · 촬영자9명</span></div><div class="card"><div class="num">10,788 / 10,788</div>초기 출력 일치<br><span class="muted">입력4종 × 기존3조건</span></div><div class="card"><div class="num">4,000개</div>새 보조부 변수<br><span class="muted">기존47,236개는 고정</span></div></div>
<section><h2>어떻게 달라졌나요?</h2><p>기존 모델이 읽은 각 순간의 특징을 작은 기억장치가 순서대로 받아갑니다. 다음 장면을 읽을 때 앞 장면의 정보도 함께 참고하도록 만드는 구조입니다.</p><div class="flow"><div class="box"><b>기존 T3-F</b><br>관절·시간 정보를 읽고<br>원래 단어 점수 계산<br><span class="state">배운 내용 전체 고정</span></div><div class="arrow">→</div><div class="box new"><b>새 순서 기억 보조</b><br>앞 특징 → 다음 특징 → 마지막 특징<br>기억한 흐름으로 보정 점수 계산<br><span class="state">지금은 보정 점수0으로 시작</span></div></div><p><strong>최종 점수 = 기존 점수 + 새 보조 점수</strong></p><p>예를 들어 앞에서 손을 올린 정보를 뒤의 접촉·이동 장면을 볼 때 참고할 수 있게 합니다. 아직 ‘올리기·접촉’이라는 동작 이름이나 정답 규칙을 배웠다는 뜻은 아닙니다.</p></section>
<section><h2>무엇을 확인했나요?</h2><table><thead><tr><th>확인 항목</th><th>결과</th></tr></thead><tbody>
<tr><td>붙이자마자 기존 결과가 바뀌는가?</td><td>10,788출력 모두 기존 T3-F와 완전히 일치</td></tr>
<tr><td>기존 지식이 함께 바뀌는가?</td><td>기존 가중치·정규화 통계 그대로 유지</td></tr>
<tr><td>손이 안 보인 순간을 삭제하는가?</td><td>양손 미관측6,252프레임도 시간과 함께 유지</td></tr>
<tr><td>영상 길이 맞춤용 빈칸까지 기억하는가?</td><td>빈칸은 제외하고 실제 마지막 순간의 기억 사용</td></tr>
<tr><td>앞 영상의 기억이 다음 영상에 섞이는가?</td><td>매 영상 새로 시작, 배치 순서를 바꿔도 유지</td></tr>
<tr><td>새 부분에만 학습 신호가 가는가?</td><td>기울기 전달 검증 통과 · 실제 학습 갱신0회</td></tr>
</tbody></table><details><summary>검증 숫자의 의미와 근거</summary><p>899영상 × 관측2종(masked/sensor) × 구간2종(전체/핵심 주변) = 3,596입력입니다. 이를 기존3개 seed의 T3-F에 각각 연결해 10,788출력을 확인했습니다. 신규영상 수나 학습 횟수가 아닙니다.</p><p>입력변형 합계236,617프레임 중 양손 미관측6,252프레임을 유지했습니다. 동일 원영상이 관측·구간별로 중복 계수되며, 미관측을 실제 손 모양으로 복원한 것은 아닙니다.</p><p>별도 NumPy 계산으로 GRU 수식을 검산한 최대오차는2.22×10⁻¹⁶입니다. 기존 저장 출력1,798건과의 최대오차는0입니다. 같은 Codex의 별도 코드 검산이며 독립 에이전트·수어 전문가 검증은 아닙니다.</p><p><a href="RESULT.json">구현 검증</a> · <a href="INDEPENDENT.json">별도 계산 검산</a> · <a href="NEXT_PLAN.json">다음 계획</a> · <a href="RESULT_KO.txt">텍스트 보고서</a></p></details></section>
<section><h2>아직 해결됐다고 말할 수 없는 점</h2><ul><li><strong>‘믿다–사육장’ 같은 혼동:</strong> 초기 결과가 같으므로 현재 혼동도 그대로입니다. 보조 부분을 학습한 뒤 실제 개선 여부를 비교해야 합니다.</li><li><strong>가려진 손의 모양:</strong> 시간을 남겨 두어도 보이지 않은 손가락 정보가 생기지는 않습니다.</li><li><strong>동작을 빠짐없이 했는지:</strong> 단어 이름을 맞히는 학습만으로 모든 필수 동작의 수행 여부까지 보장할 수 없습니다.</li><li><strong>기존 정답의 회귀:</strong> 기존 가중치를 고정해도 보조 점수가 잘못 더해지면 최종 판단은 나빠질 수 있습니다.</li></ul></section>
<section><h2>다음 비교 계획</h2><ol><li>같은 공개 학습899영상·개발 검증300영상과 같은3개 시작 조건을 사용합니다.</li><li>기존 T3-F 전체는 고정하고 새4,000개 변수만 학습합니다.</li><li>기존 C·T3-F의 저장 점수와 정확도, 상위5위, 사람별·미검출별 변화, 학습 후반 흔들림을 비교합니다.</li><li>특정 단어만 좋아지고 다른 단어가 나빠지지 않는지 함께 보고합니다. 개선이 확인돼도 자동으로 웹 모델을 교체하지 않습니다.</li></ol><p><strong>사용자·멘토 촬영은 이번에 읽지 않았고, 이후 학습·튜닝·모델 선택에도 사용하지 않습니다.</strong> 기존 C·T3-F·웹·DTW는 그대로입니다.</p><p>반복 사용한 개발 검증 결과를 새 사용자 성능으로 표현하지 않습니다. 미사용 촬영자ID14~16은 최종 검증용으로 남겨 둡니다.</p></section></main></html>'''
(U/'RESULT_KO.html').write_text(page,encoding='utf8')
print('Report and next-plan written; no training authorization')
