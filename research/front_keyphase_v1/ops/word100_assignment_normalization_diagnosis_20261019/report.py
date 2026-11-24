"""Publish a local diagnosis from saved evidence. No model execution."""
from pathlib import Path
import json, html
import numpy as np
from collections import Counter
U=Path(__file__).resolve().parent;P=U.parent/'word100_assignment_training_20261018';B=U.parent/'word100_body_boundary_20261017'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(n,x):
 p=U/n;assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
d=read(U/'INPUT_DECOMPOSITION.json');s=read(U/'SAVED_SCORE_DIAGNOSIS.json');f=read(U/'FROZEN_C_TRAIN_SENSITIVITY.json');inv=read(U/'TRAIN_SCALE_INVENTORY.json');prev=read(P/'COMPARISON_INDEPENDENT.json');labels=read(P/'TRAIN_CONTEXT.json')['labels']
summary={}
for a in ['P','Q']:
 es=[e for e in d['details'] if e['arm']==a];fs=[e for e in f['items'] if e['arm']==a];delta=np.array([e['scale_after_direct_margin_delta'] for e in fs]);prob=np.array([100*(e['variants']['new_new']['target_probability']-e['variants']['new_old']['target_probability']) for e in fs]);cl=d['changed_words'][a]
 summary[a]={'changed_unique_train_clips':len({e['clip'] for e in es}),'changed_word_count':len(cl),'words':[labels[y] for y in cl],
  'propagated_L1_fraction':sum(e['l1_propagated_unchanged_frames'] for e in es)/sum(e['l1_total'] for e in es),
  'sensitivity_entries':len(fs),'scale_only_top1_changes':sum(e['variants']['new_old']['prediction']!=e['variants']['new_new']['prediction'] for e in fs),
  'combined_top1_changes':sum(e['variants']['old_old']['prediction']!=e['variants']['new_new']['prediction'] for e in fs),
  'scale_margin_abs_median':float(np.median(abs(delta))),'scale_margin_abs_max':float(max(abs(delta))),
  'target_probability_pp_range':[float(prob.min()),float(prob.max())],
  'C_accuracy_on_affected_words':{v:[{'correct':sum(x['correct'] for x in read(B/f'training_C_{seed}_RESULT.json')['predictions'][v] if x['label_index'] in cl),'total':len(cl)*3} for seed in [20261013,20261014,20261015]] for v in ['masked','sensor']}}
write('DIAGNOSIS_SUMMARY.json',summary)
next_plan={
 'status':'PROPOSED_NOT_AUTHORIZED_TO_TRAIN','production_model':'Keep existing C and all user/mentor files unchanged.',
 'question':'Are finite hand-scale denominator changes a causal contributor to retrained performance changes, separately from assignment/mask changes?',
 'inputs':'Same public TRAIN899/DEV300, labels/ROI/timestamps/masks/schedule/model/optimizer/initial states frozen. No user/mentor/held-out inputs.',
 'design':[
  {'id':'C00','assignment':'old C','finite_scale':'old C','execution':'Reuse original three saved runs'},
  {'id':'Q11','assignment':'corrected Q','finite_scale':'corrected Q','execution':'Reuse original three saved runs'},
  {'id':'R10_diagnostic','assignment':'corrected Q','finite_scale':'old C where both old and Q scales finite','execution':'Three paired seeds, only after approval'},
  {'id':'R01_diagnostic','assignment':'old C','finite_scale':'corrected Q where both old and Q scales finite','execution':'Three paired seeds, only after approval'}],
 'nonfinite_rule':'When either scale is unavailable, use the scale and masks belonging to that observation branch. Never synthesize missing coordinates or set arbitrary scale=1. This keeps scale availability tied to actual observations; finite-denominator effects are isolated.',
 'analysis':'Saved best-CE checkpoint rule unchanged. Report per-seed gains/losses, mean/range, each signer and frozen missingness group; factorial contrasts and interaction. Six new runs total. No post-hoc best-epoch or best-seed replacement. DEV remains development-only.',
 'controls':'Off-diagonal inputs are diagnosis-only; old wrong-assignment scale is not a production target. Validate tensor equality except finite local denominators and preserve all masks/global channels/times before training.',
 'followup_if_supported':'Then propose one deployable scale stabilization candidate: estimate a single reliable scale per hand from the current observed view before whole/context slicing, reuse within both temporal crops; quality and fallback rules fixed using TRAIN only. Exclude unsupported/provisional identity from size calibration only when sufficient confirmed measured support exists; never erase its genuine raw observation. Validate identity support coverage first, because most earlier label-only normal assignments remain provisional.',
 'followup_if_not_supported':'Keep normalization, stop expanding hand-assignment rules for accuracy alone. Investigate persistent public DEV confusions and common training/selection instability without word-specific patches; obtain a fresh public holdout before claims of improvement.',
 'adoption':'Any later production candidate must beat C on the predeclared primary sensor criterion, improve at least 2/3 paired seeds, preserve masked accuracy/Top5 and per-signer/frozen-group results. Diagnostic off-diagonals are not deployed.',
 'causal_limit':'Three seeds and reused DEV cannot establish all-sign vocabulary or new webcam-user generalization.'}
write('NEXT_PLAN.json',next_plan)
text='''손 배정 보완 뒤 인식률이 오르지 않은 이유와 손 크기 정규화 진단
2026-10-19 · 공개 자료만 사용 · 추가 학습 0회 · 기존 C 유지

결론
손 배정의 정확성을 높인 것과 단어 정답률을 높이는 것은 서로 다른 평가다. 이번 보완은 소수 관측을 올바르게 정리했지만, 그 단어들은 기존 C도 이미 대부분 맞혔다. 반면 작은 입력 변화로도 재학습 경로가 달라져 다른 단어에서 새 정답과 새 오답이 생겼다. 손 크기 정규화는 변화가 다른 프레임에 퍼지는 경로임을 확인했지만, 그것이 정확도 하락의 단독 원인이라는 증거는 없다.

1. 기존 결과를 다시 계산한 것이 아니라, 저장된 예측을 짝지어 분석했다
DEV300을 같은 세 초기값으로 비교한 평균. sensor는 상반신 화면만 보고 검출한 공개 영상 조건, masked는 전체 화면 검출 뒤 상반신 좌표만 사용한 조건이다. 실제 웹캠 사용자 정확도가 아니다.
sensor: C 98.67%, P 98.67%(0.00%p), Q 98.22%(−0.44%p).
masked: C 98.44%, P 98.78%(+0.33%p), Q 98.00%(−0.44%p).
sensor 300개×3회=900건의 예측 비교에서 P는 새 정답3/새 오답3, Q는 새 정답3/새 오답7이었다. 900명의 독립 시험이나 900개 원본이 아니다.

2. 손 배정 오류를 고쳐도 단어 정답 수가 늘지 않을 수 있었던 이유
실제 TCN 입력이 달라진 학습 원본은 P 7/899개, Q 17/899개다. 해당 단어는 각각 7종, 14종이다. 기존 C는 이 단어들의 sensor DEV를 매 seed마다 21/21, 42/42로 이미 맞혔다. 올바른 손 좌표가 되는 것과 이미 맞힌 문제에서 정답을 하나 더 늘리는 것은 다르다.
새 sensor 오답 P3/Q7은 모두 기존과 검증 입력이 같았던 영상이다. P의 정답→오답에는 강진군수어통역센터, 백육십이, 동대문이 포함된다. Q는 강진군지부·강진군주민지원센터·상처·경산시수어통역센터·백십팔 등에서도 새 오답이 생겼다. 따라서 새 오답 영상 자체의 손 배정이 나빠져 생긴 직접 오류로 설명할 수 없다.
같은 초기값·순서·학습 조건이라도 학습 입력이 바뀌면 모든 단어가 공유하는 모델 가중치의 갱신이 달라진다. 이것이 이번 결과에서 확인된 경로이며, 그 안에서 배정·마스크·정규화 중 어느 변화가 얼마나 기여했는지는 추가 통제 실험이 필요하다.
미검출 손을 새로 복원한 실험도 아니다. 보류한 관측은 원본에 남기되 좌우 모델 칸에 억지로 채우지 않았다. 겹침 때문에 원래 검출하지 못한 손 모양 정보는 여전히 부족하다.

3. 손 크기 정규화가 실제로 하는 일
사람마다 손이 크거나 카메라 거리가 달라지는 영향을 줄이기 위해, 손목 기준 좌표를 손의 기준 길이로 나눈다. 현재 기준 길이는 사용 구간에서 손목–네 손가락 뿌리까지의 관측 거리들을 모은 중앙값이다. 물리적인 실제 손 크기를 측정한 값은 아니다.
Q의 무산 공개 학습 영상에서는 왼손에 제대로 들어온 관측이 9프레임에서10프레임으로 늘며 기준 길이가62.80px에서57.90px로 바뀌었다(−7.80%). 동일한10px 거리도 정규화 후0.159에서0.173으로 약8.47% 커진다. 이는 새로운 동작이 아니라 같은 좌표를 재는 자의 길이가 바뀐 효과다. 기준이 바뀐 것이 반드시 오류라는 뜻은 아니다.
관측이 그대로인데 크기 기준 때문에 값만 달라진 텐서 프레임은 P591/Q1,061이다. 전체·핵심 구간과 두 관측 조건의 중복을 포함한다. 이 프레임들의 변화량은 모든 채널 절대차 합의 P7.20%/Q4.75%였으므로, 개수가 많다고 영향의 대부분이라고 말할 수 없다. 이 합은 정확도 하락의 기여율도 아니다.

4. 크기 변화만 분리하니 실제 예측은 어떻게 됐나
기존 C 세 개를 고정하고, 변경된 공개 학습 입력45개 조합에 대해 관측 전/후×크기 기준 전/후 네 조건을 비교했다. 총540건 추론이며 학습·DEV 재채점은 없었다.
같은 관측에서 크기 기준만 바꾼135쌍에서 1위 단어 변화는0건이었다. 정답 점수의 상대 우위는 조금 오르기도, 내리기도 했다. 이미 학습한 자료에 대한 민감도 검사이므로 새 사람 성능이나 재학습 효과를 보장하지 않는다. 현재 증거는 '정규화가 입력을 바꾼다'까지이며 '정규화를 고치면 성능이 오른다'까지가 아니다.

5. 다른 원인과 남은 문제
P는 C와 선택된 학습 시점이 세 seed 모두77/76/75회로 같다. 따라서 P의 정체를 체크포인트 선택 탓만으로 돌릴 수도 없다. Q는77/78/78회로 두 seed에서 선택 시점도 달라졌다.
예를 들어 마지막 seed의 Q는75회 sensor296/300, 최종 선택78회293/300이었다. 선택 기준은 정답 개수만이 아니라 두 관측 조건의 평균 교차엔트로피(정답에 주는 확신까지 반영하는 손실)였다. 점수가 좋아 보이는 과거 시점으로 교체하지 않았다. 기존 C 역시 후반 학습에서 정답 수가 변동하므로, 새로운 손 배정만의 문제라고 볼 수 없다.
기존 C 입력에서도 같은 영상의 masked/sensor 기준 길이 차이는 중앙값0.82%,90백분위4.10%,최대60.22%였다. 검출 좌표와 관측 가능한 프레임이 함께 다른 비교라, 최대값을 순수 정규화 오류나 이번 회귀 원인으로 해석하면 안 된다. 빈 손의 크기 계산 불가도 정상 미관측을 포함하며 자동 오류가 아니다.
정확도 차이는 작고 개발 검증을 반복 사용했다. 모든 수어·새 촬영자·웹캠 환경에서 개선되었다는 결론은 낼 수 없다.

6. 다음 계획 — 사용자 확인 전 학습하지 않음
우선 모든 보완을 더 넣지 않고, 손 배정과 손 크기 기준을 분리하는 통제 비교를 제안한다. 기존 C와 Q의 저장 결과는 그대로 재사용한다. 추가할 두 조건은 ①Q의 손 배정+기존 크기 기준 ②기존 손 배정+Q의 크기 기준이며, 각3seed로 총6회다. 양쪽 모두 기준 길이가 유효한 손에서만 분모를 교환하고, 계산 불가 손은 해당 관측의 원래 마스크·미관측을 유지한다. 좌표를 만들거나 임의의 크기를 채우지 않는다.
이 두 조건은 원인 분석용이다. 잘못 배정되던 과거 기준 길이를 실서비스의 정답으로 삼지 않는다. 학습 자료·시간·모델·선택 규칙을 고정하고, 평균뿐 아니라 seed별·촬영자별·관측 유형별 개선과 회귀를 함께 보고한다.
정규화 영향이 실제로 재현될 때에만, 같은 손의 신뢰할 수 있는 관측에서 한 번 정한 크기를 전체·핵심 구간에 같이 쓰는 후보를 별도로 설계한다. 현재 보이는 영상만으로 계산하고, 정상 손과 실제 미관측은 보존한다. 신뢰 관측만 남기다가 대부분 잠정 상태인 정상 손이 제외되지 않도록 TRAIN 지원량부터 확인한다.
정규화가 주원인이 아니라면 이를 계속 바꾸지 않고, 남은 공개 자료 혼동과 학습 후반 안정성을 다음 대상으로 삼는다. 어떤 경우에도 단어별 예외나 사용자 촬영에 맞춘 계수 조정은 하지 않는다.

검증과 범위
별도 산술 코드로180개 손 크기,5,400개 저장 예측 쌍,540개 민감도 출력 및4,545개 입력 핀을 확인했다. 같은 Codex의 별도 계산이며 다른 사람/에이전트의 독립 검토로 표현하지 않는다.
이전의 입력 조합 수 P16/Q36에는 선택 구간 안의 값이 실제로 변하지 않은 P1/Q6도 포함돼 있었다. 이번 실제 비동일 텐서는P15/Q30이다. 변경 프레임611/1,108 및 변경 채널27,684/57,318은 이전 결과와 일치한다. COUNT_RECONCILIATION.json에 구분을 기록했다.
기존 C·웹·DTW·사용자/멘토 자료는 변경하지 않았다. 추가 학습0, 개발 검증 재추론0, 원본 다운로드/외부전송/Git 변경0.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
esc=html.escape
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+esc(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
metric=table(['조건','기존 C','P','Q'],[['화면 안에서만 검출','98.67%','98.67% · 동일','98.22% · −0.44%p'],['전체 검출 후 범위 제한','98.44%','98.78% · +0.33%p','98.00% · −0.44%p']])
causes=table(['확인한 사실','의미'],[['입력이 달라진 학습 원본 P7 / Q17개','899개 중 일부 관측을 정정한 실험'],['정정 대상 단어를 C도 이미 대부분 정답','손 배정 정정이 곧 정답 수 증가를 뜻하지 않음'],['새 오답 영상의 검증 입력은 그대로','재학습 후 분류 기준의 변화가 관여'],['정규화만 변경한135쌍: 1위 변화0건','입력 영향은 있으나 하락 단독 원인 미입증']])
body=''
for block in text.split('\n\n')[2:]:
 lines=block.splitlines();body+='<section><h2>'+esc(lines[0])+'</h2>'+''.join('<p>'+esc(l)+'</p>' for l in lines[1:])+'</section>'
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>손 배정·크기 정규화 원인 분석</title><style>
body{margin:0;background:#eef3f7;color:#172b40;font:17px/1.75 "Malgun Gothic",sans-serif}main{max-width:1080px;margin:45px auto;padding:0 24px 60px}header{background:#173e57;color:white;padding:35px;border-radius:18px}h1{font-size:32px;line-height:1.4;margin:0 0 14px}h2{font-size:23px;line-height:1.5;margin:0 0 18px}section{background:white;padding:30px;border-radius:14px;margin-top:22px}p{margin:12px 0}table{width:100%;border-collapse:collapse;margin:18px 0}th,td{text-align:left;padding:14px;border-bottom:1px solid #dce4ec}th{background:#e9f2f6}a{color:#006c94}.ruler{font-size:24px;font-weight:bold;text-align:center;background:#e9f6f0;border-radius:12px;padding:24px;margin:20px 0}@media print{body{background:white;font-size:11pt}main{margin:0;max-width:none}section{break-inside:avoid;padding:12px 0}header{color:#172b40;background:white;padding:0}table{font-size:10pt}}</style><main>
<header><h1>손 배정은 더 정확해졌는데,<br>왜 단어 인식률은 오르지 않았을까?</h1><p>공개 자료 진단 · 2026-10-19 · 추가 학습 없이 분석 · 기존 C 유지</p></header>
<section><h2>핵심 결과</h2>'''+metric+causes+'''<p>공개 개발 검증300개×3seed 평균입니다. 실제 사용자 정확도나 최종 독립 시험 결과가 아닙니다.</p></section>
<section><h2>크기 정규화: 같은 움직임을 재는 ‘자’가 달라지는 효과</h2><div class="ruler">같은 10px 거리 ÷ 62.80 → 0.159<br>같은 10px 거리 ÷ 57.90 → 0.173</div><p>올바른 손 관측이 하나 추가돼 기준 길이가 바뀌면, 나머지 프레임의 정규화 값도 달라집니다. 이번에는 이런 전달 경로를 확인했으며, 성능 하락의 단독 원인인지는 확인되지 않았습니다.</p></section>'''+body+'''<section><h2>근거 파일</h2><p><a href="SAVED_SCORE_DIAGNOSIS.json">저장 점수·오답 변화</a> · <a href="INPUT_DECOMPOSITION.json">입력 변화 분리</a> · <a href="DIAGNOSIS_SUMMARY.json">진단 집계</a> · <a href="VERIFICATION.json">별도 검산</a> · <a href="NEXT_PLAN.json">다음 통제 실험안</a></p></section></main></html>'''
(U/'RESULT_KO.html').write_text(page,encoding='utf8')
print(json.dumps(summary,ensure_ascii=False))
