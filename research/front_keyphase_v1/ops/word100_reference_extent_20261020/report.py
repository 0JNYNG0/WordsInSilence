from pathlib import Path
from collections import Counter
import json, html
import numpy as np

U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def save(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
assert read(U/'VERIFICATION.json')['status']=='PASS'
s=read(U/'SUMMARY.json');st=read(U/'STRESS.json');lower=read(U/'LOWERING.json')
key=lambda r:(r['clip_id'],r['side'],r['key'],r['condition'])
by={key(r):r for r in st}
old=read(U.parent/'word100_evidence_gate_20261020/STRESS.json')
escaped=[]
for r in old:
    if r['removed_frames'] and r['g1']['status']=='OBSERVED_ENVELOPE':
        a=by[key(r)]
        escaped.append(dict(clip_id=r['clip_id'],label=r['label'],side=r['side'],key=r['key'],condition=r['condition'],
            old_status=r['g1']['status'],previous_research_difference_fell=r['witness_score_decreased'],
            new_status=a['after']['status'],removed_s=a['removed_s'],new_missing_expected_s=a['removed_expected_s']))
low={k:dict(clip_hands_with_rest_missing=sum(r['key']==k and r['rest_missing_s']>0 for r in lower),
    rest_missing_seconds=sum(r['rest_missing_s'] for r in lower if r['key']==k),
    rest_misexpected_seconds=sum(r['rest_expected_s'] for r in lower if r['key']==k),
    inside_missing_seconds=sum(r['inside_missing_s'] for r in lower if r['key']==k),
    inside_missing_expected_seconds=sum(r['inside_missing_expected_s'] for r in lower if r['key']==k)) for k in ('handshape','location','movement')}
diag=dict(previous_escaped=escaped,lowering_per_metric=low,
    warnings=['Counts are metric-hand conditions, not independent people/videos.',
    'Rest and inside-source evidence are offline geometry estimates, not expert semantic ground truth.',
    'Short lost evidence outside reference-projected expected area remains a limitation; no complete-action verdict.'])
save('DIAGNOSTICS.json',diag)
esc_hit=sum(r['new_missing_expected_s']>1e-8 for r in escaped)
esc_fell=[r for r in escaped if r['previous_research_difference_fell']]
example=next(r for r in escaped if r['label']=='순종' and r['key']=='movement' and r['condition']=='peak10')
decision=dict(status='OFFLINE_VALIDATED_PARTIAL_IMPROVEMENT_NOT_DEPLOYED',
    improvements=['Full-query pose alignment and public TRAIN references preserve extent under hand21-only loss.',
        'Known optional hands excluded; offline stable-lowered missing intervals do not overlap expected-observation intervals.'],
    unresolved=['Repeated whole query collapses localization against single-cycle references.',
        'Arm pose-wrist loss changes core in156/591 synthetic controls.',
        'Reference consensus may omit short distinct phases; remaining88word hand roles not verified.'],
    next='Review results. Then freeze an isolated episode-aware reference-alignment comparison that retains the full recording and unresolved times; repetition is a candidate alignment structure, not an error. Test one/two cycles, normal lowering, short missing phases, and missing pose anchors without per-word tuning or new learning.',
    deployment=False,additional_training=0,private_data_read=0,production_models_preserved=True,
    implementation='Codex local',verification='Same Codex separate scalar/path/vote implementation; no separate agent or expert review')
save('DECISION.json',decision)

text=f'''다중 공개 참고를 이용한 비교 동작 범위 보존 검증

결론: 손 관절이 사라져도 비교 범위가 함께 줄어들지 않는 별도 후보를 구현하고 공개 자료로 검증했다. 손 관절만 사라지는 조건에서는 범위가 보존됐다. 다만 두 번 이어서 수행한 동작을 한 번짜리 참고에 맞출 때 범위가 크게 어긋났다. 따라서 현재 후보는 웹에 적용하지 않으며, 반복 처리 보완이 필요하다. C·T3-F, 운영 DTW 수치와 웹은 그대로다.

1. 무엇을 바꿨나
기존에는 관측된 손의 시작과 끝을 따라 비교 범위가 짧아질 수 있었다. 이번 후보는 전체 영상의 어깨·팔꿈치·몸 관절 손목으로 시간 흐름을 맞춘 다음, 같은 단어를 수행한 공개 TRAIN 촬영자 8~9명의 주석 구간을 옮겨 온다. 서로 다른 5명 이상이 지지하는 구간을 참고한다. 어깨는 좌표 정규화에 사용하며, 정렬점은 양쪽 팔꿈치와 몸 관절 손목이다. 손가락 21점은 이 시간 정렬에 넣지 않는다.
예를 들어 손가락이 가려져도 팔 움직임이 남아 있으면 “참고 자료상 아직 비교해야 할 시간인데 손 모양을 못 봤다”고 기록한다. 보이지 않은 좌표를 만들어 채우지 않고, 잘못된 동작이라는 판정도 내리지 않는다. 위치·손가락 모양·궤적 근거는 각각 구분한다.
참고 영상에서 충분히 관측된 구간과 안정적으로 손을 내려놓은 구간을 분리했다. 사용자 손이 안 보인다는 이유만으로 정상 손내림이나 사용하지 않는 손이라고 결정하지 않는다. 기존 공개 역할 기록이 있는 한 손 단어만 반대손을 비교 대상에서 제외한다. 나머지는 손별 진단이며 단어 전체 합격·불합격 판정이 아니다.

2. 사용 자료와 검증
공개 AI Hub 100단어, 1,199영상. TRAIN899영상(촬영자01~09)을 참고로, DEV300영상(11~13)을 점검 대상으로 사용했다. 기존 DTW 관측과 상반신 sensor 관측 두 경로를 비교했다. 같은 영상의 두 관측 경로이며 2,398개의 서로 다른 영상이 아니다. DEV는 이전에도 사용한 개발자료로 새 최종시험이 아니다.
DEV 주석과 원본 전체화면의 손 상태는 결과 검증에만 사용했다. 비교 범위를 정하는 후보에는 DEV 주석·화면 밖 손 정보가 들어가지 않는다. 학습·분류 추론·사용자 및 멘토 자료 접근은 하지 않았다.

3. 개선된 점
1,199영상×2경로의 모든 입력에서 손가락 좌표를 없애도 시간 정렬용 팔 입력이 같았다. 공개 DEV의 전체 손·중앙10%·움직임이 큰10%·20% 구간을 가린 총7,092개 항목·손 조건에서 비교 범위가 줄지 않았고, 가린 만큼 관측 근거가 늘어났다고 표시되는 현상도 없었다. 이 숫자는 591영상·손×4가림조건×3항목이며 독립7,092영상이 아니다.
이전 기준에서 관측 범위가 줄어 공백을 놓친 {len(escaped)}개 항목·손 조건 중 {esc_hit}개는 이번에 참고 구간의 미관측으로 드러났다. 개선된4조건은 같은 공개 ‘순종’ 영상의2항목×2가림조건이며, ‘자석’과 ‘백십육’의4조건은 여전히 놓쳤다. 8조건 중 기존 연구 차이값까지 낮아졌던3조건은2개만 포착했다. 다만 누락된 시간 전체를 다 포착한 것은 아니다. 공개 ‘순종’ 오른손 궤적 예에서 제거한 {example['removed_s']:.3f}초 중 참고 구간 안 {example['new_missing_expected_s']:.3f}초만 추가 미관측으로 표시했다. 이전 연구값을 연결한 비교이며 운영 DTW 점수를 다시 계산한 것이 아니다.
전체 손 제거에서 실제 근거를 지운1,508항목·손 중1,494조건의 추가 미관측을 포착했다. 짧은 이동구간10% 제거는1,503 중1,320조건으로 낮아졌다. 비교 범위가 고정되는 것과 모든 구별 동작이 그 범위에 포함되는 것은 다른 문제다.

4. 정상 동작과 구간 일치
한 손 동작: 기존 공개 참고에서 오른손만 필요한 것으로 확인된3단어·9개DEV영상×2경로에서 반대손을 가려도 결과가 같았다(18영상경로, 54항목). 모든100단어의 한 손/양손 역할을 새로 확인한 것은 아니다.
손내림: 공개sensor300영상의 안정된 손내림 추정 구간에서 손이 안 보이는 시간을 “손이 관측되어야 하는 구간”으로 잘못 겹쳐 잡은 시간은0초였다. 위치 항목 기준 {low['location']['clip_hands_with_rest_missing']}영상·손, 총{low['location']['rest_missing_seconds']:.2f}초의 내림·미관측 근거를 점검했다. 원본 기반 추정 상태와의 대조이며, 모든 수어의 정상 손내림을 판정할 수 있다는 보장은 아니다.
속도: 시간값만0.75배/1.5배로 바꾼1,200대조에서 구간 대응이 유지됐다. 실제 빠른 촬영의 검출 성능을 시험한 것은 아니다.
원래 공개300영상의 주석 동작시간을 덮은 비율의 중앙값은sensor94.35%, 기존관측94.00%였다. 후보가 잡은 시간 중 주석 구간에 속한 비율의 중앙값은각각97.60%,97.92%였다. 이것은 시간 구간이 얼마나 겹치는지 나타내며, 단어 인식률이나 올바른 수행 비율이 아니다. 주석 구간도 수어의 필수 단계 정답표는 아니다.

5. 남은 중요한 문제
반복: 전체 영상을 두 번 이어 붙이고 한 번짜리 참고에 맞추면 sensor의 동작시간 포착 비율 중앙값이94.35%에서48.92%로 낮아졌다. 잡은 시간의 주석 포함 비율도97.60%에서50.00%로 낮아졌다. 정상 반복을 “틀렸다”고 판정하지는 않았지만, 반복 동작을 안정적으로 비교하는 요구는 아직 충족하지 못했다. 이 합성 시험이 모든 단어의 두 번 반복이 언어적으로 허용됨을 의미하지는 않는다.
팔 근거 누락: 손가락뿐 아니라 몸 관절 손목도 함께 가린591대조 중156에서 비교 범위가 달라졌다. 팔꿈치가 남아도 충분하지 않은 경우가 있다. 손가락만의 가림에서 확인한 범위 보존을 모든 미검출로 일반화하면 안 된다.
짧은 동작: 여러 참고의 공통 구간 밖에 있는 짧은 구별 동작은 여전히 놓칠 수 있다. 96개 시점으로 정렬하고 최근접 시점으로 구간을 옮기는 근사도 한계다. 손내림 근거가 없거나 참고들이 동의하지 않는 부분은 미확인으로 남긴다.

6. 다음 보완 계획
전체 영상을 한 번의 참고에 강제로 맞추지 않고, 한 번 수행/반복 수행의 구간 대응 후보를 함께 비교하는 방법을 별도로 검증한다. 손이 사라졌다는 이유만으로 구간을 자르지 않으며, 여러 참고가 동의하지 않는 시간은 삭제하지 않고 비교 근거 부족으로 보존한다. 한 번·반복·정상 손내림·짧은 구별구간 누락·팔 관절 누락을 같은 조건으로 대조한다. 반복 횟수만으로 오류 판정하지 않는다. 특정 단어 예외나 사용자 촬영 기반 조정 없이 공통 규칙으로 진행한다. 추가 학습은 이번 결과를 함께 검토한 뒤 결정한다.

검증: 자체9검사 PASS, 별도 구현으로600영상경로의5,394참고 정렬경로·투표 및7,092고정범위 조건 검산 PASS, 입력·보호·과거결과4,535해시 확인. 구현자와 별도 검산자는 같은Codex이며 다른 에이전트나 수어 전문가 검토가 아니다. C/T3-F6가중치 포함32보호파일 불변. 원본/사용자/멘토자료 수정, 새 학습, 운영 적용 없음.
'''
(U/'RESULT_KO.txt').write_text(text,'utf-8')

def bar(label,value,color='#277e74'):
    return f'<div class="barrow"><span>{label}</span><div class="track"><i style="width:{value}%;background:{color}"></i></div><b>{value:.2f}%</b></div>'
sections=''.join(f'<details><summary>{html.escape(p.splitlines()[0])}</summary><div class="detail">{html.escape(chr(10).join(p.splitlines()[1:]))}</div></details>' for p in text.split('\n\n')[2:])
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>공개 다중 참고 · 비교 범위 검증</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f8;color:#172b40;font:16px/1.7 'Malgun Gothic',sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 26px 70px}}h1{{font-size:36px;line-height:1.35;margin:10px 0 20px}}h2{{font-size:23px;margin:0 0 15px}}p{{margin:8px 0 18px}}.tag{{color:#34645d;font-weight:700;letter-spacing:1px}}.lead{{font-size:21px;max-width:900px}}section,.box{{background:white;border:1px solid #dbe3ea;border-radius:16px;padding:26px;margin:20px 0}}.warn{{background:#fff4df;border-color:#eed398}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{border-left:4px solid #277e74;padding:12px 18px;background:#eef7f5;border-radius:6px}}.card strong{{display:block;font-size:26px;color:#17675c}}.step{{display:flex;gap:10px;align-items:stretch;margin:20px 0}}.step div{{flex:1;background:#eef2f7;padding:15px;border-radius:10px}}.arrow{{align-self:center;color:#57718a}}.small{{font-size:14px;color:#536477}}.barrow{{display:grid;grid-template-columns:150px 1fr 90px;gap:14px;align-items:center;margin:14px 0}}.track{{height:25px;background:#edf0f5;border-radius:5px;overflow:hidden}}.track i{{display:block;height:100%}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:15px 10px;border-bottom:1px solid #e1e7ec;vertical-align:top}}th{{background:#f3f6f9}}td:first-child{{width:20%;font-weight:700}}.bad{{color:#9b5a19;font-weight:700}}.detail{{white-space:pre-wrap;padding:16px 2px}}details{{border-top:1px solid #dbe3ea;padding:14px 0}}summary{{cursor:pointer;font-weight:700}}a{{color:#17675c}}.badge{{display:inline-block;font-size:14px;background:#e5edf5;border-radius:6px;padding:3px 9px}}@media(max-width:700px){{main{{padding:22px 16px}}h1{{font-size:27px}}.lead{{font-size:18px}}.grid{{grid-template-columns:1fr}}.step{{flex-direction:column}}.arrow{{display:none}}section,.box{{padding:20px 16px}}.barrow{{grid-template-columns:115px 1fr 72px;gap:6px;font-size:13px}}td,th{{padding:10px 7px;font-size:14px}}}}</style>
<main><div class="tag">WordsInSilence · 공개 자료 검증 · 2026.10.20</div><h1>손이 사라져도<br>비교할 동작 범위를 남기는 방법</h1>
<p class="lead">여러 사람의 참고 동작으로 비교 구간을 정하면, 손가락이 가려졌다는 이유로 구간을 줄이지 않을 수 있습니다. <strong>다만 반복 동작은 아직 제대로 정렬되지 않아 웹 적용은 보류했습니다.</strong></p>
<div class="grid"><div class="card">검증 대상<strong>100단어 · 300영상</strong>899개 공개 학습 영상은 참고로 활용</div><div class="card">손 관절만 누락<strong>비교 범위 보존</strong>7,092개 항목·손·가림 조건 확인</div><div class="card">운영 모델<strong>C · T3-F 유지</strong>추가 학습·사용자 자료 사용 없음</div></div>
<section><h2>1. 손이 보인 구간만 비교하지 않습니다</h2><div class="step"><div><b>① 전체 영상의 팔 흐름</b><br>어깨 기준 좌표 + 팔꿈치·몸 관절 손목으로 정렬</div><span class="arrow">→</span><div><b>② 여러 사람의 참고 구간</b><br>8~9명 중 5명 이상이 지지하는 동작 시간</div><span class="arrow">→</span><div><b>③ 못 본 시간도 기록</b><br>동작을 봐야 할 시간에 손이 사라지면 비교 근거가 부족한 것으로 표시</div></div><p>보이지 않는 손을 만들어 채우거나, 미검출 자체를 잘못된 수어로 판단하지 않습니다.</p><p class="small">별도 연구 후보입니다. 현재 웹의 DTW 점수나 TCN 분류 결과는 바뀌지 않았습니다.</p></section>
<section><h2>2. 확인한 개선과 보호 조건</h2><table><thead><tr><th>항목</th><th>공개 자료에서 확인한 결과</th></tr></thead><tbody>
<tr><td>끝부분 손 누락</td><td>기존에 공백을 놓친 {len(escaped)}개 항목·손 조건 중 {esc_hit}개에서 미관측을 포착했습니다. 개선된4조건은 같은 ‘순종’ 영상의 항목·가림 조합이고, ‘자석’·‘백십육’의4조건은 여전히 놓쳤습니다.</td></tr>
<tr><td>정상 손내림</td><td>원본으로 추정한 손내림·미관측 시간과 관측을 요구하는 시간의 겹침은 <b>0초</b>였습니다. 이는 확인한 공개 자료 범위의 결과입니다.</td></tr>
<tr><td>한 손 동작</td><td>역할이 확인된 <b>3단어 · 9영상</b>에서 반대손을 가려도 두 관측 경로의 결과가 유지됐습니다.</td></tr>
<tr><td>속도 차이</td><td>시간값을 0.75배·1.5배로 바꿔도 구간 대응이 유지됐습니다. 실제 빠른 촬영의 검출 성능까지 확인한 것은 아닙니다.</td></tr></tbody></table></section>
<section class="warn"><h2>3. 반복은 아직 해결되지 않았습니다</h2><p>한 번짜리 참고 영상에 두 번 수행한 영상을 맞추면, 일부 동작만 잡는 문제가 남았습니다.</p>{bar('원래 공개 영상',94.35290732192214)}{bar('두 번 이어 붙인 영상',48.92460944975893,'#bf8131')}<p><b>위 비율은 주석 동작시간을 얼마나 덮었는지의 중앙값입니다.</b> 단어 인식률이나 사용자 동작의 정답률이 아닙니다.</p><p>반복을 틀린 수어라고 판정하지는 않았지만, 반복 동작도 안정적으로 비교한다는 조건은 아직 통과하지 못했습니다. 각 수어의 반복 허용 여부를 인증한 시험도 아닙니다.</p></section>
<section><h2>4. 함께 남은 문제</h2><p><b>팔 근거도 사라지면 범위가 흔들립니다.</b> 몸 관절 손목까지 가린 591대조 중 156에서 비교 구간이 바뀌었습니다.</p><p><b>짧은 구별 동작을 전부 보장하지 못합니다.</b> 여러 참고가 공통으로 지지하는 구간 밖에 있는 짧은 동작은 놓칠 수 있습니다. 공개 ‘순종’ 예에서는 사라진 {example['removed_s']:.3f}초 중 {example['new_missing_expected_s']:.3f}초를 추가 미관측으로 포착했습니다.</p><p>나머지 88단어의 한 손·양손 역할은 아직 확정하지 않았습니다. 해당 단어는 손별 관측 근거만 보여주는 대상으로 다뤘습니다.</p></section>
<section><h2>5. 다음 단계</h2><p>전체 영상을 한 번의 참고에 강제로 맞추지 않고, <b>한 번 수행과 반복 수행의 구간 후보를 함께 비교</b>하는 방법을 검증합니다. 손이 사라졌다는 이유로 구간을 끊지 않고, 확인하지 못한 시간은 그대로 남깁니다.</p><p>정상 손내림·한 손 동작·반복·짧은 구간 누락을 같은 조건으로 확인합니다. 이번 결과를 함께 검토한 뒤 진행하며, 추가 학습은 시작하지 않습니다.</p></section>
<section><h2>검증 근거와 상세 설명</h2><p class="small">AI Hub TRAIN899 / DEV300 · 2개 관측 경로. 기존 개발자료의 재점검이며 새 최종평가가 아닙니다. 별도 구현 검산: 5,394참고 경로와 7,092가림 조건, 4,535해시 확인 통과. 같은 Codex의 별도 구현 검산이며 독립 에이전트 또는 수어 전문가 검토는 아닙니다.</p><p><a href="RESULT_KO.txt">전체 설명문</a> · <a href="SUMMARY.json">집계</a> · <a href="VERIFICATION.json">별도 검산</a> · <a href="DECISION.json">판정과 다음 계획</a></p>{sections}</section></main></html>'''
(U/'RESULT_KO.html').write_text(page,'utf-8')
print(json.dumps(dict(escaped=len(escaped),revealed=esc_hit,previous_difference_fell=len(esc_fell),example=example,lowering=low),ensure_ascii=False))
