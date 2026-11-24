from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,html,shutil
U=Path(__file__).resolve().parent;LAB=U.parents[1];ROOT=LAB.parents[1]
def read(p):return json.loads(p.read_text('utf-8-sig'))
a=read(U/'ANALYSIS.json');v=read(U/'INDEPENDENT_SCORES.json');assert v['status']=='PASS'
stamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
bylabel={r['label']:r for r in a['rows']};believe=bylabel['믿다'];like=bylabel['마음에들다'];star=bylabel['별']
def percent(r,model):return r['models'][model]['target_model_score_percent']
lines=['T3-F / 기존 C — 같은 촬영 12회 비교',f'작성: {stamp}',
 '촬영: 2026-10-19 14:13~14:18 KST / 본인1명·12표현·각1회·12확인기록',
 '범위: 닫힌 세션 ses_067e0348ff32의 전체12시도. 추가/재시도/누락0. 성공만 선별하지 않았다.',
 '', '결과', '기존 C: 1위10/12=83.33%, 상위5개12/12=100.00%',
 'T3-F: 1위12/12=100.00%, 상위5개12/12=100.00%',
 '1위 정답률 +16.67퍼센트포인트. C오답→T3-F정답2, 역회귀0, 둘다정답10.',
 '기준은 사용자가 선택하고 확인한 목표 단어와 100개 후보 중 1위의 일치다. 정확한 수어 수행에 대한 전문가 판정은 아니다.',
 '', '표현별 결과 (C 목표순위 → T3-F 목표순위; 괄호는 각 모델의 목표 점수)',
 *[f"{r['label']}: {r['models']['C']['rank']}위 → {r['models']['T3F']['rank']}위 ({percent(r,'C'):.2f}% → {percent(r,'T3F'):.2f}%)" for r in a['rows']],
 '', '달라진 표현',
 f"마음에 들다: C는 수어1위·강진군수어통역센터2위·목표3위({percent(like,'C'):.2f}%). T3-F는 목표1위({percent(like,'T3F'):.2f}%), 초대권2위(1.47%).",
 f"믿다: C는 사육장1위(95.53%)·목표2위({percent(believe,'C'):.2f}%). T3-F는 목표1위({percent(believe,'T3F'):.2f}%)·사육장2위({believe['models']['T3F']['top2_model_score_percent']:.2f}%).",
 f"별: 두 모델 모두1위. 목표점수 {percent(star,'C'):.2f}% → {percent(star,'T3F'):.2f}%. 이는 추가 정답1개가 아니라 후보 간 구분 점수차가 커진 사례다.",
 '', '왜 달라졌는가 / 무엇까지 말할 수 있는가',
 '두 모델은 동일 촬영의 동일 C 관절입력(좌우배정·미검출·시간·정규화 포함)을 받았다. T3-F의 기존 C 부분은 고정되어 있다. 따라서 이번 두 예측의 순위변화는 동작 앞·중간·뒤를 요약하는 보조 부분의 추가 점수에 의해 생겼다.',
 '이것만으로 어떤 개별 손가락이나 정확히 어느 프레임을 이해해서 맞혔는지, 사용자 동작이 올바른지까지 설명한 것은 아니다.',
 '', '남은 문제',
 f"1. T3-F 오답은 이번12회에서0개. 다만 믿다는 1·2위 모델 점수 차이가 {believe['models']['T3F']['first_second_gap_score_points']:.2f}점으로 이번12개 중 가장 작다. 맞혔지만 사육장과의 혼동 가능성은 남았다. 이 차이에 새 합격선을 만들지 않았다.",
 '2. 믿다의 C 입력 동작구간42프레임 중13프레임은 사용 가능한 왼손 관절이 없었다(오른손42/42, 어깨42/42). 원본 영상을 보지 않았으므로 가림·정상내림·좌우보류 중 무엇인지 확정하지 않았다. 이것이 혼동의 원인이라는 인과도 미확정이다.',
 '3. 멀다·공항·별은 동작구간 내 왼손 사용가능 관절0프레임이어도 두 모델 모두1위였다. 이번 관측에서는 반대손이 안 보이는 것 자체가 실패를 일으키지 않았다. C에서도 이미 맞힌 사례라 이를 T3-F만의 개선으로 세지 않는다.',
 '4. DTW는 기존 공통계산이다. 저장된12개×3항목 모두 계산가능(OK)이지만 정상수어라는 판정은 아니다. 위치·궤적·손모양 차이에는 수행차와2D검출오차 등이 함께 들어가며, 이번 T3-F 점수개선을 DTW 개선으로 볼 수 없다.',
 '5. 본인1명·각표현1회·일상12개 결과이며100개 단어 전체 또는 모든 새 사용자의100% 성능이 아니다. 이전부터 연습·시험해온 사용자이므로 최초사용자 맹검시험도 아니다.',
 '', '검증과 보존',
 '12쌍 모델ID/SHA·동일입력SHA/shape·원본/candidate NPZ SHA·프레임ID/시각 일치 확인. 저장100점수×24예측=2400값의 순위/softmax를 Python math와 별도 NumPy 방식으로 검산했다. 입력채널재생성/모델재추론0.',
 '세션원본 및 두체크포인트102파일 해시불변. 촬영RGB복호화0, 학습0, 튜닝0, 임계값조정0, 후보선택0. 현재 C·T3-F·웹·DTW를 수정하지 않았다.',
 '분석/별도산술검산: Codex 로컬. 별도사람/agent검토는 아님. 세션 purpose=development는 기존UI의 일반 메타값이며 각 결과 data_use는 evaluation_only_never_training_tuning_or_model_selection이다. 이번 자료는 평가 전용이다.',
 '', '다음 확인',
 '이번 비교 결과를 사용자와 확인한다. 기존 C와 T3-F 선택형 웹은 유지하며 자동 운영교체하지 않는다. 추가 원인분석이 필요하면 공개자료에서 믿다/사육장 등 동작단계 구분을 확인하는 계획을 먼저 정한다. 사용자촬영으로 모델·임계값을 맞추지 않는다.']
(U/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n','utf-8')
rows=''.join(f"<tr class='{r['outcome']}'><th>{html.escape(r['label'])}</th><td>{r['models']['C']['rank']}위</td><td>{r['models']['T3F']['rank']}위</td><td>{percent(r,'C'):.2f}%</td><td>{percent(r,'T3F'):.2f}%</td><td>{'개선' if r['outcome']=='gain' else '정답 유지'}</td></tr>" for r in a['rows'])
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>T3-F와 C · 동일 촬영 비교</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f3;color:#20363b;font:16px/1.7 'Malgun Gothic',sans-serif}}main{{max-width:1040px;margin:auto;padding:42px 24px}}h1{{font-size:32px;line-height:1.35;margin:8px 0 18px}}h2{{font-size:22px;margin:0 0 18px}}h3{{font-size:18px}}p{{margin:8px 0 16px}}.eyebrow{{font-size:13px;letter-spacing:.08em;color:#55786c}}section{{background:white;border:1px solid #dce3dd;padding:28px;border-radius:14px;margin:22px 0}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.card{{padding:22px;background:#eef2ee;border-radius:10px}}.card strong{{display:block;font-size:34px}}.track{{height:14px;background:#d8e2db;margin:12px 0;border-radius:7px;overflow:hidden}}.fill{{height:100%;background:#46725c}}.old{{background:#6f8186}}.gain-note{{background:#e5f1e6;padding:14px 18px;border-radius:8px;font-weight:bold}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:11px 10px;border-bottom:1px solid #e1e6e2;text-align:left}}thead{{background:#f0f3ef}}.gain{{background:#eaf4e9}}.note{{font-size:13px;color:#55676b}}.warning{{border-left:5px solid #be8738;background:#fcf7eb;padding:16px 20px}}.scroll{{overflow:auto}}a{{color:#215f4a}}@media(max-width:620px){{main{{padding:24px 12px}}h1{{font-size:26px}}section{{padding:18px}}.cards{{grid-template-columns:1fr}}}}</style>
<main><div class="eyebrow">WORDS IN SILENCE · 저장 결과 비교 · 2026.10.19</div><h1>같은 촬영에서<br>C는 10개, T3-F는 12개 정답</h1><p>본인 1명 · 12개 표현을 각 1회 · 같은 관절 입력 · 100개 후보 중 분류</p>
<section><div class="cards"><div class="card">기존 C<strong>83.33%</strong>10 / 12개가 1위<div class="track"><div class="fill old" style="width:83.3333%"></div></div></div><div class="card">T3-F<strong>100.00%</strong>12 / 12개가 1위<div class="track"><div class="fill" style="width:100%"></div></div></div></div><p class="gain-note">1위 정답률 +16.67%p · 개선 2개 · 퇴보 0개 · 정답 유지 10개</p><p>상위 5개 안에 목표 표현이 포함된 비율은 두 모델 모두 12/12, 100%입니다.</p><p class="note">이 수치는 이번 촬영에서 목표 단어를 맞힌 비율입니다. 수어 동작 자체가 100% 올바르다는 뜻은 아닙니다.</p></section>
<section><h2>어떤 표현이 달라졌나</h2><div class="cards"><div><h3>마음에 들다 · 3위 → 1위</h3><p>C는 ‘수어’를 1위로 골랐습니다. T3-F는 ‘마음에 들다’를 1위로 구분했습니다.</p><p>목표 모델 점수 <b>14.36% → 97.66%</b></p></div><div><h3>믿다 · 2위 → 1위</h3><p>C는 ‘사육장’을 1위로 골랐습니다. T3-F는 ‘믿다’를 1위로 올렸지만 두 후보가 아직 가깝습니다.</p><p>목표 모델 점수 <b>3.34% → 53.66%</b></p></div></div><p class="note">모델 점수는 후보를 나누는 참고 수치이며 동작의 정확도나 보정된 정답 확률이 아닙니다.</p></section>
<section><h2>12개 전체 결과</h2><div class="scroll"><table><thead><tr><th>표현</th><th>C 순위</th><th>T3-F 순위</th><th>C 목표 점수</th><th>T3-F 목표 점수</th><th>변화</th></tr></thead><tbody>{rows}</tbody></table></div><p>‘별’은 두 모델 모두 정답이었으며, 목표 점수는 50.91%에서 93.49%로 높아졌습니다. 추가 정답으로 중복 집계하지 않았습니다.</p></section>
<section><h2>남은 문제: ‘믿다’의 혼동</h2><div class="warning"><b>T3-F: 믿다 53.66% · 사육장 44.15%</b><p>1·2위 점수 차이는 {believe['models']['T3F']['first_second_gap_score_points']:.2f}점으로 이번 12개 중 가장 작습니다. 이번에는 맞혔지만, 반복해도 안정적인지는 아직 모릅니다.</p></div><p>동작 구간 42프레임 중 13프레임은 모델 입력에서 왼손을 사용할 수 없었습니다. 가림인지 정상적으로 내려간 것인지, 이것이 혼동의 원인인지까지는 확정하지 않았습니다.</p><p>반면 <b>멀다·공항·별</b>은 반대손이 관측되지 않아도 C와 T3-F 모두 1위였습니다. 한 손이 안 보인다는 이유만으로 실패하지는 않았습니다.</p></section>
<section><h2>이번 개선이 의미하는 것</h2><p>입력과 기존 C 부분은 그대로입니다. T3-F가 추가한 <b>동작의 앞·중간·뒤를 구분하는 보조 부분</b>이 두 표현의 순위를 바꿨습니다. 새 손 검출기를 적용하거나 이번 촬영으로 학습해서 얻은 결과가 아닙니다.</p><p>DTW는 두 모델에서 공통으로 사용하는 기존 분석입니다. 저장된 12개 촬영의 세 항목 모두 계산 가능한 상태이지만, T3-F 인식 개선이 DTW 교정 기준의 개선을 뜻하지는 않습니다.</p><p><b>본인 1명·표현당 1회</b>의 결과이므로 모든 사람이나 전체 100개 단어에서 100%라고 해석할 수 없습니다. 반복 테스트와 별도 사용자 검증은 남아 있습니다.</p></section>
<section><h2>검산과 자료 보호</h2><p>12쌍의 모델·입력 해시와 촬영 프레임을 확인하고, 저장된 2,400개 점수의 순위를 별도 계산으로 검산했습니다. 원본과 체크포인트 102개 파일이 그대로 유지됐습니다.</p><p><b>학습 0 · 튜닝 0 · 재추론 0 · 모델 선택 0</b><br>모든 촬영을 평가 기록으로만 분석했습니다. 기존 C·T3-F·웹·DTW도 변경하지 않았습니다.</p><p class="note">자료 범위와 수치: <a href="RESULT_KO.txt">상세 설명</a> · <a href="ANALYSIS.json">전체 집계</a> · <a href="INDEPENDENT_SCORES.json">별도 검산</a></p></section></main></html>'''
(U/'RESULT_KO.html').write_text(page,'utf-8')
conclusion=dict(status='COMPLETE',at=stamp,summary=a['results'],delta_top1_pp=a['delta_top1_pp'],
  gains=2,regressions=0,both_correct=10,remaining_top1_errors=[],
  closest_competition=dict(word='믿다',target_score=percent(believe,'T3F'),rival='사육장',
   rival_score=believe['models']['T3F']['top2_model_score_percent'],gap_score_points=believe['models']['T3F']['first_second_gap_score_points']),
  one_hand_invisible_other_preserved=['멀다','공항','별'],source_session=a['session_id'],
  training=0,tuning=0,re_inference=0,model_selection=0,originals_preserved=True,
  caveat='One practised tester, 12 intended labels, one attempt each; no expert form or population validation',
  report=str(U/'RESULT_KO.html'),arithmetic_verified=True)
(U/'CONCLUSION.json').write_text(json.dumps(conclusion,ensure_ascii=False,indent=2),'utf-8')

paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',LAB/'ops/STATE.json']
(U/'before').mkdir(exist_ok=True)
for p in paths:
 dest=U/'before'/p.name
 if not dest.exists():shutil.copy2(p,dest)
next_action='T3-F/C 동일촬영12회 비교 결과를 사용자와 검토한다. 기존 C와 선택형 T3-F 웹 유지. 남은 믿다↔사육장 혼동은 필요하면 공개자료 기반 계획부터 검토하며 사용자촬영을 학습·튜닝·임계값·모델선택에 쓰지 않는다. 추가학습·운영기본교체 없음.'
s=paths[0].read_text('utf-8');marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
entry=f'''\n### 2026-10-19 T3-F/C 사용자 동일촬영12회 비교 완료 — 2개 개선·퇴보0
갱신 {stamp}. ops/{U.name}, 닫힌세션ses_067e0348ff32의12표현각1회/확인12건 전수(재시도·누락0). 동일촬영·동일입력SHA/shape·관절NPZ SHA·프레임/시각·고정모델핀 검증. C10/12=83.33%→T3-F12/12=100.00%, +16.67%p. Top5둘다12/12. 마음에들다 C3위(수어1위)→T3-F1위, 믿다 C2위(사육장1위)→T3-F1위. 정답유지10/개선2/회귀0. 믿다T3-F53.66%vs사육장44.15%로접전남음(모델점수,실행정답률아님). 별1위유지·목표점수50.91→93.49%. 멀다/공항/별의반대손관측0에도둘다정답. 믿다active42중왼손사용불가13프레임은관측사실이며혼동원인확정아님. 기존DTW12×3계산가능,수형정답판정아님. 사용자1명/12회비맹검실용평가로인구전체·100단어일반화아님. 2400저장logits Python math+별도NumPy검산PASS, 원본/모델102파일해시불변. 학습·튜닝·재추론·모델선택0, 웹/기존C/T3-F/DTW불변. RESULT_KO.html/txt, ANALYSIS, INDEPENDENT_SCORES, CONCLUSION. 구현/별도산술검산Codex로컬이며다른agent검토아님.\n\n'''
assert marker in s;s=s.replace(marker,marker+entry,1)
marker9='## 9. 다음 단일 작업\n';assert marker9 in s
s=s.replace(marker9,marker9+'\n### 2026-10-19 T3-F/C 실사용 비교 결과 검토\n'+next_action+'\n\n',1);paths[0].write_text(s,'utf-8')
state=read(paths[2]);state.update(current_stage='WORD100_T3F_WEB_PAIRED_REVIEW_COMPLETE',updated_at=stamp,updated_date_kst='2026-10-19',next_single_action=next_action)
state[U.name]={**conclusion,'next_single_action':next_action}
paths[2].write_text(json.dumps(state,ensure_ascii=False,indent=2),'utf-8')
paths[1].write_text(f'''# WordsInSilence 최신 인수인계

갱신: {stamp}

현재 목표: 사용자가 촬영한12개 표현의 같은입력 T3-F/C 저장결과 비교 완료. 결과 검토 대기.

- 단위: research/front_keyphase_v1/ops/{U.name}
- 범위: runs/ses_067e0348ff32, 2026-10-19 14:13~14:18 KST, 본인1명·12표현각1회·확인기록12. 세션CLOSED, 재시도/누락0. 성공선별없음.
- 결과: 기존C1위10/12=83.33%, T3-F12/12=100%, +16.67%p. 상위5둘다12/12. 개선2/퇴보0/정답유지10.
- 개선: 마음에들다 C3위(수어1위·강진군수어통역센터2위)→T3-F1위; 목표점수14.36→97.66%. 믿다 C2위(사육장1위)→T3-F1위;3.34→53.66%.
- 남은점: 믿다T3-F53.66%/사육장44.15%로9.52점차(12개중최소). active42프레임중왼손사용가능점없음13프레임·오른손42/42, 원인/인과미확정. 별은둘다1위·목표점수50.91→93.49%. 멀다/공항/별반대손관측0이어도둘다1위. 점수는동작일치율/정답확률아님.
- DTW: 기존공통분석·저장12×3항목계산가능. 이것은모델간DTW개선/수어수행정확성검증아님. 원본RGB검토/복호화없음.
- 검산: 모델핀/동일입력SHA·shape/관절파일SHA/프레임ID·시각일치12쌍. 2400저장logits의순위·softmax Python math 및별도NumPy검산PASS. 입력채널재생성·모델재추론0. 원본/체크포인트102파일해시유지. 같은Codex의별도산술코드이며독립agent검토아님.
- 금지준수: 사용자자료평가전용. 학습0/튜닝0/임계값조정0/모델선택0/추가카메라0/외부전송0/Git변경0. 기존C/T3-F/웹/DTW불변. 자료session purpose=development는기존UI일반메타이고각result data_use는evaluation_only_never_training_tuning_or_model_selection.
- 한계: 한명의연습해온사용자·표현별1회. 전체100개/모든새사용자성능/전문가수형평가아님. 이번12개정답을운영교체근거로자동적용하지않음.
- 웹: http://127.0.0.1:8765/practice.html?model=t3f . T3-F선택시같은C입력한번으로T3-F/C비교. C선택은기존C/M0. 기본M0보존. 최근기록서버PID43256,require_escalated(이번턴재시작/프로세스점검없음). 카메라는사용자버튼으로만.
- 고정모델: C seed20261013/epoch77 SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab. T3-F동일seed/epoch69 SHAb0faa0df5055204295c60b78c596a0b490cd63ef14f30225582a85ea39855846.
- 직전공개보류평가: ops/word100_t3f_heldout_20261019의ID17/18총200. sensor평균99.6667→100%,masked99.3333→99.6667%(각+.3333%p). 2/3seed상승기준미충족(2동률1상승),masked3개선1회귀. ID17/18평가사용됨,14~16정상299개미사용유지. 완료평가재실행금지.
- 근거: SCOPE/ANALYSIS/INDEPENDENT_SCORES/CONCLUSION/RESULT_KO.html·txt. 사용자동작자료는읽기전용으로계속보존.

다음 한 작업: {next_action}
''','utf-8')
assert read(paths[2])['current_stage']=='WORD100_T3F_WEB_PAIRED_REVIEW_COMPLETE'
print(json.dumps(dict(status='COMPLETE',report=str(U/'RESULT_KO.html'),docs_updated=3),ensure_ascii=False))
