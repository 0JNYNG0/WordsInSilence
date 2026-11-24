"""Package completed public-only diagnostic results; never changes algorithms."""
import html
import json
from datetime import datetime
from pathlib import Path

U = Path(__file__).resolve().parent
def read(name): return json.loads((U/name).read_text('utf-8'))
def save(name, obj):
    with (U/name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
def write(name, text):
    with (U/name).open('x', encoding='utf-8') as f: f.write(text)

check = read('INDEPENDENT_CHECK.json')
assert check['status'] == 'PASS'
scope = read('SCOPE.json')
pred = read('PREDICTIONS.json')
extra = check['broad_location_comparison']['additional12_blind_review']
assert [extra[x] for x in ('missing_reviewed','broad_agree','broad_disagree','undetermined')] == [26,21,1,4]
assert len(scope['records']) == 48
assert len({r['word_code'] for r in scope['records']}) == 16
assert len({r['signer_id'] for r in scope['records']}) == 9

result = {
    'status': 'DIAGNOSTIC_IMPLEMENTED_PARTIAL_VALIDATION_NO_PRODUCTION_ADOPTION',
    'at': datetime.now().astimezone().isoformat(timespec='seconds'),
    'unit': 'word100_hand_visibility_20261017',
    'operator': 'Codex root implementation, execution, visual review; independent scalar check by same operator, not another agent or expert',
    'scope': {'public_train_videos': 48, 'word_types': 16, 'known_train_people': 9,
              'previously_reviewed_videos': 36, 'additional_videos': 12,
              'whole_frames': 6352, 'hand_frame_states': 12704, 'core_frames': 2582,
              'new_training': 0, 'classifier_inference': 0, 'new_detection': 0, 'user_recording_reads': 0,
              'production_changes': 0, 'landmark_recovery': 0},
    'broad_location_comparison': check['broad_location_comparison'],
    'visual_confusion': check['visual_confusion'],
    'overlap_specific_sample': {'visually_overlap_missing_frames': 19, 'overlap_likely': 1, 'inside_missing': 18,
                                'note': '14 correlated frames from one known gap plus 5 additional selected frames. Not population recall.'},
    'safety_review': {'core_confident_runs': 35, 'sampled_hand_frames': 94,
                     'boundary_samples': 89, 'overlap_samples': 5,
                     'clear_interior_active_hand_false_outside_observed': 0,
                     'note': 'Post-output inspection, sampled first/middle/last; not exhaustive per-frame validity.'},
    'findings': [
        'Common causal sidecar distinguishes likely boundary/outside from interior missing using current-view arm geometry and recent usable own-hand track.',
        'Known45 broad location agreement45. Additional26 missing samples:21 agreement,1 boundary-to-inside error,4 undetermined. Existing16 controls remain observed.',
        'Overlap coverage inadequate:19 visually overlapping missing-hand frames,1 overlap-likely and18 generic interior-missing.',
        'Mind06 left55..68 remains interior-missing, not recovered or specifically explained by overlap.',
        'R016 left wrist remains interior while hand extent crosses lower boundary. Wrist point alone cannot represent whole-hand visibility.',
        'No label/signer-specific branches, full-source input, future lookahead, threshold search or coordinate substitution.'
    ],
    'acceptance': {
        'implementation_and_arithmetic': 'PASS',
        'known_cross_cause_safety': 'PASS: known overlap not outside, known outside not overlap',
        'selected_post_output_safety': 'No clear interior active hand labelled outside in94 samples',
        'physical_cause_separation': 'PARTIAL: 1 location disagreement, 4 abstentions, overlap mostly unresolved',
        'automatic_training_or_dtw_exclusion': 'NOT_APPROVED_OR_SUPPORTED',
        'recognition_improvement': 'NOT_MEASURED'
    },
    'next_single_action': '사용자와 결과 확인. 다음 후보는 손목 점이 아닌 손 영역·움직임 불확실성을 함께 쓰는 공통 관측 판단의 고정 계획. 현재 오류에 맞춘 계수 완화 금지. 학습은 별도 결과 검토 후.',
    'limitations': [
        'Public TRAIN examples; no new-person generalization or recognition-accuracy estimate.',
        'Selected and correlated hand-frame samples, one operator visual interpretation, no expert dense labels.',
        'Boundary is partly cut OR outside, not proof of completely invisible or inactive hand.',
        'Inside is a pose-wrist estimate; overlap is suspicion, not certified occlusion cause.',
        'Observed means existing palm5 usable, not correct anatomy/identity or fully visible hand.',
        'Actual public boundary review concerns bottom edge; top/side physical cases and long interruptions unvalidated.',
        'No fabricated landmarks, detector repair, DTW application, model changes or new learning.'
    ],
    'evidence': {n: str(U/n) for n in ('SCOPE.json','visibility.py','SYNTHETIC_TESTS.json','VISUAL_LABELS.json','PREDICTIONS.json','ASSERTION_REVIEW.json','INDEPENDENT_CHECK.json')}
}
save('RESULT.json', result)

text = '''손 미검출 원인 구분 — 구현 및 공개 자료 검증 결과

결론
공통 진단 부품의 구현과 검산을 마쳤다. ‘화면 경계/밖’과 ‘화면 안에서 손이 안 잡힘’을 나누는 데 일부 도움이 되지만, 겹침이 원인인지를 안정적으로 특정하는 단계까지는 도달하지 못했다. 새 학습과 웹 적용은 하지 않았다. 단어 인식률의 개선을 측정한 결과가 아니다.

무엇을 바꿨나
이전에는 손이 없다는 관측만 있었다. 이번에는 그 옆에 이유의 단서를 붙인다.
• 관측됨: 기존에 쓸 수 있다고 저장된 손바닥 5점이 있다.
• 경계·화면 밖 추정: 어깨–팔꿈치–손목이 화면 경계 바깥쪽으로 이어진다. 손 일부가 잘린 경우도 포함한다.
• 화면 안 미관측: 손목은 화면 안쪽으로 추정되지만 손 관절이 잡히지 않는다. 겹침인지 검출 실패인지는 아직 모른다.
• 겹침 의심: 위 조건에 더해 반대 손과 가깝고, 직전 같은 손의 관측과도 연결된다.
• 판단 보류: 팔 위치가 불확실하거나 좌우 배정이 보류되면 억지로 결론을 내리지 않는다.
현재 및 과거의 같은 촬영 범위 관측만 사용한다. 미래 프레임·전체 원본의 숨겨진 관절·단어명·사람 ID를 판단 입력으로 쓰지 않는다. 손 좌표, 좌우 배정, 기존 관측 마스크는 바꾸지 않는다. 정지 손/필수 손 여부를 자동으로 확정하는 기능도 아니다.

검증 자료
AI Hub 공개 TRAIN 16표현·촬영자9명·48영상. 기존 문제와 관련한4표현36영상에, 나머지 표현 목록에서 일정 간격으로 선택한12표현12영상을 추가했다. 추가 영상은 새 사람이 아니라 기존 TRAIN 사람의 다른 표현이다.
전체6,352프레임, 양손 합계12,704개 상태를 저장 관측으로 계산했다. 검출기를 다시 돌리거나 TCN을 다시 채점하지 않았다. 원본 RGB는 시각 판독에만 사용했다.

시각 확인 결과
기존 문제 장면45개: 화면 안/밖 큰 구분45개 일치.
추가 공개 영상의 미검출 장면26개: 큰 구분21개 일치,1개 불일치,4개 판단 보류.
별도로 정상 관측 장면16개: 관측됨16개 유지.
여기서 장면1개는 한 프레임의 한 손이다. 연속된 장면이 포함되어 있으므로 이를 여러 사람의 시험 횟수나 새 사용자 정확도로 해석하면 안 된다. 판단 보류를 정답으로 계산하지 않았다.

구체적인 성과와 남은 문제
1. 내려간 손을 무조건 겹침으로 취급하지 않는다.
   기존 가평군수어통역센터·강진군수어통역센터의 손 내림 구간30프레임은 경계/화면 밖 추정으로 구분됐다.
2. 마음에 들다의 누락을 화면 밖으로 취급하지 않는다.
   TRAIN06 왼손55~68의14프레임은 모두 화면 안 미관측으로 잡혔다. 하지만 ‘겹침’이라고 특정하지 못했고 사라진 관절도 복구하지 않았다.
3. 겹침을 찾아내는 범위가 좁다.
   눈으로 겹침을 확인한 미검출19장면 중1개만 겹침 의심,18개는 일반적인 화면 안 미관측이었다. 손가락끼리 겹쳐도 손목과 반대 손 중심은 멀 수 있다. 긴 누락이나 빠른 이동에서는 직전 손의 연결 근거도 약해진다. 이 실패에 맞춰 거리나 시간 기준을 늘리지 않았다.
4. 경계에 걸친 손1개를 화면 안으로 잘못 분류했다.
   추가 표본 R016(유성구주민지원센터 왼손): 손목은 안쪽에 있지만 손가락 쪽은 아래 경계에 걸쳤다. 손 전체 크기를 보지 않고 손목 한 점으로 위치를 판단한 한계다.
5. 근거가 부족한4장면은 보류했다.
   팔 연결이 불확실한1장면과 검출의 좌우 배정이 보류된3장면이다. 그 손을 없는 손 또는 반대 손으로 임의 처리하지 않았다.

추가 안전 확인
‘경계/밖’ 또는 ‘겹침’이라고 판정한 핵심 구간35개 모두에서 시작·중간·끝94장면을 확인했다. 경계89장면은 내려간 손 또는 경계에 걸친 손, 겹침5장면은 손 접촉/겹침과 일관됐다. 이 표본에서는 화면 안에 온전히 있는 동작 손을 밖이라고 분류한 명백한 사례는 찾지 못했다. 결과를 본 뒤의 안전 점검이며 모든 프레임의 정확성을 보장하지 않는다.

검산
합성 조건23개 통과. 별도 수식 코드로12,704개 상태가 모두 일치했다. 자료192개와 기존 모델/실행 코드 등10개 보호 파일의 해시가 그대로임을 확인했다. 같은 Codex 담당자의 별도 코드 검산이며 다른 에이전트나 전문가의 독립 판독은 아니다.

판단과 다음 방향
이 부품을 관측 상태를 설명하는 진단 기록으로 남기는 것은 타당하다. 다만 이 결과만으로 손을 학습에서 빼거나, DTW가 무시하게 하거나, 인식이 좋아졌다고 판단할 수 없다.
다음 보완은 손목 한 점 대신 마지막 실제 손 관측의 손 영역과 움직임의 불확실성을 함께 살피는 공통 방식이 적합하다. 경계에 걸치거나 근거가 오래되면 화면 안/밖을 강제로 고르지 않고 경계/판단 보류로 남긴다. 겹침도 손목 거리만이 아니라 두 손 영역과 팔 연결을 함께 검토해야 한다. 새 고정 조건과 별도 공개 장면으로 확인해야 하며, 이번 오류에 맞춘 임계값 재탐색은 하지 않는다. 보이지 않는 손의 정확한 모양을 만들어 채우지 않는다.
현재 후보는 겹침 원인 판정의 범위가 좁아 즉시 학습에 넣을 준비가 된 상태가 아니다. 사용자와 이 결과를 확인한 다음 다음 단계를 결정한다. 촬영 간격 보강은 계속 보류한다.

보존 범위
사용자/멘토 촬영 읽기·학습·튜닝·후보 선택0. 새 학습0, TCN 재채점0, 새 MediaPipe 실행0, 운영 모델/웹/DTW 변경0. 공개 원본과 과거 동결 결과는 읽기 전용으로 유지했다.
'''
write('RESULT_KO.txt', text)

state_rows = [
    ('관측됨', '기존 손바닥 관절을 사용할 수 있음'),
    ('경계·화면 밖 추정', '팔이 경계 쪽으로 이어짐. 일부 잘린 손도 포함'),
    ('화면 안 미관측', '손목은 안쪽으로 추정되지만 손이 잡히지 않음'),
    ('겹침 의심', '반대 손과 가깝고 직전 같은 손의 위치와도 연결'),
    ('판단 보류', '팔 연결·좌우 배정 등의 근거가 부족함')]
state_table = ''.join(f'<tr><th>{a}</th><td>{b}</td></tr>' for a,b in state_rows)
gallery = ''.join(f'<details><summary>추가 공개 표본 {i} / 6</summary><img loading="lazy" src="review_{i:02d}.jpg" alt="추가 공개 표본 시각 확인 {i}"></details>' for i in range(1,7))
gallery += ''.join(f'<details><summary>판정 후 안전 점검 {i} / 12</summary><img loading="lazy" src="assertions_{i:02d}.jpg" alt="판정 후 안전 점검 {i}"></details>' for i in range(1,13))
body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손 미검출 원인 구분 · 공개 검증</title>
<style>body{{margin:0;background:#eef2f6;color:#18273a;font:17px/1.7 'Malgun Gothic',sans-serif}}main{{max-width:1040px;margin:auto;padding:42px 24px 70px}}h1{{font-size:34px;line-height:1.3}}h2{{font-size:24px;margin:0 0 16px}}h3{{font-size:19px}}section{{background:white;border:1px solid #d8e1e9;border-radius:16px;padding:28px;margin:22px 0}}p{{margin:12px 0}}.eyebrow{{color:#235d95;font-weight:bold}}.lead{{font-size:21px}}.notice{{padding:16px 20px;background:#fff1d8;border-left:5px solid #c78011}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{background:#f2f6fa;padding:20px;border-radius:12px;text-align:center}}.card strong{{display:block;font-size:36px;color:#164d80}}.bar{{display:flex;height:40px;border-radius:8px;overflow:hidden;margin:16px 0}}.bar span{{display:flex;align-items:center;justify-content:center;color:white;font-weight:bold}}.good{{background:#258474}}.bad{{background:#b75043}}.unknown{{background:#66788c}}table{{border-collapse:collapse;width:100%;margin:14px 0}}th,td{{border-bottom:1px solid #d8e1e9;padding:13px;text-align:left}}th{{vertical-align:top}}a{{color:#175e9b}}details{{border-top:1px solid #d8e1e9;padding:12px 0}}summary{{cursor:pointer;font-weight:bold}}img{{max-width:100%;height:auto}}.small{{font-size:15px;color:#485b6d}}@media(max-width:650px){{.cards{{grid-template-columns:1fr}}main{{padding:20px 12px}}section{{padding:20px}}h1{{font-size:27px}}}}</style>
<main><div class="eyebrow">WordsInSilence · 2026-10-17 · 공개 자료 진단</div><h1>안 보이는 손,<br>화면 밖일까? 겹친 걸까?</h1><p class="lead">공통 구분 부품을 만들고 검증했습니다.<br><b>화면 안·밖 구분은 일부 유효하지만, 겹침 원인 판정은 아직 부족합니다.</b></p>
<div class="notice">새 학습과 웹 적용은 하지 않았습니다. 이번 수치는 <b>손이 보이지 않는 위치의 구분 결과</b>이며 단어 인식률의 개선 수치가 아닙니다.</div>
<section><h2>1. 무엇을 추가했나</h2><p>손이 검출되지 않았을 때, 현재 팔 연결과 직전 실제 손 관측을 보고 다음과 같은 설명을 붙입니다.</p><table>{state_table}</table><p>단어명과 사람 ID는 판단에 사용하지 않습니다. 화면 밖의 숨겨진 손을 원본에서 가져오거나, 미래 프레임을 미리 보거나, 없던 관절을 만들지 않습니다.</p><p class="small">관측됨은 저장된 손바닥 5점을 쓸 수 있다는 뜻입니다. 손 전체가 보인다거나 관절·좌우가 반드시 정확하다는 판정은 아닙니다. 경계/밖은 사용하지 않는 손이라는 뜻도 아닙니다.</p></section>
<section><h2>2. 공개 자료 검증 범위</h2><div class="cards"><div class="card"><strong>48</strong>공개 학습 영상</div><div class="card"><strong>16</strong>서로 다른 표현</div><div class="card"><strong>9</strong>기존 학습 촬영자</div></div><p>기존 문제 4표현·36영상 + 다른 12표현·12영상. 전체 6,352프레임에서 양손 상태 12,704개를 계산했습니다.</p><p>새로 학습한 결과가 아니라 <b>이미 저장된 같은 촬영 범위의 관절 관측을 분석한 결과</b>입니다. 원본 영상은 눈으로 확인할 때만 사용했습니다. 추가 표본의 촬영자는 새 사람이 아닙니다.</p></section>
<section><h2>3. 새로 고른 공개 장면에서 얼마나 맞았나</h2><p>손이 검출되지 않은 <b>26장면</b>의 화면 안·밖 구분:</p><div class="bar" aria-label="일치21, 불일치1, 판단보류4"><span class="good" style="flex:21">일치 21</span><span class="bad" style="flex:1;min-width:26px">1</span><span class="unknown" style="flex:4">보류 4</span></div><table><tr><th>시각 확인과 일치</th><td>21장면</td></tr><tr><th>잘못 구분</th><td>1장면 — 경계에 걸친 손을 화면 안으로 판정</td></tr><tr><th>판단 보류</th><td>4장면 — 팔 위치 또는 좌우 배정 근거 부족</td></tr></table><p>별도로 정상 관측 16장면은 모두 기존 관측을 유지했습니다. 기존 문제 구간 45장면은 안·밖 구분이 모두 일치했습니다.</p><p class="small">1장면 = 한 프레임의 한 손. 연속 장면이 포함된 선택 표본이며, 새 사용자 정확도나 전체 영상의 오류율로 확대할 수 없습니다. 보류는 정답에 포함하지 않았습니다.</p></section>
<section><h2>4. 좋아진 점과 아직 안 되는 점</h2><table><tr><th>내려간 손</th><td>기존 센터 표현의 30프레임을 경계/화면 밖 추정으로 구분했습니다. 손이 없다고 모두 겹침으로 처리하지 않습니다.</td></tr><tr><th>마음에 들다</th><td>문제였던 14프레임을 모두 <b>화면 안 미관측</b>으로 구분했습니다. 다만 겹침이라고 특정하지 못했고, 관절을 복구한 것도 아닙니다.</td></tr><tr><th>겹침 탐지 범위</th><td>눈으로 겹침을 확인한 미검출 19장면 중 <b>1개만 겹침 의심</b>, 18개는 화면 안 미관측에 머물렀습니다.</td></tr></table><p class="notice">즉, “화면 안에 있는데 손이 안 잡혔다”는 구분은 도움이 되지만, “두 손이 겹쳐서 그렇다”까지는 충분히 설명하지 못합니다.</p></section>
<section><h2>5. 왜 한계가 생겼나</h2><h3>손목 한 점과 손 전체는 다릅니다</h3><p>유성구주민지원센터의 R016 장면은 손목이 화면 안쪽에 있어도 손가락은 아래 경계에 걸쳐 있었습니다. 이번 규칙은 손목을 중심으로 보므로 화면 안으로 잘못 구분했습니다.</p><h3>손가락이 겹쳐도 손목끼리는 멀 수 있습니다</h3><p>마음에 들다에서는 손이 실제로 겹쳐도, 손목과 반대 손 중심의 거리가 고정 기준보다 컸습니다. 누락이 길거나 손이 빨리 움직이면 직전 손과의 연결도 불확실해집니다. 해당 단어가 통과하도록 거리·시간 기준을 늘리지는 않았습니다.</p><p>경계/밖 또는 겹침이라고 판단한 35구간의 시작·중간·끝 94장면도 추가로 살폈습니다. 이 표본에서 완전히 화면 안에 있는 동작 손을 밖이라고 분류한 명백한 사례는 없었습니다. 모든 프레임을 전문가가 판정한 결과는 아닙니다.</p></section>
<section><h2>6. 다음 보완 방향</h2><p><b>손목 점 → 손 영역과 움직임의 불확실성까지 함께 확인.</b> 마지막에 실제로 보였던 손의 영역과 팔 연결을 이용하되, 오래된 정보이거나 경계에 걸친 경우는 보류합니다. 겹침도 손목 거리뿐 아니라 두 손 영역의 관계로 확인하는 방향입니다.</p><p>새 조건을 먼저 고정하고 다른 공개 장면으로 검증해야 합니다. 이번 실패 장면에만 맞춘 기준 완화, 특정 단어 예외, 보이지 않는 관절의 임의 복원은 하지 않습니다.</p><p><b>현재 결과만으로 손을 학습에서 제외하거나 DTW가 무시하도록 연결하지 않습니다.</b> 사용자와 결과 확인 후 다음 단계를 결정하며, 촬영 간격 보강은 보류 상태를 유지합니다.</p></section>
<section><h2>검증 기록</h2><p>합성 조건 23개 통과 · 별도 수식 코드로 12,704개 상태 일치 · 자료 192개와 보호 파일 10개 해시 유지.</p><p>사용자·멘토 촬영 읽기 0 · 새 학습 0 · TCN 추론 0 · 새 MediaPipe 실행 0 · 운영 모델/웹/DTW 변경 0.</p><p class="small">구현과 검증 모두 Codex 담당. 독립 검산은 별도 코드로 했으며, 다른 에이전트·사람·수어 전문가의 판독을 뜻하지 않습니다. 실제 영상의 경계 검토는 주로 아래쪽으로, 위/옆 경계와 긴 관측 단절에 대한 일반화는 미검증입니다.</p><p><a href="RESULT_KO.txt">전체 설명</a> · <a href="RESULT.json">결과 원장</a> · <a href="INDEPENDENT_CHECK.json">별도 검산</a> · <a href="SCOPE.json">실행 전 고정 조건</a></p><details><summary>공개 영상 시각 확인 근거 보기</summary><p>노란 선은 관절 검출에 사용한 실제 촬영 범위의 하단입니다. 선 아래 원본은 시각 확인용이며 알고리즘에는 제공하지 않았습니다. R016은 추가 표본 2번 묶음에 있습니다.</p>{gallery}</details></section></main></html>'''
write('RESULT_KO.html', body)
print(json.dumps({'status': result['status'], 'report': str(U/'RESULT_KO.html'), 'bytes': len(body.encode('utf-8'))}, ensure_ascii=False))
