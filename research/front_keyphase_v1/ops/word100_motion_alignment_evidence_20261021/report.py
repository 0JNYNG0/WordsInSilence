import html,re
from pathlib import Path
import study as d
U=d.U
v=d.read(U/'VERIFY.json');a=d.read(U/'ANALYSIS.json');assert v['status']=='PASS'
text='''손 관측·움직임·누락 시간 분리 검증
2026-10-21 · 연구용 구현/검증 완료 · 자동 오류 판정 미채택 · 추가 학습 없음

결론
손이 보인다는 사실과 실제 변화의 근거를 구분하고, 잠깐 보이지 않은 시간을 기록에 남기는 기능은 구현·검산했다. 하지만 여러 동작 순간을 한 장면에 맞추었다는 경고만으로 잘못된 수어 동작을 가려낼 수는 없었다. 정상 속도 차이에도 경고가 많아서 점수 차감·오답 판정·운영 적용에는 쓰지 않는다. 이번 작업으로 인식률이나 시간 정렬 정확도가 올라갔다고 말할 수는 없다.

무엇을 바꿨나
손이 보임 / 손 위치 변화 근거 / 손가락 각도 변화 근거 / 확인된 안정 손내림 / 비교 근거 부족을 시간표에 따로 표시했다. 서로 겹칠 수 있는 정보이다. 예를 들어 손 위치는 보이지만 일부 손가락은 가려져서 비교하지 못할 수 있다. 좌표·좌우 배정·각도 자체는 지우거나 수정하지 않았다.
기존 검출과 현재 이미지의 재확인 결과가 어느 정도 다를 수 있다는 점을 고려해, 그 차이 범위를 넘는 변화가 있는지를 표시했다. 이것은 기하학적 변화 근거이지 수어의 의미나 통계적으로 보정된 신뢰도는 아니다. 변화가 안 잡혔다고 반드시 정지한 것은 아니며, 손이 안 보인다고 손을 내린 것으로 처리하지 않는다. 안정 손내림에는 기존의 긍정적 관측 근거가 필요하다.

검증 범위
공개 TRAIN 자료 수어·후원·멀다 3개 단어, 각 9명, 총 27영상. 한 영상을 다른 8명의 같은 단어와 비교했다. 기존 세 조건의 216쌍씩 총 648결과를 재사용해 진단했다. 100단어 전체의 새 인식 성능 검증은 아니다. 기존 80개 정상·교차 대조는 앞선 입력 보존 검증 근거이며, 이번에 새로 80개를 정렬한 것이 아니다.

확인된 개선
1. 앞선 비교에서 새로 표시됐던 25개 쌍-손-프레임은 손이 보인다는 이유만으로 활동 중처럼 해석될 여지가 있었다. 이번 검사에서는 25개 모두 위치/각도 변화 근거가 없다고 구분했다. 서로 다른 참고가 같은 장면을 중복해서 포함할 수 있으므로 25개 영상이나 25개의 확정 오류라는 뜻은 아니다. 낮게 멈춰 있는 손이 엄격한 손내림 조건을 통과하지 못해도, 무조건 움직이는 손으로 취급하지 않는다.
2. 중간 3프레임의 손 관측을 지운 27개 조건에서 81프레임×15각도=1,215개 누락 항목을 모두 표시했다. 참고와 다시 연결한 경로가 바뀌어도 이 시간표는 사라지지 않았다. 손 전체 누락 54조건 중 필요한 손 45조건은 해당 동작 구간의 누락을 표시했고, 원래 쓰지 않는 왼손 9조건은 필수 비교 대상으로 바꾸지 않았다. 참고측 비교 시간 범위도 전부 유지했다.
3. 기존 좌표·좌우·정상 교차·각도 상태를 변경하지 않았다. 확실한 손내림 시간은 비교 근거 부족으로 표시하지 않는다. 다만 손내림을 확인하지 못한 구간은 모름으로 남고, 경고가 과한 문제까지 해결된 것은 아니다.

남은 문제
여러 시점이 한 장면으로 연결되는 현상은 속도 차이를 맞추는 DTW에서 자연스럽게 생길 수도 있다. 이번 경고 기준은 이를 잘못된 압축과 충분히 구분하지 못했다.
8명 중 5명 이상 참고에서 같은 손·각도·시간에 경고가 겹친 영상은 원래 영상 14/27, 길이 0.75배로 재표본한 영상 19/27, 길이 1.5배 영상 27/27, 중간 0.5초 정지 영상 19/27이었다. 정상 변형에서 이렇게 많이 반응하므로 이 경고를 오류로 쓰면 안 된다.
2회와 3회 이어붙임은 각각 14/27로 원래 영상과 같았다. 이는 영상과 참고, 연결 경로까지 함께 복사한 보존 검사이다. 반복 횟수를 스스로 맞췄다는 증거나 서로 다른 반복 횟수를 올바르게 정렬했다는 증거는 아니다.
공개 후원 REAL04의 61번째 프레임처럼 굽힌 손을 펴진 참고에 연결했던 문제는 경고가 남았다. 하지만 REAL06의 28번째 프레임 문제는 한 참고에서는 경고했어도 5명 합의 기준에는 잡히지 않았다. 다수 참고가 있다고 모든 잘못된 연결이 검출되는 것도 아니다. 원래 관측 근거가 없는 프레임의 경고는 그 프레임 자체에서 움직임을 봤다는 뜻이 아니라, 그 프레임이 속한 연결 구간의 경고이다.

중요한 해석
누락 표시를 붙였다고 기존 DTW 차이 점수나 관측 비율이 자동으로 보정된 것은 아니다. 이전에 짧은 누락 후 주변 장면으로 다시 맞추면서 비율이 올라가던 계산은 그대로 두었다. 이번에는 그와 별개로 빠진 시간을 숨기지 않도록 만든 것이다. 잘 맞음 판정이나 통과·탈락 기준과의 연결은 아직 미적용이다.
이 실험은 관절 입력을 인위적으로 지우거나 시간을 재표본한 시험이다. 사용자가 실제 수어의 필수 동작을 생략한 것인지, 카메라에서 잠깐 가린 것인지를 판별한 실험은 아니다. 공개 주석의 동작 구간을 진단에 이용했으므로 실시간 웹에서 그 구간을 찾는 문제도 별도로 남는다.

다음 제안
먼저 이번 결과를 함께 확인한다. 관측/누락 시간표는 진단 기능으로 보존하되, 현재 경고로 감점하거나 새 학습을 시작하지 않는다. 다음 검증은 하나의 공개 원본에서 시간만 늘이거나 줄인 쌍을 만들어, 실제 대응 시점을 아는 상태로 정상적인 시간 정렬을 확인하는 것이다. 그 뒤 확실히 관측된 모양 변화의 전후 순서가 유지되는지 검사하여, 정상 속도 차이와 짧은 단계가 사라진 연결을 구분한다. 단순히 경고 문턱을 높여 결과를 좋게 만들지 않는다. 특정 단어별 규칙도 추가하지 않는다.

검증 및 보존
Codex가 구현하고 별도 코드로 재계산했다. 코사인 기반 각도 계산·쌍별 구간 겹침 검사로 98,970개 프레임-손-각도 입력과 2,376개 저장 비교 결과를 확인했다. 648개 원래 비교, 648개 누락 비교, 1,080개 정상 변형 비교이며, 원본 재촬영이나 새 분류 학습 횟수가 아니다. 코드·입력·기존 결과 핀과 모델/운영 보호 핀 32개를 확인했다. 새 속도 변형의 저장 DTW 경로는 단조성·범위·진단 값을 검산했으며 별도 알고리즘으로 다시 최적화하지는 않았다. 동일 담당자의 독립 수식 검산이고 별도 수어 전문가 정답 검증은 아니다.
기존 C·T3-F와 웹은 유지했다. 사용자·멘토 촬영은 접근·학습·튜닝·모델 선택에 사용하지 않았다. 추가 학습·분류 추론·운영 교체는 0회이다.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
style='''body{margin:0;background:#f1f5f9;color:#15263b;font:17px/1.8 "Malgun Gothic",sans-serif}main{max-width:1120px;margin:auto;padding:40px 24px 80px}h1{font-size:36px;line-height:1.35;margin:16px 0}h2{font-size:24px;margin:0 0 15px}h3{font-size:19px;margin:10px 0}p{margin:10px 0}section{background:white;border:1px solid #dbe4ef;border-radius:18px;padding:28px;margin-top:24px}.lead{background:#152e4c;color:#fff;padding:32px;border-radius:20px}.lead p{color:#d9e6f5}.pill{display:inline-block;padding:3px 11px;background:#edf3f9;border-radius:12px;color:#356181;font-size:14px;margin:3px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.box{padding:18px;background:#f5f8fc;border-radius:12px}.big{font-size:28px;font-weight:700;color:#146c72}.warn{background:#fff6e8;border-left:5px solid #d69030;padding:18px;border-radius:9px}.good{background:#edf8f5;border-left:5px solid #389a81;padding:18px;border-radius:9px}img{max-width:100%;height:auto;border-radius:9px;border:1px solid #e1e8ef;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:16px}th,td{text-align:left;border-bottom:1px solid #e0e6ed;padding:12px}th{background:#f2f6fb}small,.muted{color:#576b80;font-size:15px}details{margin:15px 0}summary{cursor:pointer;font-weight:700}a{color:#135b99}li{margin:7px 0}@media(max-width:750px){.grid{grid-template-columns:1fr}h1{font-size:27px}main{padding:20px 12px}section{padding:18px}table{font-size:14px}}'''
tr=''.join('<tr><td>'+html.escape(k)+'</td><td>'+str(n)+' / 27</td><td>'+html.escape(reason)+'</td></tr>' for k,n,reason in [
('원래 공개 영상',14,'원래 동작에서도 경고가 있음'),('길이 0.75배 (빠르게 재표본)',19,'정상 속도 변화에 반응'),('길이 1.5배 (느리게 재표본)',27,'모든 영상에서 경고'),('중간 0.5초 정지 추가',19,'잠깐 머무는 동작에도 반응'),('2회 / 3회 이어붙임',14,'각각 14개; 알려진 반복과 경로를 함께 복사')])
doc=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손 관측·움직임·누락 시간 분리 검증</title><style>{style}</style><main>
<span class="pill">2026-10-21</span><span class="pill">공개 자료 연구용 검증</span><span class="pill">추가 학습 없음</span>
<div class="lead"><h1>보이는 손과 움직이는 손을 나누고,<br>잠깐 놓친 시간을 숨기지 않기</h1><p>구분·기록 기능은 검증했다. 다만 “여러 장면을 한 장면에 맞춤”이라는 경고는 정상 속도 차이에도 많이 반응해, 자동 오류 판정에는 채택하지 않았다.</p></div>
<section><h2>이번에 확인한 핵심</h2><div class="grid"><div class="box"><div class="big">25 / 25</div><h3>보임 ≠ 움직임</h3><p>이전 표시 25개 쌍-손-프레임 모두 “보이지만 변화 근거 없음”으로 구분했다.</p><small>중복 참고 포함. 25개 영상·확정 오류가 아니다.</small></div><div class="box"><div class="big">27 / 27</div><h3>짧은 누락 기록 보존</h3><p>중간 3프레임을 지운 모든 시험에서 빠진 시간을 기록에 남겼다.</p><small>81프레임 × 15각도 = 1,215항목.</small></div><div class="box"><div class="big">자동 판정 보류</div><h3>경고의 구별 능력 부족</h3><p>정상적인 속도 차이·정지에도 경고가 많다. 감점이나 오답 판정에 사용하지 않는다.</p></div></div></section>
<section><h2>어떤 정보를 나누었나</h2><table><tr><th>표시</th><th>쉽게 말하면</th></tr><tr><td>관측됨</td><td>현재 손의 점이 보였다.</td></tr><tr><td>위치 / 각도 변화 근거</td><td>두 검출 결과 사이의 차이를 감안해도 앞뒤 위치나 모양이 달라 보였다.</td></tr><tr><td>확인된 안정 손내림</td><td>원본 영상에서 아래로 내려놓고 안정된 상태라는 근거가 있다.</td></tr><tr><td>비교 근거 부족</td><td>비교에 필요한 시간인데 손·각도의 확인 정보가 없다.</td></tr></table><p>네 가지 정보는 겹칠 수 있다. 손 위치는 보여도 일부 손가락은 비교하기 어려울 수 있다. 변화가 검출되지 않았다는 것만으로 정지나 손내림을 확정하지 않는다.</p><img src="visuals/evidence_timeline.png" alt="후원 영상의 손별 관측, 변화 근거, 안정 손내림, 비교 부족 시간표"><p class="muted">보라색은 각도 하나라도 변화 근거가 있다는 뜻이고, 빨간색은 각도 하나라도 비교 근거가 부족하다는 뜻이다. 빨간색은 틀린 동작이 아니다. 이 구분은 기하학적 근거이며 수어 의미나 보정된 확률을 뜻하지 않는다.</p></section>
<section><h2>손이 잠깐 사라져도 그 시간을 남겼다</h2><p>기존에는 중간 관측이 없어지면 주변 장면을 다시 연결하면서 관측 비율이 올라갈 여지가 있었다. 이번에는 <b>원래 촬영 시간의 누락 기록을 정렬 경로와 별도로 보존</b>했다.</p><img src="visuals/missing_timeline.png" alt="중간 3프레임 삭제 위치와 15개 각도 누락 표시"><p>노란 범위가 인위적으로 지운 3프레임이다. 해당 시간의 오른손 15각도가 모두 미관측으로 남는다. 그 앞의 빨간 구간은 원래 있던 미관측이다.</p><div class="good">참고의 비교 시간 범위는 그대로 유지했다. 손 전체 누락 54조건도 확인했으며, 원래 쓰지 않는 손을 새로 필수 대상으로 삼지 않았다.</div><p><b>단, 기존 DTW 점수·비율 계산을 바꾼 것은 아니다.</b> 빠진 시간을 보여주도록 보완했으며, 이를 통과·탈락 판정과 연결하는 작업은 아직 하지 않았다.</p></section>
<section><h2>경고가 생겼다고 “잘못했다”고 말할 수는 없었다</h2><p>DTW는 빠른 동작과 느린 동작을 맞추기 위해 여러 시점을 한 시점에 연결할 수 있다. 따라서 이런 연결 자체를 오류로 보면 정상 동작도 걸린다.</p><img src="visuals/normal_warnings.png" alt="정상 영상과 속도·정지·반복 변형의 경고 영상 수"><table><tr><th>조건</th><th>경고가 겹친 영상</th><th>의미</th></tr>{tr}</table><p class="muted">각 손·각도·시간에 8명 중 5명 이상의 참고에서 경고가 겹친 경우를 집계했다. 속도 시험은 좌표를 재표본한 합성 대조이며 새로운 RGB 촬영·검출 시험이 아니다. 반복은 2회·3회로 이어붙이고 참고와 경로도 복사했다. 반복 횟수 자동 검출의 검증 결과는 아니다.</p><div class="warn">정상 변형에서도 경고가 많고, 일부 알려진 문제는 다수 참고의 경고가 모이지 않아 놓쳤다. 현재 경고 기준으로 감점하거나 “수어를 틀렸다”고 판정하지 않는다.</div></section>
<section><h2>앞서 확인한 잘못된 장면 연결은 어떻게 보였나</h2><p>공개 ‘후원’ REAL04의 61번째 프레임은 손을 굽힌 장면인데 새 정렬이 펴진 참고 장면에 연결했다. 이번 진단에서는 해당 연결 구간에 경고가 남았다. 64번째 프레임은 현재 재확인 정보가 부족하여 15각도 모두 비교 근거 부족도 함께 표시한다.</p><img src="../word100_rest_aware_alignment_20261021/visuals/raw_02.jpg" alt="기존 원본 검토: 비교 영상과 기존 및 새 정렬 참고 장면"><p>위 두 줄은 이전 검증에서 보관한 동일 장면이다. 화면의 ‘active’ 표기는 이전 자동 진단명이며 실제 움직임의 정답 주석이 아니다. 아래 줄은 초기 대기 장면이다.</p><p>또 다른 REAL06의 28번째 프레임 문제는 한 참고에서는 경고했지만 다수 참고 조건에는 잡히지 않았다. <b>경고 수가 줄었다는 것만으로 연결 정확도가 좋아졌다고 말할 수 없다.</b></p></section>
<section><h2>정상 손내림·한 손 동작·교차는 보존했나</h2><p>좌표와 좌우 배정, 각도 상태, 저장된 세 조건의 연결 경로는 모두 그대로 두었다. 확인된 손내림에는 근거 부족 표시를 하지 않았고, 한 손 동작 ‘멀다’의 사용하지 않는 왼손은 필수 비교에서 제외된 기존 역할을 유지했다. 손이 안 보인다는 이유로 새로 역할을 제외하지 않았다.</p><details><summary>양손 동작 / 교차가 있는 공개 ‘수어’ 시간표</summary><img src="visuals/crossing_timeline.png" alt="양손 동작 수어의 관측 근거 보존"><p>손을 지우거나 재배정한 결과가 아니라 기존 관측 위에 진단 정보를 덧붙인 결과다.</p></details><details><summary>한 손 동작 ‘멀다’ 시간표</summary><img src="visuals/onehand_timeline.png" alt="멀다에서 선택적 왼손과 필수 오른손의 근거 구분"><p>왼손이 잠깐 보여도 필수 비교 손으로 바꾸지 않았다. 오른손의 빨간 구간은 오류가 아니라 일부 각도의 근거 부족이다.</p></details><p>다만 정상 변형에 대한 경고 과다는 남았다. “정상 움직임에 대한 잘못된 경고까지 모두 해결했다”는 의미는 아니다.</p></section>
<section><h2>이번 결과를 어디까지 믿어도 되나</h2><ul><li><b>범위:</b> 공개 TRAIN 3단어 × 9명 = 27영상. 각자 다른 8명을 참고했다. 100단어 전체의 새 인식 검증은 아니다.</li><li><b>저장 결과 검산:</b> 원래 세 조건 648쌍 + 누락 648쌍 + 정상 변형 1,080쌍 = 2,376결과. 각도/위치 계산과 구간 경고를 별도 수식으로 확인했다.</li><li><b>한계:</b> 실제 사용자가 필수 동작을 생략한 것인지, 카메라에서 가린 것인지 구분한 시험은 아니다. 공개 주석의 동작 구간을 사용했으므로 실시간 웹 구간 검출은 별도 과제다.</li><li><b>보존:</b> 기존 C·T3-F·운영 웹 유지. 사용자·멘토 자료 접근/학습/튜닝/선택 0. 추가 학습·분류 추론 0.</li></ul><p class="muted">Codex가 구현하고 별도 코드로 수식 검산했다. 다른 수어 전문가의 의미 정답 검증은 아니다. 보호 대상 32개 해시와 기존 입력·결과 핀을 확인했다. 새 속도 변형 경로를 별도 최적화 알고리즘으로 재산출하지는 않았다.</p></section>
<section><h2>다음은 이렇게 좁혀서 확인하는 것이 맞다</h2><ol><li>이번 결과를 함께 확인하고, 관측·누락 시간표는 진단 기능으로 보존한다.</li><li>같은 원본의 시간만 바꾼 쌍으로 <b>정답 대응 시점을 아는 시험</b>을 만든다. 정상 속도 차이를 먼저 통과하는지 확인한다.</li><li>한 장면에 연결된 횟수 대신, 확실히 관측된 모양 변화의 <b>앞뒤 순서가 실제로 유지됐는지</b> 검토한다.</li></ol><p>이번에는 여기까지 결과를 정리했다. 경고 문턱을 결과에 맞춰 조정하거나, 특정 단어 예외를 추가하거나, 추가 학습을 시작하지 않았다.</p></section>
<section><h2>검증 파일</h2><p><a href="RESULT_KO.txt">전체 설명 원고</a> · <a href="VERIFY.json">별도 수식 검산</a> · <a href="ANALYSIS.json">세부 결과와 알려진 장면 확인</a> · <a href="FROZEN.json">실행 전 고정 조건</a></p></section>
</main></html>'''
(U/'RESULT_KO.html').write_text(doc,encoding='utf8')
links=re.findall(r'(?:src|href)="([^"]+)"',doc)
for link in links:assert (U/link).is_file(),link
assert '<script' not in doc and len(links)==10
d.write(U/'REPORT_CHECK.json',dict(status='PASS',at=d.now(),static_links=len(links),browser_render=False,
 visual_review=['evidence_timeline.png','crossing_timeline.png','onehand_timeline.png','normal_warnings.png','missing_timeline.png','previous/raw_02.jpg'],
 review='Five charts and previous raw comparison sheet visually inspected. No clipped labels; uncertainty not presented as error.',
 warning='No classification or timing accuracy improvement claim. Repeat paths copied, speed controls synthetic, partial scope explicit.'))
print('REPORT',U/'RESULT_KO.html')
