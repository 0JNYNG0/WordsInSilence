from pathlib import Path
import json,html
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
c=read(U/'COMPLETION.json');r=read(U/'RESULT.json');v=c['verification']
text='''손 위치와 손가락 모양의 근거 분리 — 공개 영상 검증 결과

이번 결론
손끝을 정확히 따라가지 못했을 때에도 팔과 손 영역에서 얻은 정보를 별도로 남기도록 연구 후보를 구현했다. 위치와 좌우를 뒷받침하는 정보는 일부 늘었지만, 기존 좌우 충돌10프레임을 해결하거나 새 관절을 안전하게 복원한 것은 아니다. C·T3-F와 실제 웹/DTW는 그대로다.

어떻게 바꿨나
이전에는 손목과 손끝4/5를 앞뒤 장면에서 모두 따라가는 엄격한 조건을 사용했다. 이번에는 그 정밀 근거를 그대로 보존하면서, 원본 전체 화면의 손 영역에 있는 이미지 특징점과 현재 어깨·팔꿈치·손목 연결, 팔 양쪽 윤곽을 별도로 조사했다. 영역 추적점은 손가락 관절점이 아니며, 그 영역에 손이 있다고 가려진 손가락 모양이 확인되는 것은 아니다.
중복 좌우가 의심되는 관측은 추적 시작점으로 쓰지 않았다. 같은 손에 대해 앞뒤0.5초 이내에서 가장 가까운 깨끗한 관측을 사용했다. 손끼리 겹친 부분은 시작 특징점에서 제외했고, 앞뒤 중 하나라도 근거가 부족하거나 팔과 영역이 다른 쪽을 가리키면 좌우를 확정하지 않았다. 정면 화면을 좌우 반으로 나누는 규칙은 쓰지 않았다. 원본 전체영상 검출/추적 후 고정상반신 입력범위라는 기존 원칙은 유지하며, 보고서 확대이미지는 설명용이다.

어떤 자료를 확인했나
공개TRAIN 7영상·4명·5표현·812프레임. 저장된HaMeR243요청과 기존 관측을 확인했다. 이중원래손이통째로빠진칸은57개다. 이전부터고정한독재자접촉·추출겹침2장면도함께확인했다.100개단어전수실험·새사용자시험이아니다. 단어이름/촬영자ID는알고리즘입력에사용하지않았다.

결과를 정확하게 읽기
- 기존 정밀추적이 좌우를 뒷받침했던63건은 그대로 보존했다. 새 영역 방식도 지지한44건, 기존 정밀근거만 남은19건이다.
- 기존 정밀추적에서 미확정이었던16건에 위치·좌우 보조근거가 추가됐다. 같은Codex가16건원본/겹쳐그린화면을검토했고보이는팔·손과제안좌우는일치했다. 독립전문가정답/손목정밀좌표검증은아니다.
- 16건 중15건은 원래손21점이있던장면이다. 나머지1건은대지36프레임왼손이며,이미지난G/GT에HaMeR추정값이있었다. 이번에는그추정의위치근거만추가로확인했다. '손16개복구'나'새손1개복구'가아니다.
- 243요청 중기존정밀63/새영역60은서로다른근거수다.44건이겹치고19건은기존근거만,16건은새근거만있다. 이를인식정확도비교나정답률로해석하면안된다. 새영역검사실패로기존정밀근거를삭제하지않았다.
- 실제새좌표채택0,기존좌우충돌10프레임미해결. TCN인식률·DTW피드백정확도향상미측정.

왜 후원은 아직 보류인가
82프레임에서 이전74프레임의 오른손 영역은12개특징점중8개를따라왔다. 이전손끝정밀추적은2/5에그쳤으므로'손이있는영역'을파악하는정보는더남았다. 그러나이후88프레임에서거슬러추적할때는11개중5개만남아실패했다. 손이회전하거나다른손과접촉하며보이는무늬가달라진다. 기존후보와HaMeR후보의손목위치도달라각각다른팔윤곽에맞는듯보일수있다. 영역/팔윤곽만으로하나를정답으로확정하지않았다. 이모든장면은추적실패이지손이실제로없다는판정이아니다.

정상 손과 교차에서 확인한 한계
독재자두주먹접촉에서는새영역검사가오른손후보를왼손영역과연결하려는모호함을보였다. 현재오른팔근거와충돌해채택을차단했고,기존정밀근거는양손모두그대로남겼다. 따라서영역추적은기존정밀검사를대체할수없다.
추출겹침은오른손근거유지/왼손불확실로남았다. 대지57프레임교차에서해부학적오른팔을따른영역근거는추가됐다. 원래관측14149점,지정정상교차20프레임840점,정상대조94프레임과기존비교가능상태를모두그대로보존했다. 배열보존은코드의보존계약검증이지새방식이모든정상손을정확히검출한다는증거는아니다.

검증과 제한
3198영역추적경로/3595단계/1814후보판단/7127영역거리계산을별도scalar·기하검산해PASS. 계약7PASS. 보호32파일과지난1963파일의해시불변. 팔윤곽은지난독립수치검산을마친코드를고정재사용했다. 구현·시각검토·별도검산동일Codex,독립사람평가아님. 근거가약한영역제안은정보로만보존하고실제좌표교체/운영비교가능승격에쓰지않았다.
전체오프라인진단실행약79.4초. 저장HaMeR재사용이며앞뒤미래프레임도쓰므로실시간FPS·전체추론속도가아니다. 새신경망추론/학습/사용자·멘토자료사용0. HTML정적링크와이미지시각검토,브라우저렌더미확인.

다음 한 작업
기존10충돌과정상접촉·교차공개대조에서각후보손목이RGB의실제팔과손이이어지는부위에맞는지확인할평가기준부터정리한다. 팔윤곽이이어진다는근거와정확한손목위치근거를분리하고,가려진점은범위/불확실로표시한다. 결과에맞춰문턱을낮추거나같은장면만반복튜닝하지않는다. 위치근거를보존하는방향은유효하지만,현단계에서기존모델을교체하거나숨은손가락교정에쓰지는않는다.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
sections='''<section class="hero"><span class="eyebrow">WORDS IN SILENCE · PUBLIC RESEARCH</span><h1>손이 어디 있는지와<br>손가락이 어떤 모양인지를<br><em>따로 확인했습니다.</em></h1><p>위치 근거는 더 남겼습니다. 하지만 가려진 손가락이나 남은 좌우 충돌을 해결했다고 판단하지 않았습니다.</p><div class="tag">연구 후보 검증 완료 · 운영 미적용 · 추가 학습 없음</div></section>
<section><h2>이번 결과, 세 가지로 보면 됩니다</h2><div class="cards"><article><strong>16건</strong><h3>위치·좌우 보조근거 추가</h3><p>손끝 정밀 추적이 부족했던 장면에서 팔과 손 영역의 연결 정보를 따로 남겼습니다.</p></article><article><strong>1건</strong><h3>원래 손이 빠져 있던 장면</h3><p>16건 중 나머지15건은 이미 손이 검출돼 있었습니다. 이1건도 이전HaMeR추정을 뒷받침한 것이며, 새 복원이 아닙니다.</p></article><article><strong>0건</strong><h3>새 관절 좌표 채택</h3><p>기존 좌우 충돌10프레임은 여전히 미해결입니다. 인식률 향상은 이번에 측정하지 않았습니다.</p></article></div></section>
<section><h2>왜 나누었나요?</h2><div class="flow"><article><span>① 위치</span><h3>손이 이 영역에 있나?</h3><p>손 주변 이미지 특징점을 앞뒤 장면에서 따라옵니다. 관절점과는 다릅니다.</p></article><b>→</b><article><span>② 좌우</span><h3>어느 팔에 연결되나?</h3><p>현재 어깨·팔꿈치·손목, 팔 윤곽과 대조합니다. 화면의 왼쪽·오른쪽으로 정하지 않습니다.</p></article><b>≠</b><article><span>③ 모양</span><h3>손가락까지 보이나?</h3><p>위치가 맞아도 가려진 손가락은 확정하지 않습니다. 기존 정밀 근거도 별도로 유지합니다.</p></article></div><p class="note">추적은 원본 전체 화면에서 진행했습니다. 아래 확대 이미지는 설명용이며, 학습 영역을 더 좁힌 것이 아닙니다.</p></section>
<section><h2>기존 정보는 버리지 않았습니다</h2><p>저장된HaMeR 후보243건을 같은 조건에서 확인했습니다. 아래는 정확도가 아니라 <b>어떤 근거가 남았는지</b>를 나타냅니다.</p><div class="bars"><div style="width:7.82%;background:#709cd9" title="기존 정밀 근거만19"></div><div style="width:18.11%;background:#48ba9f" title="공통44"></div><div style="width:6.58%;background:#efb56a" title="새 영역 근거만16"></div><div style="width:67.49%;background:#263a50" title="둘 다 좌우 미확정164"></div></div><div class="legend"><span>기존만19</span><span>공통44</span><span>추가16</span><span>둘 다 미확정164</span></div><p>기존 정밀추적의63건은 그대로 보존했습니다. 새 영역 검사가 실패한19건도 기존 근거를 삭제하지 않았습니다. 새 영역 제안은 아직 자동 좌표 교체에 사용하지 않습니다.</p></section>
<section><h2>후원: 영역은 따라왔지만, 정답 확정은 보류</h2><img src="views/control_0_82.jpg" alt="후원82프레임의 기존 왼손칸과 HaMeR오른손 후보"><div class="cards two"><article><h3>늘어난 정보</h3><p>이전74→82프레임에서 손 영역12점 중8점을 따라왔습니다. 앞선 손끝 정밀 추적은2/5만 유효했습니다.</p></article><article><h3>아직 부족한 정보</h3><p>이후88→82역추적은11점 중5점만 남아 실패했습니다. 후보마다 손목 뿌리도 달라 현재 팔 윤곽만으로 하나를 정답으로 고를 수 없습니다.</p></article></div><p class="note">초록·보라 점은 이미지 특징점입니다. 손가락 관절이나 정확한 손바닥 중심을 복원한 점이 아닙니다. 그림의 -1은 보류를 뜻합니다.</p></section>
<section><h2>정상 두 손이 가까우면 영역도 헷갈릴 수 있습니다</h2><img src="views/control_7_59.jpg" alt="독재자 정상접촉 대조"><p>두 주먹이 붙은 장면에서 영역만 보면 오른손 후보가 왼손 쪽 영역에 가까웠습니다. 현재 팔 연결과 맞지 않아 새 제안은 차단했고, 기존의 양손 정밀 근거는 보존했습니다.</p><p class="callout">영역 추적을 기존 손 배정의 대체재로 쓰지 않습니다. 손가락 근거와 나란히 보관하는 보조 정보입니다.</p><details><summary>추출 겹침 대조 보기</summary><img src="views/control_8_53.jpg" alt="추출 겹침의 좌우 근거"><p>오른손 근거는 유지되고 왼손은 불확실합니다. 가려진 왼손 모양을 정답으로 확정하지 않았습니다.</p></details></section>
<section><h2>추가 근거16건 — 원본과 직접 비교</h2><p>원본 왼쪽 / 분석 표시 오른쪽. 16건 모두 보이는 팔·손과 제안 좌우를 시각 확인했습니다. 같은Codex의 검토이며 독립전문가 정답 검증은 아닙니다. 정밀한 손목 좌표와 숨은 손가락의 정확도를 보증하지 않습니다.</p>'''
for i in range(1,5):sections+=f'<details{" open" if i==1 else ""}><summary>추가 근거 {(i-1)*4+1}–{i*4} 보기</summary><img src="views/new_support_{i}.jpg" alt="추가 근거 원본 대조 {i}"></details>'
sections+='''<p>실제 손이 빠졌던 장면은 대지36프레임 왼손1건입니다. 기존G/GT에 이미HaMeR후보가 있었으므로 이번 결과를 “새 손 복구”로 세지 않았습니다.</p></section>
<section><h2>보존·검증 범위</h2><table><tr><th>공개 자료</th><td>7영상 ·4명 ·5표현 ·812프레임 +고정 대조2장면</td></tr><tr><th>원래 누락된 손</th><td>57손칸 · 기존 좌우 충돌10프레임</td></tr><tr><th>기존 결과 보존</th><td>관측14,149점 · 정상교차20프레임840점 · 정상대조94프레임</td></tr><tr><th>별도 수치 검산</th><td>3,198추적 경로 ·3,595단계 ·1,814판단 ·7,127영역거리 계산</td></tr><tr><th>보호 확인</th><td>계약7PASS · 보호32파일/이전1,963파일 해시 동일</td></tr></table><p>정상 결과 보존은 기존 배열과 마스크를 유지했는지 확인한 것입니다. 새 보조 방식이 모든 손을 정확히 판별한다는 의미는 아닙니다. 공개TRAIN 진단자료를 재사용했고100단어 전수/새사용자 최종평가가 아닙니다.</p><p>촬영 후 앞뒤 영상을 사용하는 오프라인 시험입니다. 약79.4초는 이번 진단 전체 실행시간으로, 신경망을 포함한 실시간 처리속도나FPS가 아닙니다.</p></section>
<section><h2>다음은 손목의 실제 연결 부위를 확인합니다</h2><p>같은 공개 대조에서 후보손목이 <b>이미지의 실제 팔과 손이 이어지는 부위</b>에 맞는지 확인할 기준부터 정리합니다. “팔 윤곽이 이어진다”와 “손목 위치가 맞다”를 구분하고, 가려진 점은 불확실한 범위로 남깁니다.</p><p>기존 C·T3-F, 웹과DTW는 보존했습니다. 사용자·멘토 촬영은 열거나 학습·튜닝·선택에 사용하지 않았습니다. 추가학습·운영교체는 하지 않았습니다.</p><a class="button" href="RESULT_KO.txt">상세 결과 원문</a> <a class="button secondary" href="VERIFICATION.json">검산 기록</a></section>'''
css='''*{box-sizing:border-box}body{margin:0;background:#0b1421;color:#e8eff6;font-family:"Malgun Gothic",sans-serif;line-height:1.8}main{max-width:1180px;margin:auto;padding:30px 30px 80px}section{margin:0 0 32px;padding:34px;background:#101f31;border:1px solid #25354a;border-radius:20px}.hero{padding:56px 44px;background:linear-gradient(140deg,#112b3d,#142036)}.eyebrow{font-size:12px;letter-spacing:.2em;color:#82cfc6}h1{font-size:44px;line-height:1.5;margin:20px 0}em{font-style:normal;color:#75ddc4}h2{font-size:25px;line-height:1.5;margin:0 0 20px}h3{font-size:18px;line-height:1.6;margin:8px 0}p{color:#c8d4e0;margin:12px 0}.tag{display:inline-block;border:1px solid #477666;border-radius:22px;color:#8ce6cc;padding:6px 17px;margin-top:15px}.cards,.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.cards article,.flow article{background:#182a3f;border-radius:14px;padding:22px}.cards strong{display:block;font-size:38px;color:#83e5cd}.two{grid-template-columns:repeat(2,1fr)}.flow{grid-template-columns:1fr 22px 1fr 22px 1fr;align-items:center}.flow span{color:#83e5cd}.flow b{font-size:24px}img{display:block;width:100%;height:auto;border-radius:12px;margin:20px 0}.note{font-size:14px;color:#9db1c5}.callout{border-left:4px solid #e0b46e;padding:12px 20px;background:#243043}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:15px;border-bottom:1px solid #2b3c50}th{color:#90d7ca;width:26%}summary{cursor:pointer;font-weight:bold;padding:16px;background:#192e43;border-radius:10px}details{margin-top:14px}.bars{display:flex;width:100%;height:30px;border-radius:8px;overflow:hidden;margin:24px 0 10px}.legend{display:flex;gap:24px;flex-wrap:wrap;color:#bccde0}.button{display:inline-block;text-decoration:none;background:#81d7c3;color:#0e2330;padding:10px 22px;border-radius:9px;margin-top:15px;font-weight:bold}.secondary{background:#293f57;color:#e7f1fa}@media(max-width:800px){main{padding:12px}section,.hero{padding:24px}h1{font-size:30px}.cards,.two,.flow{grid-template-columns:1fr}.flow b{display:none}td,th{font-size:14px;padding:8px}}'''
page=f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손 위치·모양 근거 분리 | WordsInSilence</title><style>{css}</style></head><body><main>{sections}</main></body></html>'
(U/'RESULT_KO.html').write_text(page,encoding='utf-8');print('Korean report and7image links ready')
