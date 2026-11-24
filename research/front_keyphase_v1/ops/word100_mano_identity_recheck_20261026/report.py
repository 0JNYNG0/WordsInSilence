from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,html
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_mano_duplicate_guard_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
r=read(U/'RESULT_V3.json');v=read(U/'VERIFICATION.json');assert v['status']=='PASS'
review={'operator':'Codex source image and overlay review, same implementer; not expert anatomical GT','images':['tracking82.jpg','context74_82_88.jpg','conflict78.jpg','conflict85.jpg','conflict86.jpg'],
'prior_reviews_reused':str(G/'VISUAL_REVIEW.json'),
'findings':['f82原본 shows one open hand behind the other fist. Existing-left and HaMeR-right follow the same open fingers, existing root is displaced toward fist. No automatic winner adopted.',
'f74/f88 are before/after visible hand states; f88 hand rotation and lowering differs markedly. A 2D patch cannot be assumed to follow the same fingertip through rotation/occlusion.',
'Flow-valid2/5forward,1/5backward atf82 is algorithm reliability, not expert visible count. Failed red points do not imply incorrect learner motion.',
'f78/85/86 also show same open fingers in two raw hypotheses; these3 were already held before the previous7addition rejection. No newly removed correct hand.',
'Prior distinct contact/crossing controls remain unchanged numerically. Existing broad overlap has one supported right-slot hypothesis but hidden left shape unresolved; both inputs preserved.']}
write(U/'VISUAL_REVIEW.json',review)
text='''팔 연결·앞뒤 영상으로 좌우 신원 재확인 — 2026-10-26

결론
잘못된 기존 좌우 배정을 앞뒤 추적으로 다시 정답처럼 확인하는 위험을 줄였다. 그러나 새 HaMeR 손을 안전하게 채택해 좌표를 교정한 프레임은0이다. 이번 결과는 인식률 향상이나 손 복원 완료가 아니라, 신뢰할 수 없는 확인 근거를 차단한 연구 후보이다. 기존 C·T3-F/운영웹/DTW는 그대로다.

목적과 방법
이전 작업은 같은 실제 손을 두 손으로 추가하는7프레임을 보류했다. 이번에는 기존 손과 HaMeR 후보 중 어느 손이 실제 팔과 연결되고 앞뒤 움직임으로 이어지는지 확인했다.
원본 전체 영상에서 이전/이후0.5초 이내의 가장 가까운 팔 연결이 분명한 관측을 기준점으로 삼아 영상 특징을 현재 장면까지 정방향/역방향으로 추적했다. 현재 팔·손목 연결과 앞뒤 손목/손끝이 모두 지지하는 후보만 잠정적으로 선택하도록 했다. 추적점은 정확한 해부학적 관절 정답이 아니다. 이후 프레임을 쓰므로 촬영 완료 후 분석용이며 실시간 검증은 아니다. 손가락 정답/목표 단어는 입력하지 않았다.

발견과 보완
첫 후보는 겹침 구간에 이미 잘못 배정된 기존 왼손칸도 추적 기준점으로 삼았다. 이러면 같은 잘못된 손을 계속 따라가는 것만으로 왼손이라는 판단을 다시 강화한다. 후원79~83프레임5곳의 내부 재확인 판단에서 이 위험이 나왔다. 최종 후보는 서로 다른 좌우칸의 후보가 같은 손을 따라가는 충돌이 알려진 관측을 기준점에서 제외한다. 해당 잘못된 재확인5→0이다. 첫 후보도 실제 좌표 교체까지 하지는 않았으므로5개 좌표 오류를 고쳤다는 뜻은 아니다.
손끝이 닿는 정상 두 주먹은 손목 연결이 서로 다르다. 손목까지 함께 맞는 가설끼리 경쟁하도록 해, 손끝이 가깝다는 이유만으로 별개 손을 막지 않도록 했다. 독재자 접촉 대조에서 첫 후보 왼손불확실/오른손지지→최종양쪽지지. 원래 관절은 전 과정에서 보존됐다.
중간 후보에서는 같은 좌우칸의 HaMeR가 기존 손을 정상 재확인한 경우까지 기준점에서 제외하는 범위 오류를 발견했다. 최종에서는 서로 다른 좌우칸 사이의 충돌만 제외하도록 고쳤다. 모든 후보와 결과를 보존했으며 거리·시간·손가락 수 기준을 결과에 맞춰 완화하지 않았다.

최종 검증 결과
공개7영상4명5표현812프레임, 저장HaMeR243손요청, 원래57빈손칸을 재사용했다. 앞뒤 영상의 실제 픽셀 추적은 이번에 새로 수행했다. 학습과 새 HaMeR추론은0이다.
같은 손으로 의심되는 충돌은77~86의10프레임이다. 앞서 보류한 신규추가7프레임과 그전부터 보류돼 있던3프레임(78/85/86)을 합한 수이다. 오류가7에서10으로 늘었다는 뜻은 아니다.
최종 자동 좌표 교체0. f82에서 이전74프레임에서 따라온 오른손 손끝은5개중2개만 추적 품질 조건을 통과했고, 이후88프레임에서 거꾸로 따라온 손끝은1개였다. 이후 장면으로부터 손목 추적도 유효하지 않았다. 기준을 낮추어 성공처럼 만들지 않고 확인 불가로 남겼다. 이 숫자는 보이는 손가락 수나 사람의 정답 판정이 아니다.
기존14,149관절점, 지정정상교차20프레임840점, 정상대조94프레임 결과 유지. 충돌이 없는 곳은 비교가능 상태도 그대로다. 충돌10프레임의양쪽20손칸은 비교가능 여부를별도보류한다(한칸은이미비어있어20개의실제손을지웠다는뜻아님). 원본 좌표/새후보/기존G모두그대로보존했다. 실제DTW연결은하지않았다.
추가 대조는 지난300영상 검사에서 나온2개원본영상장면을이번방법으로재확인했다. 300전체에새추적/HaMeR를실행한것이아니다. 독재자 정상접촉은양쪽신원지지, 추출겹침은오른손가설만지지/왼손가설보류다. 추출을자동교정하거나가려진손을복원하지않았다.

남은 문제
눈으로보면새HaMeR손목이기존값보다팔연결에가깝게보이는장면이있다. 하지만현재RGB점추적은손회전·겹침중손끝이나손목의동일지점을충분히따라가지못해일반규칙으로선택할근거가부족하다.
이전부터남아있던E02/E06과도보류가능성/E15손목오차/숨은손가락3D정답부족은미해결. 정상관측보존은새보완의정확복원능력이나전체수어일반화검증이아니다.

다음 한 작업
신원·손목 위치의 근거와 손가락 모양의 근거를 실제 선택 단계에서도 더 분리한다. 현재 프레임의 팔→손목→손바닥 연결과 손 영역 대응을 먼저 검증하고, 손가락 정밀 추적은 별도 상태로 둔다. 손끝4개추적을못했다는이유만으로위치근거까지버리지않되,영역추적만으로정밀손가락이맞다고승격하지않는다. 현재시험에서보인손회전·겹침·정상접촉을공통반례로유지한다. 기준완화/특정단어예외/기존모델교체/사용자튜닝없이공개자료에서검증한뒤채택여부를결정한다.

검증 및 보존
Codex구현/시각확인,별도스칼라검산도같은Codex로독립전문가GT는아니다. 영상추적637경로1949단계/좌우판단257회검산,계약검사7개통과. 보호32파일과과거609파일해시불변. 사용자·멘토촬영접근/학습/튜닝/선택0,운영모델/웹/DTW변경0,카메라/다운로드/외부전송/Git작업0.
V1/V2/V3개발·검증이같은공개진단자료이므로새사람최종평가가아니다. HTML정적구조/링크/이미지확인,브라우저렌더미확인.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
def img(path,caption):return f'<figure><img src="{path}" loading="lazy" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
tracking=img('views/tracking82.jpg','현재 장면의 두 후보와 앞뒤 영상 추적. 청록색 점은 추적 품질 조건 통과, 붉은 점은 실패/불확실을 뜻합니다. 손동작의 정답·오답 표시는 아닙니다.')
context=img('views/context74_82_88.jpg','원본 전체 장면. 이후88프레임에는 손이 회전하며 내려갑니다. 이 변화를 지나 같은 손끝을 계속 추적할 수 있다고 가정하면 안 됩니다.')
others=''.join(img(f'views/conflict{t}.jpg',f'{t}프레임 · 이미 새 손 추가가 보류돼 있던 장면. 원본 후보를 보여줄 뿐 두 손으로 채택한 결과가 아닙니다.') for t in [78,85,86])
old='../word100_mano_duplicate_guard_20261025/views/'
normal=img(old+'PROXY1_NIA_SL_WORD0429_REAL01_F_0059.jpg','정상 접촉: 가까운 두 주먹을 한 손으로 합치지 않습니다. 최종 진단에서 현재 팔과 앞뒤 영상이 양쪽을 각각 지지했습니다.')
ambig=img(old+'PROXY2_NIA_SL_WORD0920_REAL02_F_0053.jpg','남은 겹침: 오른손 가설은 지지되지만 다른 손의 신원·모양은 불확실합니다. 원본을 바꾸거나 보이지 않는 손을 만들어 넣지 않았습니다.')
page=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>팔·앞뒤 영상 신원 재확인</title><style>
:root{{color-scheme:dark;font-family:"Malgun Gothic",system-ui;background:#0d1725;color:#eef3fa}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1220px;margin:auto;padding:45px 24px 80px}}h1{{font-size:38px;line-height:1.4;letter-spacing:-1px}}h2{{font-size:25px;margin-top:0}}p,li{{line-height:1.85}}section{{margin:30px 0;background:#152338;border:1px solid #304359;border-radius:16px;padding:28px}}.eyebrow{{color:#74ded1;font-size:14px}}.lead{{font-size:20px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}}.card{{border:1px solid #385169;background:#192d40;border-radius:12px;padding:20px}}.big{{display:block;color:#6fe2cd;font-size:34px;font-weight:800}}.warn{{border-left:4px solid #f3bd70;background:#302b25;padding:16px 20px;line-height:1.85}}.small{{font-size:14px;color:#bfccdc}}figure{{margin:22px 0}}img{{width:100%;border-radius:10px}}figcaption{{line-height:1.8;color:#c1d0e3;margin-top:9px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:14px 10px;border-bottom:1px solid #3b4c62;line-height:1.7}}a{{color:#7bdfd7}}summary{{cursor:pointer;color:#7bdfd7;padding:10px 0}}@media(max-width:760px){{main{{padding:25px 14px}}h1{{font-size:28px}}.cards{{grid-template-columns:1fr}}section{{padding:20px}}}}
</style></head><body><main><div class="eyebrow">WordsInSilence · 2026.10.26 · 공개 자료 · 운영 미적용</div><h1>앞뒤로 잘 따라갔어도,<br>처음 손 이름이 틀렸다면?</h1><p class="lead">기존 손과 HaMeR 손의 충돌을 다시 확인했습니다. <b>잘못된 좌우 판단을 재확인하는 위험은 줄였지만, 안전하게 좌표를 교체한 장면은 없었습니다.</b></p><div class="cards"><div class="card"><span class="big">5 → 0</span>잘못된 기존 신원을 다시 지지한 판단<br><span class="small">내부 확인 단계 · 좌표교정5건 아님</span></div><div class="card"><span class="big">0프레임</span>충분한 근거로 새 좌표 채택<br><span class="small">남은 충돌은 억지로 확정하지 않음</span></div><div class="card"><span class="big">20프레임</span>지정 정상 교차 유지<br><span class="small">정상 대조94프레임도 유지</span></div></div>
<section><h2>이번에 하려던 일</h2><p>같은 펼친 손에 기존 뼈대와 HaMeR 뼈대가 함께 붙을 때, <b>어느 팔에서 이어진 손인지</b> 확인하려고 했습니다. 현재 손목·팔 연결을 확인하고, 앞뒤에 보였던 손을 원본 영상에서 현재까지 따라와 비교했습니다.</p><ol><li>현재보다 이전·이후0.5초 이내에서 팔 연결이 분명한 가까운 관측을 고릅니다.</li><li>영상의 특징점을 현재 장면까지 정방향·역방향으로 추적합니다.</li><li>현재 팔 연결과 앞뒤 손목·손끝 근거가 함께 지지하는 후보만 선택합니다.</li><li>좌우나 모양을 확정할 수 없으면 원본을 보존하고 비교 근거 부족으로 남깁니다.</li></ol><p class="small">전체 영상 좌표를 사용합니다. 상반신 제한은 후보 선택 뒤에 적용합니다. 미래 프레임을 사용하므로 촬영 완료 후 분석용입니다. 손가락 모양의 정답이나 목표 단어는 추적에 넣지 않았습니다.</p></section>
<section><h2>새로 발견한 문제: 오류가 오류를 확인하는 순환</h2><p>첫 후보는 겹침 중 <b>이미 잘못 이름 붙은 손</b>을 앞뒤 추적의 기준점으로 삼기도 했습니다. 그러면 같은 손을 계속 따라갔다는 사실만으로, 틀린 왼손이라는 이름까지 다시 믿게 됩니다.</p><div class="warn">손을 계속 따라갔다는 근거와, 그 손이 왼손이라는 근거는 다릅니다.</div><p>최종 후보는 <b>서로 다른 좌우칸의 후보가 같은 손을 따라가는 충돌 장면을 기준점에서 제외</b>했습니다. 후원79~83프레임에서 잘못된 기존 신원을 지지하던 내부 판단5개가 보류로 바뀌었습니다. 첫 후보도 실제 좌표를 바꾸지는 않았으므로, 좌표오류5개를 교정했다는 뜻은 아닙니다.</p><p>반대로 같은 좌우칸의 두 검출기가 정상적으로 같은 손을 확인한 경우는 유지합니다. 중간 후보에서 이를 과도하게 제외하던 범위 오류도 바로잡았습니다.</p></section>
<section><h2>왜 새 손으로 바로 바꾸지 못했나</h2>{tracking}<table><tr><th>82프레임으로 따라온 경로</th><th>손끝 추적 품질 통과</th><th>손목 근거</th></tr><tr><td>이전74 → 현재82</td><td>2 / 5개</td><td>새 후보 가까이에 유지</td></tr><tr><td>이후88 → 현재82</td><td>1 / 5개</td><td>추적 근거 부족</td></tr></table><p>새 후보가 그럴듯하게 보여도, 앞뒤 영상에서 충분한 동일 손 근거를 확보하지 못했습니다. 그래서 기준을 낮추거나 빈 점을 채워 넣지 않고 <b>자동 교체0프레임</b>으로 남겼습니다. 보이는 손가락 개수나 사용자가 잘못 움직였다는 판단이 아닙니다.</p>{context}</section>
<section><h2>정상 손은 계속 살려 두었습니다</h2>{normal}<p>가까운 손끝만으로 두 손을 경쟁시키지 않고, <b>손목까지 일치하는 가설끼리</b> 비교하도록 했습니다. 정상적인 두 주먹 접촉에서 최종적으로 양쪽 손이 각각 지지됐습니다.</p><table><tr><th>보존 항목</th><th>결과</th></tr><tr><td>기존 유효 관절</td><td>14,149점 보존</td></tr><tr><td>지정 정상 교차</td><td>20프레임 · 840점 및 기존 보완 결과 유지</td></tr><tr><td>정상 대조 영상</td><td>94프레임 결과·비교가능 상태 유지</td></tr><tr><td>충돌 없는 영역</td><td>좌표·시간·범위·비교가능 상태 유지</td></tr></table><p class="small">원래 정상 값이 새로 손상되지 않았다는 검증입니다. 모든 관절의 정밀 정확도나 새로운 사용자 성능을 확인했다는 뜻은 아닙니다.</p></section>
<section><h2>남은 충돌과 이번 시험의 범위</h2>{ambig}<p>공개7영상·4명·5표현·812프레임에서 저장HaMeR243손요청을 재사용했습니다. 실제 앞뒤 영상 추적은 새로 수행했고, 추가 학습과 새 HaMeR 추론은 하지 않았습니다. 지난300영상 점검에서 나온 위 두 장면을 추가로 확인했으며, 300전체에 새 추적을 수행한 것은 아닙니다.</p><p>충돌10프레임은 <b>지난번 중복 추가를 막은7개 + 그전부터 보류돼 있던3개</b>입니다. 새 오류가3개 늘어난 것이 아닙니다. 양쪽20손칸의 비교가능 여부를 따로 보류하되, 원본 좌표와 두 후보는 그대로 보관합니다. 한쪽은 이미 비어 있으므로20개의 실제 손을 삭제했다는 뜻도 아닙니다.</p><details><summary>이미 보류돼 있던3프레임 보기</summary>{others}</details></section>
<section><h2>다음에는 무엇을 바꿔야 하나</h2><p><b>손의 신원·위치 확인과 손가락 모양 확인을 실제 후보 선택 단계에서도 더 분리해야 합니다.</b></p><ol><li>현재 장면의 <b>팔 → 손목 → 손바닥</b> 연결과 손 영역 대응을 먼저 검증합니다.</li><li>손끝을 충분히 추적하지 못해도 남아 있는 위치 근거는 따로 보존합니다.</li><li>손 영역을 찾았다는 이유로 가려진 손가락 모양까지 정확하다고 승격하지 않습니다.</li><li>정상 접촉·교차·회전·손내림을 공통 대조로 유지하고, 확실한 이득이 확인될 때만 다음 연결을 검토합니다.</li></ol><p>기존 과도 보류 가능성, 손목 오차, 가려진 손가락의 정답 부족은 남아 있습니다. <b>C·T3-F와 운영 웹·DTW는 그대로</b>이며 사용자 촬영은 학습·튜닝·후보 선택에 사용하지 않았습니다.</p></section>
<section><h2>검산과 한계</h2><p>저장 영상추적637경로·1949단계와 좌우판단257회를 별도 계산으로 검산했습니다. 정책 검사7개 통과, 보호32파일과 과거609파일 해시도 유지됐습니다. 초기·중간·최종 후보와 문제 발견 기록은 모두 보존했습니다.</p><p class="small">구현·시각확인·별도 수치검산 담당은 같은 Codex입니다. 독립 전문가의 손가락·좌우 정답 검증을 대신하지 않습니다. 같은 공개 진단 자료에서 개발·재검토했으므로 독립 최종 평가도 아닙니다. HTML 구조·링크·이미지 정적 점검, 브라우저 렌더링 미확인.</p><p><a href="RESULT_KO.txt">전체 설명</a> · <a href="RESULT_V3.json">최종 수치</a> · <a href="VERIFICATION.json">별도 검산</a> · <a href="FROZEN_V3.json">수정 근거</a> · <a href="VISUAL_REVIEW.json">시각 확인</a></p></section></main></body></html>'''
(U/'RESULT_KO.html').write_text(page,encoding='utf-8')
write(U/'COMPLETION.json',{'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'status':'IDENTITY_EVIDENCE_GUARD_VERIFIED_NO_COORDINATE_RECOVERY_NOT_DEPLOYED','implementation':'Codex','verification':v,'comparison':{'v1_wrong_old_identity_support':5,'v3_wrong_old_identity_support':0,'duplicate_conflict_frames':10,'previous_new_addition_holds':7,'previously_held_other_frames':3,'safe_coordinate_replacements':0,'distinct_fists_identity_support_v1':[-1,1],'distinct_fists_identity_support_v3':[0,1]},'report':str(U/'RESULT_KO.html'),'training':0,'new_neural_inference':0,'user_data_access':0,'C_T3F_change':False,'production_change':False,'running_processes':[]})
print('Report complete')
