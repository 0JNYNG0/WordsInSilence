from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,html,hashlib
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
r=read(U/'RESULT_V2.json');v=read(U/'VERIFICATION.json');items=read(U/'VISUAL_ITEMS.json');assert v['status']=='PASS'
review=[]
for x in items:
 if x['type']=='changed':
  finding='펼친 실제 손 하나의 손가락 위에 기존 좌표와 새 HaMeR 좌표가 함께 놓임. 새 추가를 보류함. 기존 뼈대의 손목 치우침과 좌우 칸 오류는 남아 있으며 주먹 손을 복원한 것은 아님.'
  category='duplicate_new_addition_withheld_existing_identity_still_unresolved'
 elif x['type']=='protected_control' and x['frame'] in [66,90]:
  finding='주먹과 펼친 손은 서로 다른 두 손으로 보임. 기존 재검출/보완 결과를 새 규칙이 추가로 막지 않음. 각 손가락의 정밀 정확도 보증은 아님.';category='distinct_overlap_addition_preserved'
 elif x['type']=='protected_control':
  finding='기존부터 주먹 손의 새 후보가 보류된 장면. 새 규칙으로 달라지지 않았으며 과도한 보류 가능성은 해결되지 않음.';category='previous_hold_unchanged'
 elif 'PROXY1' in x['id']:
  finding='원본과 전후 장면에서 서로 다른 두 주먹이 가까이 닿음. V1 근접 검사만으로 의심 처리했으나 V2 손끝 일대일 대응 검사에서는 의심하지 않음.';category='normal_contact_v1_false_alarm_resolved_v2'
 else:
  finding='실제 두 손이 교차하지만 기존 양손 좌표가 앞쪽 같은 손가락을 따라가는 모습. 중복/혼합 좌표 가능성이 있어 V2도 불확실로 남김. 가려진 손가락의 정확한 대응은 확정하지 않음.';category='overlap_existing_coordinate_identity_unresolved'
 review.append({**x,'finding':finding,'category':category})
write(U/'VISUAL_REVIEW.json',{'operator':'Codex, same author as implementation; not independent expert annotation','reviewed_items':13,'changed_frames':7,'protected_examples':4,'broad_proxy_examples':2,'broad_context_panels_reviewed':2,'primary_visuals_reused_from_v1_after_all_arrays_verified_identical':True,'items':review,'limitations':'No expert 3D/finger ground truth; repeated frames of one episode are not independent successes. Broad control is a cached coordinate-pair proxy, not HaMeR applied to300videos.'})

body='''같은 손의 중복 추가 방지 — 연구 후보 검증 결과

결론: 기존 손과 HaMeR가 같은 펼친 손을 두 번 가리키던 7개 프레임에서 새 손의 중복 추가를 막았다. 이미 관측된 손 좌표는 유지했다. 기존 뼈대의 좌우 칸 오류나 잘못된 손목을 고친 결과는 아니며, 숨은 주먹 손을 복원한 것도 아니다.

원인과 변경
이전 통합은 두 후보의 손목이 서로 다른 팔에 연결됐다고 나오면 손가락의 중복 여부를 확인하지 않고 통과시켰다. 손목 자체가 잘못 잡힌 장면에서는 같은 실제 손이 두 개로 들어갈 수 있었다.
새 후보는 원본 전체 영상의 좌표에서 여러 손가락 마디와 끝의 일치를 확인한다. 손목이나 좌우 라벨이 달라도 같은 손가락을 함께 따라가면 중복 가능성을 표시한다. 단지 가까운 두 손을 혼동하지 않도록 엄지·검지 등의 손끝이 서로 일대일로 대응하는지도 확인한다.
공통 규칙은 단어명·촬영자·프레임 번호를 입력받지 않는다. 기존 점 거리 기준 0.08×어깨너비와 5개 중 4개 손가락 조건을 사용한다. 보편적인 정답 판정이나 확률이 아닌 공학적 후보 기준이다.
충돌하면 새 추정의 사용만 보류하고 기존 좌표와 원본 HaMeR 후보를 보존한다. 기존 손이 맞다고 확정하지 않으며, 관련 양쪽 손칸에 identity_conflict 상태를 따로 기록한다. 향후 DTW 연결 시 이 상태를 함께 읽어야 하며 현재 운영 DTW에는 연결하지 않았다.

실제 비교 범위
공개 7영상·4명·5표현·812프레임의 57개 빈 손칸을 같은 저장 예측으로 비교했다. 새 HaMeR 추론이나 TCN 학습은 하지 않았다. 57은 독립 영상이나 독립 성공 횟수가 아닌 프레임별 손칸 수이다.
HaMeR 보완 추가: 기존33 → 중복 검사 후26. 시간 검사까지 적용한 보완 추가: 기존31 → 중복 검사 후24. 줄어든7은 모두 후원 REAL01의77,79~84프레임으로, 같은 겹침 구간의 중복 추가를 보류한 것이다. 정확한 복원 성공률이나 TCN 인식률 변화는 아니다.
기존 유효 관절14,149점, 지정 교차20프레임840점, 정상 대조94프레임의 결과가 유지됐다. 주먹과 펼친 손이 정상적으로 겹치는 두 대표 장면에서도 기존 보완을 유지했다.

넓은 정상 사례 점검
공개100단어에서 TRAIN촬영자3명씩300영상의 저장된 전체 화면 관절을 검사했다. 총36,869프레임 중 양손 좌표가 모두 있는34,466쌍을 검사했다. 300영상에 HaMeR를 새로 돌린 것이 아니라, 정상 양손을 잘못 중복으로 볼 위험을 확인한 좌표쌍 대조 시험이다.
첫 후보V1은 서로 닿는 두 주먹(독재자)도 의심했다. 원본을 확인하고 첫 결과를 보존한 뒤 손끝 일대일 대응을 추가한V2에서는 이 장면을 유지했다. 임계값을 바꾸거나 단어별 예외를 만들지 않았다.
남은1프레임(추출)은 실제 두 손이 교차하지만 기존 양손 좌표가 같은 앞손을 따라가는 것으로 보인다. 중복/혼합 추정 가능성과 정상 교차에서의 과도한 보류 가능성을 완전히 구분하지 못해 불확실로 남겼다. 전체34,466쌍의 RGB정답을 독립 주석한 것이 아니므로1건을 오류율로 환산하지 않는다. V2도 같은 자료로 개발·점검했으므로 독립 최종 평가가 아니다.

남은 문제와 다음 한 작업
1. 중복으로 세는 것은 막았지만 기존 손의 좌우 배정·손목 위치가 틀린 경우 어느 후보를 채택할지는 미해결이다. 현재 영상의 팔 연결과 앞뒤 관측으로 두 후보의 신원을 다시 확인하되, 확실하지 않으면 위치와 손가락 모양의 비교 근거를 분리해 보류하는 것이 다음 단계다.
2. 손가락이 전부 가려졌거나 점이 뭉개지면 이 검사가 중복을 놓칠 수도 있다. 검사 통과가 정확한 손 모양이라는 뜻은 아니다.
3. 기존 E02/E06 보류 장면과 E15 손목 오차는 이번 중복 검사로 해결하지 않았다.
4. 실제 사용자 환경·다른 단어 전체의 이득, 손가락 각도 정확도, TCN 인식률 개선은 아직 입증하지 않았다.

검증과 보존
구현·실행·원본 시각 확인: Codex. 별도 스칼라 계산으로417후보-기존손쌍 판정과34,466공개쌍을 검산하고 저장 배열·해시를 확인했다. 같은 담당자의 별도 구현 검산으로, 독립 전문가의 정답 검증은 아니다.
V1자체11항목·V2자체9항목 통과. 보호32파일과 과거 결과564파일의 해시 동일. 첫 검산은 정상 대조의 role이름을 잘못 읽어94프레임 대신0으로 집계해 실패했으며, 검산기 이름만 수정했다. 최초 오류 기록과 V1/V2결과는 모두 보존했다.
C·T3-F, 운영 TCN·DTW·웹은 변경하지 않았다. 사용자·멘토 촬영 열람/학습/튜닝/후보선택 없음. 추가학습·새추론·외부전송·Git작업 없음.
HTML 구조·이미지·로컬 링크를 점검했으며 브라우저 렌더링은 확인하지 않았다. 시각 확대는 설명용이며 원본 전체 화면 검출→통합→고정 상반신 범위 제한의 처리 순서는 유지한다.
'''
(U/'RESULT_KO.txt').write_text(body,encoding='utf-8')
def image_block(id,caption):
 x=next(i for i in review if i['id']==id)
 return f'<figure><a href="source_frames/{Path(x["source"]).name}"><img loading="lazy" src="views/{Path(x["image"]).name}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'
hero=image_block('NIA_SL_WORD0185_REAL01_F_0082','후원 · 82프레임. 왼쪽 원본 / 가운데 이전 통합 / 오른쪽 새 중복 검사. 주황색 새 추가만 보류했다. 남은 청록색의 잘못된 좌우 칸·손목 위치는 아직 고치지 않았다.')
normal=image_block('NIA_SL_WORD0185_REAL01_F_0066','후원 · 66프레임. 주먹과 펼친 손은 서로 다른 손이다. 이런 정상 겹침은 새 규칙이 막지 않았다.')
normal2=image_block('NIA_SL_WORD0185_REAL03_F_0090','다른 촬영자 · 90프레임. 정상 겹침의 기존 보완 결과 유지.')
proxy1=image_block('PROXY1_NIA_SL_WORD0429_REAL01_F_0059','독재자 · 정상 두 주먹의 접촉. 첫 후보에서는 의심했으나 손끝 일대일 대응을 추가한 최종 후보에서는 유지했다.')
proxy2=image_block('PROXY2_NIA_SL_WORD0920_REAL02_F_0053','추출 · 미해결 사례. 실제 두 손이 있으나 저장된 양손 좌표가 앞손을 함께 따라가는 것으로 보인다. 손가락 신원은 불확실로 남겼다.')
details=''.join(image_block(x['id'],f'{x["label"]} · {x["frame"]}프레임. '+x['finding']) for x in review if x['type']=='changed' and x['frame']!=82)
holds=''.join(image_block(x['id'],f'{x["label"]} · {x["frame"]}프레임. '+x['finding']) for x in review if x['category']=='previous_hold_unchanged')
page=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>같은 손 중복 추가 방지 · 공개 검증</title><style>
:root{{color-scheme:dark;font-family:"Malgun Gothic",system-ui,sans-serif;background:#0d1724;color:#eaf0f7}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1180px;margin:auto;padding:48px 24px 80px}}h1{{font-size:38px;line-height:1.35;letter-spacing:-1.4px}}h2{{font-size:26px;margin:0 0 20px}}h3{{font-size:19px}}p,li{{line-height:1.85}}section{{margin:36px 0;padding:28px;background:#152235;border:1px solid #2a3b51;border-radius:18px}}.eyebrow{{font-size:14px;color:#66d8cf}}.lead{{font-size:20px;max-width:950px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{padding:22px;border:1px solid #345066;border-radius:14px;background:#172b3b}}.big{{font-size:36px;font-weight:750;display:block;color:#65e0c7}}.note{{padding:16px 20px;border-left:4px solid #f2b459;background:#302a24;line-height:1.8}}.steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.step{{background:#203147;padding:16px;border-radius:12px;line-height:1.8}}figure{{margin:24px 0}}img{{display:block;width:100%;border-radius:10px}}figcaption{{color:#c6d2e2;line-height:1.8;margin-top:10px}}a{{color:#76dadb}}table{{width:100%;border-collapse:collapse}}td,th{{padding:15px 12px;text-align:left;border-bottom:1px solid #3b4a60;line-height:1.6}}th{{color:#a9c1d4}}details{{margin-top:18px}}summary{{cursor:pointer;color:#7bdfd2;padding:10px 0}}.small{{font-size:14px;color:#b5c4d7}}@media(max-width:760px){{main{{padding:24px 15px}}h1{{font-size:28px}}.cards,.steps{{grid-template-columns:1fr}}section{{padding:20px}}}}
</style></head><body><main><div class="eyebrow">WordsInSilence · 공개 영상 검증 · 2026.10.25 · 운영 미적용</div><h1>같은 손을 두 번 추가하던 문제를<br>먼저 막았습니다.</h1><p class="lead">손목이 다른 팔에 연결됐다고 나와도, 여러 손가락이 같은 실제 손을 가리키면 <b>새 손의 추가를 보류</b>하도록 보완했습니다.</p>
<div class="cards"><div class="card"><span class="big">7프레임</span>같은 손의 중복 추가 차단<br><span class="small">한 겹침 구간의 연속 장면</span></div><div class="card"><span class="big">20프레임</span>지정 정상 교차 결과 유지<br><span class="small">기존 관절 840점 보존</span></div><div class="card"><span class="big">300영상</span>추가 좌표쌍 대조 점검<br><span class="small">100단어 × TRAIN촬영자3명</span></div></div>
<section><h2>무엇이 문제였고, 어떻게 바꿨나</h2><p>이전에는 손목이 서로 다른 팔에 연결됐다고 나오면 중복 검사를 건너뛰었습니다. 그런데 손목을 잘못 잡으면 <b>펼친 손 하나에 두 뼈대가 동시에 붙는 일</b>이 생겼습니다.</p><div class="steps"><div class="step"><b>① 원본 좌표 비교</b><br>기존 손과 새 HaMeR 손을 같은 전체 화면 좌표에서 비교</div><div class="step"><b>② 여러 손가락 확인</b><br>손목 하나 대신 손가락 마디와 끝을 함께 확인</div><div class="step"><b>③ 가까운 두 손 구별</b><br>엄지·검지 등이 서로 일대일로 대응하는지 확인</div><div class="step"><b>④ 충돌은 따로 보류</b><br>기존 좌표와 새 원자료를 남기고 새 손의 추가만 보류</div></div><p>단어명이나 촬영자별 예외를 사용하지 않습니다. 같은 손인지 <b>확정</b>하는 기능이 아니라, 두 손으로 세기에는 근거가 충돌하는지 확인하는 후보입니다.</p></section>
<section><h2>실제 바뀐 장면</h2>{hero}<div class="note"><b>중복 방지 ≠ 잘못된 기존 손의 교정.</b> 남은 청록색은 왼손 칸의 기존 값이며 정답 표시가 아닙니다. 이 장면의 잘못된 좌우 배정·손목 위치와 가려진 주먹의 복원은 아직 해결하지 않았습니다. 관련 양쪽 손칸에는 불확실 상태를 별도로 기록했습니다.</div><p class="small">주황색: 오른손 칸 / 청록색: 왼손 칸. 색은 정확도 표시가 아닙니다. 화면 확대는 설명용입니다. 계산은 전체 영상 좌표에서 수행한 뒤 고정된 상반신 범위만 사용합니다. 이미지를 누르면 원본 전체 장면을 볼 수 있습니다.</p><details><summary>나머지 6개 변경 프레임 보기</summary>{details}</details></section>
<section><h2>정상 손과 교차는 유지됐나</h2>{normal}{normal2}<table><thead><tr><th>확인 항목</th><th>결과</th></tr></thead><tbody><tr><td>기존 유효 관절</td><td>14,149점 그대로 유지</td></tr><tr><td>지정 교차 장면</td><td>20프레임 · 840점 및 보완 결과 유지</td></tr><tr><td>정상 대조 영상</td><td>94프레임의 기존 결과 유지</td></tr></tbody></table><p>이는 이번 검사로 기존 정상 값을 새로 훼손하지 않았다는 뜻입니다. 모든 관절이 정확하거나 모든 교차가 해결됐다는 뜻은 아닙니다.</p></section>
<section><h2>추가로 넓혀 보니, 정상 접촉을 막는 문제가 나왔습니다</h2><p>100단어·300영상의 36,869프레임 중 양손 좌표가 있는 <b>34,466쌍</b>을 검사했습니다. 첫 후보에서 의심한 두 장면을 원본으로 확인한 뒤, 가까운 손끝을 같은 손으로 오해하지 않도록 일대일 대응 조건을 추가했습니다.</p>{proxy1}{proxy2}<div class="note">300영상에 HaMeR를 새로 실행한 것은 아닙니다. 저장된 MediaPipe 양손 좌표쌍으로 정상 손을 잘못 막을 위험을 살펴본 검사입니다. 남은 의심1건을 오류율로 환산할 수 없으며, 새로운 자료에서의 최종 성능 검증도 남아 있습니다.</div></section>
<section><h2>추정 추가 개수가 줄어든 이유</h2><table><thead><tr><th>공개 57개 빈 손칸</th><th>이전</th><th>보완 후</th></tr></thead><tbody><tr><td>HaMeR 보완 추가</td><td>33</td><td>26</td></tr><tr><td>시간 검사까지 통과한 보완 추가</td><td>31</td><td>24</td></tr></tbody></table><p>줄어든7개는 같은 손을 두 번 넣던 경우입니다. 57개는 프레임별 손칸의 수이며, 복원 성공률이나 TCN 인식률이 아닙니다. 전체 비교 범위는 공개7영상·4명·5표현·812프레임입니다.</p></section>
<section><h2>남은 문제와 다음 한 작업</h2><ol><li><b>기존 값이 틀리고 새 값이 더 나을 수도 있습니다.</b> 현재는 충돌을 찾고 추가만 보류합니다. 다음에는 현재 영상의 팔 연결과 앞뒤 관측을 이용해 어느 손인지 다시 확인해야 합니다. 근거가 부족하면 좌우를 억지로 정하지 않습니다.</li><li><b>가려진 손가락의 정확한 모양은 미해결입니다.</b> 손끝이 안 보이거나 여러 점이 뭉개지면 중복을 놓칠 수 있습니다. 검사 통과를 정확한 손 모양으로 취급하지 않습니다.</li><li><b>이전부터 보류된 손과 손목 오차가 남아 있습니다.</b> E02/E06의 과도 보류 가능성과 E15 손목 오차는 이번 검사로 고치지 않았습니다.</li></ol><details><summary>이전부터 보류돼 있던 두 장면</summary>{holds}</details><p>새 후보는 연구 결과로 보관했습니다. <b>C·T3-F와 운영 TCN·DTW·웹은 그대로</b>이며, 사용자 촬영은 학습·튜닝·후보 선택에 사용하지 않았습니다.</p></section>
<section><h2>검증 근거</h2><p>별도 계산식으로417개 후보-기존손쌍의 판정과34,466공개쌍을 검산했습니다. 기존·새 배열, 시간, 범위, 보호32파일과 과거 결과564파일의 해시도 확인했습니다. 구현과 검산 담당은 같은 Codex이며, 독립 전문가의 정답 주석이나 일반화 성능 평가를 대신하지 않습니다.</p><p class="small">첫 후보·정상 접촉에서의 문제·최종 후보를 모두 보존했습니다. 최초 검산의 정상 대조 역할 이름 오기만 검산기에서 바로잡았으며 입력·후보 결과는 바꾸지 않았습니다. HTML·이미지·로컬 링크 정적 확인, 브라우저 렌더링 미확인.</p><p><a href="RESULT_KO.txt">전체 설명</a> · <a href="RESULT_V2.json">최종 수치</a> · <a href="VERIFICATION.json">별도 검산</a> · <a href="VISUAL_REVIEW.json">장면 검토 기록</a> · <a href="FROZEN_V2.json">조건 변경 근거</a></p></section></main></body></html>'''
(U/'RESULT_KO.html').write_text(page,encoding='utf-8')
write(U/'COMPLETION.json',{'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'status':'CANDIDATE_VERIFIED_DUPLICATE_ADDITION_GUARD_ONLY_NOT_DEPLOYED','implementation':'Codex','verification':'Codex separate scalar implementation, saved-array/hash checks and image review; not expert GT','primary':r['primary'],'broad_proxy':{k:r['broad_proxy'][k] for k in ['clips','words','signers','frames','complete_pairs','flagged']},'existing_identity_and_wrist_error_unresolved':True,'broader_overlap_unresolved':1,'training':0,'new_inference':0,'production_change':False,'user_data_access':False,'report':str(U/'RESULT_KO.html'),'next':'Resolve existing-vs-new identity conflicts using current image arm continuity and temporal evidence, preserve both hypotheses, do not force hidden finger identity; no deployment/training yet.'})
print('Report and visual review saved')
