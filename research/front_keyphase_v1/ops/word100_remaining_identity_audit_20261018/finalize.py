from audit import *
import html,re,math

CAUSES={
23:('다른 손으로 관측이 바뀜','f34~36은 왼손이 이어지지만 f37에는 다른 오른손만 잡힙니다. raw0라는 번호가 같아도 같은 손이라는 뜻은 아닙니다.'),
76:('손이 2점만 보임','화면 안 관절이21점 중2점, 손바닥 기준점은1점입니다. 처음 관측을 다음 손과 잇는 충분한 근거가 없습니다.'),
143:('짧은 확인 시간 안에 근거 부족','같은 손의 연결은56~59프레임까지 이어지지만100ms 안에 현재 팔 연결이 두 번 확인되지 않습니다. 뒤의61프레임에 팔 근거가 생겨도 이 시간 조건을 소급 완화하지 않았습니다.'),
200:('잠깐 잡힌 뒤 검출이 끊김','f0에서 경계 위8점이 잡히고 f1부터 손 관측이 사라집니다. 나중에 나온 손과 연결할 근거를 만들지 않습니다.'),
207:('다른 손으로 관측이 바뀜','f20의 왼손 다음 f21에는 다른 오른손이 잡힙니다. 가까운 시간에 잡혔다는 이유만으로 같은 손으로 이어 붙이면 안 됩니다.'),
264:('손이 2점만 보임','첫 관측은21점 중2점만 화면 안입니다. 이후 부분 손·양손·검출 없음이 섞여 있고 최초 관측과 안정적인 손 연결이 없습니다.'),
272:('손이 2점만 보임','첫 관측은21점 중2점이고 f41에는 손이 잡히지 않습니다. 이를 왼손 또는 오른손의 확정된 이력으로 삼을 수 없습니다.'),
309:('같은 손의 중복 후보','같은 실제 오른손에 두 손 후보가 겹쳐 잡힙니다. 시간 연결도1→2→1로 갈라져 어느 후보를 믿을지 독립 근거가 부족합니다.'),
409:('잠깐 잡힌 뒤 검출이 끊김','f4에서 경계 위7점이 잠깐 잡히고 다음 프레임부터 사라집니다. 장시간 후의 손으로 소급 배정하지 않습니다.')}

def main():
    assert read(U/'INDEPENDENT.json')['status']=='PASS' and read(U/'CONTRACTS.json')['status']=='PASS'
    c=read(U/'CONTEXT.json');check(c['pins']);r=read(U/'REMAINING9.json');ann=read(U/'ANNOTATED.json');paired=read(U/'PAIRED_VIEW_DIAGNOSIS.json');rows={q['clip_id']:q for q in c['rows']}
    # Independent arithmetic for the additional descriptive view comparison.
    for x in paired:
        row=rows[x['clip_id']];n=load(row['views']['masked']['raw']);z=load(row['views']['sensor']['raw']);f=x['frame'];j=[q for q in r if q['id']==x['id']][0]['raw_index'];t=int(np.flatnonzero(n['frame_id']==z['frame_id'][f])[0]);b=load(P/'results'/f"{x['clip_id']}_masked.npz");a=load(P/'results'/f"{x['clip_id']}_sensor.npz")
        for q in x['native_candidates']:
            k=q['raw_index'];idx=[v for v in range(21) if a['point_mask'][f,j,v] and b['point_mask'][t,k,v]];dist=sum(math.hypot(float(z['hand_xy_px'][f,j,v,0])-float(n['hand_xy_px'][t,k,v,0]),float(z['hand_xy_px'][f,j,v,1])-float(n['hand_xy_px'][t,k,v,1])) for v in idx)/len(idx)
            assert math.isclose(dist,q['mean_distance_px'],rel_tol=1e-12,abs_tol=1e-12)
        target=min(x['native_candidates'],key=lambda q:q['mean_distance_px']);truth=next(y['actual_side'] for y in r if y['id']==x['id']);assert target['identity_side']==truth
        assert x['lead_before_annotated_core_ms']>0
    write(U/'PAIRED_VIEW_CHECK.json',{'status':'PASS','cases':9,'nearest_native_evidenced_side_matches_prior_visual_side':9,'no_transfer_into_sensor':True,'note':'Distance matching is descriptive. Partial2point cases are not treated as sufficient physical identity proof.'})
    review=[]
    for x in r:review.append({'id':x['id'],'word':x['word'],'category':CAUSES[x['id']][0],'finding':CAUSES[x['id']][1],'image':str(U/f"evidence/remaining_{x['id']}.jpg"),'status':'original sequence and existing visual annotation agree; unresolved'})
    visual={'reviewer':'Same Codex, not independent human truth','remaining9':review,'elbow_counterexamples':[{'id':x['id'],'word':x['word'],'basis':'Crossed/overlapping hands bring the anatomical right hand near the opposite elbow. Existing correct right assignment must not be overwritten by proximity.','image':str(U/f"evidence/elbow_counterexample_{x['id']}.jpg")} for x in ann if x['verdict']=='normal_identity' and x['elbow_only'] is not None and x['elbow_only']['side']!=x['actual_side']],
        'delayed_changed_hint_review':{'clip_id':'NIA_SL_WORD0920_REAL03_F','view':'masked','index':72,'old_hint':0,'new_hypothesis':1,'available100ms_later':True,'verdict':'NOT_COUNTED_AS_CORRECTED: hands overlap in original RGB; separate physical identity/landmark mixture ground truth not independently established. Anchors/coordinates are machine evidence, not proof.','image':str(U/'evidence/delay_detail.jpg')},'all15_diagnostic_sequences_inspected':True,'223_delayed_proposals_all_visually_verified':False}
    write(U/'VISUAL_REVIEW.json',visual)
    at=datetime.now(timezone(timedelta(hours=9))).isoformat();summary=read(U/'SUMMARY.json')
    next_plan='残存9건 원인·위험한 팔꿈치근접규칙(정상5회귀)·100ms지연확인한계를 사용자와 검토한다. 9건 해결로 보고하거나 지연후보223건을 정답으로 간주하지 않는다. 기존C/웹/원본/사용자자료 유지·학습대기. 공통방향은 관측좌표를 보존하면서 신체좌우와 물리손연결을 분리하고, 연결이 끊기거나 중복이면 미확정상태를 유지하는 것. 추가 확정 개선은 공개TRAIN의 부분손·교차 통제표본에서 화면 안 팔-손 영상연결 등 새로운 독립근거를 먼저 검증해야 한다. 결과맞춤 시간·신뢰 임계완화나9건 단어예외 금지. 학습 입력 연결/기존C비교는 결과 확인 후 별도 조건 동결. 완료전수감사 재실행불필요.'
    next_plan=next_plan.replace('残存','남은')
    result={**summary,'status':'VALIDATED_DIAGNOSTIC_NO_ADDITIONAL_SAFE_RECOVERY_AWAITING_REVIEW','at':at,'remaining9_recovered':0,'remaining_wrong_hints':9,'remaining_points_preserved':72,'causes':dict(Counter(x['category'] for x in review)),'native_same_frame_diagnostic':9,'delayed_candidate_adopted':False,'elbow_rule_adopted':False,'new_verified_corrected':0,'protected_pins':len(c['pins']),'independent':'PASS','production_changed':False,'training_authorized_now':False,'result':str(U/'RESULT_KO.html'),'next_plan':next_plan}
    write(U/'RESULT.json',result);write(U/'NEXT_PLAN.json',{'next_single_action':next_plan,'training':False,'production_change':False})
    report='''남은9건 좌우 미확정 원인과 공통 보완 방법 검토

결론
공개 TRAIN899영상의 기존2조건 관측을 검사했다. 남은9건을 정상 손 손상 없이 새로 확정할 추가 규칙은 이번 검증에서 확인하지 못했다.9건의 관측72점과 미확정 상태를 그대로 보존했다. 이9건은 손/프레임 진단 사건이며9개 수어 단어의 인식 실패율이 아니다. 기존 C의 단어 분류 결과도 이번에는 새로 평가하지 않았다.

무엇이 원인이었나
9건 모두 화면 아래가 실제로 잘린 sensor조건이며, 같은 순간 양쪽 Pose손목 신뢰가0.300~0.495로 기존0.5기준보다 낮다. Hand의 보이는 좌표는 남아 있어도 그 손과 어느 팔이 연결됐는지 확인할 근거가 부족하다.
-3건: 관절21점 중2점만 화면 안에 남음(독재자·강진군주민지원센터·진도군수어통역센터).
-2건: 경계의 손이 잠깐 잡힌 후 다음 관측이 끊김(유성구주민지원센터·칠천).
-2건: 다음 프레임의 raw0가 같은 손이 아니라 다른 손으로 바뀜(가래떡·가평군주민지원센터).
-1건: 손의 연결은 있으나 고정한100ms 안에 팔 연결 근거가 충분히 쌓이지 않음(추출).
-1건: 같은 실제 오른손을 두 후보로 중복 검출하여 시간 연결이 갈라짐(공항).
이는 상세 관찰에 따른 주된 원인 분류이며, 일부 사례는 여러 문제가 함께 있다. 모두 주석의 핵심 동작 시작 전이지만 준비 구간도 모델 입력/추적에 영향을 줄 수 있어 무시하거나 삭제하지 않았다.

전체 영상과 비교한 결과
동일 시각의 전체 화면 검출에서는9건 모두 기존 시각검토의 실제 좌우와 맞는 확정 근거 후보가 있다. 전체 화면에서의 양쪽 Pose손목 신뢰는 각 사례 모두0.5이상이다. 화면 아래를 자른 조건은 손을 일부 잃고 팔 검출도 달라진다. 계속 검출을 켜 두는 것만으로 관측 범위 밖의 정보를 확보할 수는 없다.
다만 전체 영상에서 보이는 화면 밖 정보를 실제 웹캠 입력에 정답처럼 옮기면 비교가 부정확해진다. 전체 영상은 원인 대조용으로만 사용했고 좌표·좌우를 sensor자료에 복사하지 않았다. 두 점만 남은 경우의 거리 대응도 확정 추적 근거로 쓰지 않았다.

방법1: 가까운 팔꿈치로 배정하기 — 미채택
남은9건만 보면9건 모두 맞는 방향을 가리킨다. 그러나 원래 정상441건에 같은 규칙을 적용하면5건의 정상 오른손을 왼손으로 잘못 바꾼다. 후원3프레임·소복1프레임·무산1프레임이다. 손을 교차하거나 겹치는 동작에서는 자기 팔꿈치보다 반대 팔꿈치에 가까워질 수 있다.9건을 없애려고 이 규칙을 적용하면 정상 손을 해친다. 특정 단어 예외나 임계값 재탐색 없이 미채택했다.

방법2: 잠깐 기다려 같은 손인지 확인하기 — 진단 구현, 운영 미채택
최대100ms 안에 같은 손의 관측이 하나의 경로로 이어지고, 현재 팔 연결 근거가 두 프레임 연속 확인될 때만 앞의 미확정 관측에 별도 지연 좌우 제안을 붙이는 공통 검사기를 구현했다. 검출 없음·연결 갈림·중복·기존 근거 충돌에서는 멈춘다. 좌우 미확정인 손의 좌표는 버리지 않는다. 미래 프레임을 쓴 결과이므로 원래 시점에 알았던 정보로 표시하지 않는다.
전체305602손/프레임 중223건에서 기계적 조건이 충족됐다.222건은 기존 잠정 방향과 같고1건은 달라졌다. 이것은223건의 실제 정답 확인이나 수어 인식 개선이 아니다. 남은9건의 안전한 정정은0건이었다.
새로 방향을 바꾸려는1건은 추출 REAL03 전체화면 f72이며100ms뒤 근거를 쓴다. 원본에서 양손이 겹쳐 있어 어느 물리손의 관절인지 독립적으로 확정하지 못했으므로 개선 성공으로 세지 않았다.223건 전체에 대한 사람 시각정답 검증도 완료된 것이 아니다. 운영이나 학습 입력에는 연결하지 않았다.
알려진 정상441건과 교차315장면에서 지연 제안으로 기존 좌우를 뒤집는 결과는0건이었다. 이는 이 통제표본의 결과이며 전체 미확정 손 모두 안전하다는 보장은 아니다.

공통 보완 방향
1. 손을 관측했다는 사실, 같은 물리손으로 이어진다는 근거, 왼손/오른손이라는 근거를 별개로 보존한다. raw0/raw1은 손의 고유번호가 아니다.
2. 새로 나타난 부분 손에는 좌우를 즉시 강제하지 않는다. 실제 관측한 좌표·모양 정보는 남기며, 빈 구간을 가짜 관절로 채우지 않는다.
3. 연결이 분명할 때만 이전 관측을 재확인한다. 다른 손으로 바뀌거나 후보가 합쳐지고 갈라지면 기존 확정 이력을 복사하지 않는다.
4. 추가로9건을 확정하려면 현재 좌표의 신뢰 기준을 낮추기보다 화면 안 팔-손의 영상 연결처럼 독립적인 근거를 공개 자료에서 검증하는 편이 타당하다. 이는 다음 제안이며 RGB추적·새 모델을 이번에 구현하거나 학습한 것은 아니다.
현재로서는9건을 억지로0건으로 만드는 것보다 정상 손과 관측을 보존하는 편이 낫다. 불확실 정보를 최종 학습 입력에 어떻게 연결할지는 별도 설계·비교가 필요하며 마스크/채널을 추가하면 인식률이 오른다고 가정하지 않는다.

검증과 한계
226362프레임/305602손후보의 새 지연검사를 완료했고, 별도 scalar 검사기로 모든 제안과 관측수·통제표본·팔꿈치 계산을 검산했다. 원래6292471관절 좌표와 기존 결과는 읽기 전용으로 유지했다. 합성 계약16항목 통과,보호23467파일 해시 일치. 구현·원본 시각검토는 같은 Codex, 수치 검산은 별도 코드다. 새 사람 시험이나 독립 사람의 정답평가가 아니다.
기존 C/TCN/DTW/웹 및 사용자 촬영은 변경하지 않았다. 사용자·멘토·DEV·보류 자료 읽기/학습/튜닝0, 새 학습/분류추론0. 현재 실행 중 프로세스 없음. 결과를 함께 검토한 뒤 다음 단계와 학습 조건을 정한다.
'''
    (U/'RESULT_KO.txt').write_text(report,encoding='utf8')
    table=''.join(f'<tr><td>{html.escape(x["word"])}</td><td>{x["all_observed_points"]}/21</td><td>{CAUSES[x["id"]][0]}</td><td><a href="#case{x["id"]}">장면 보기</a></td></tr>' for x in r)
    cases=''.join(f'<details id="case{x["id"]}"><summary>{html.escape(x["word"])} · f{x["index"]} · {CAUSES[x["id"]][0]}</summary><p>{CAUSES[x["id"]][1]}</p><img loading="lazy" src="evidence/remaining_{x["id"]}.jpg" alt="{html.escape(x["word"])} 원본 시계열과 관측"></details>' for x in r)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>남은9건: 좌우 근거가 끊긴 이유</title><style>body{margin:0;background:#eff3f7;color:#20354c;font-family:Malgun Gothic,sans-serif;line-height:1.8}main{max-width:1120px;margin:auto;padding:38px 24px}h1{font-size:32px;line-height:1.4}h2{font-size:24px}section{background:white;border-radius:14px;padding:24px;margin:20px 0}.lead{font-size:20px;font-weight:700}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{background:#e5f1f0;padding:20px;border-radius:12px}.num{font-size:32px;font-weight:700;color:#146a61}.warn{background:#fff5df;border-left:5px solid #bd8428}table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #d8e2eb;text-align:left}a{color:#14669b}details{border-top:1px solid #d8e2eb;padding:15px 0}summary{cursor:pointer;font-weight:700}img{width:100%;height:auto}.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.flow div{padding:18px;background:#eef3f7;border-radius:10px}small{display:block;color:#52667c}@media(max-width:760px){.cards,.flow{grid-template-columns:1fr}table{font-size:13px}}</style><main>
<h1>남은9건, 왜 좌우를 확정하지 못했을까?</h1><p class="lead">더 자세히 확인했지만, 정상 손을 해치지 않고9건을 모두 고칠 규칙은 아직 확인하지 못했습니다.</p><p>공개 학습용 자료의 손/프레임 진단입니다.9개 수어의 인식 실패율을 뜻하지 않습니다. 기존 C 모델의 성능은 이번에 다시 평가하지 않았습니다.</p>
<div class="cards"><div class="card"><div class="num">0 / 9</div>추가로 안전하게 확정한 사례</div><div class="card"><div class="num">72점 유지</div>남은9건에서 관측한 관절</div><div class="card"><div class="num">기존 C 유지</div>새 학습·사용자자료 이용 없음</div></div>
<section><h2>공통점: 화면 경계에서 손과 팔의 근거가 약해졌습니다</h2><p>9건 모두 아래가 잘린 웹캠 모사 조건에서 발생했습니다. 같은 시각의 전체 화면에는 실제 좌우와 맞는 근거가 있었지만, 잘린 화면에서는 손 일부가 사라지고 팔의 손목 연결 신뢰도 낮아졌습니다.</p><p>검출을 처음부터 계속 실행해도 화면 밖의 정보까지 알 수는 없습니다. 전체 영상의 숨겨진 정보를 웹캠 자료에 복사하지 않았습니다.</p><table><thead><tr><th>단어</th><th>보이는 관절</th><th>주된 원인</th><th>근거</th></tr></thead><tbody>'''+table+'''</tbody></table></section>
<section class="warn"><h2>가까운 팔꿈치로 정하면 해결될까?</h2><p class="lead">남은9건은 맞지만, 원래 맞던441건 중5건을 틀리게 만듭니다.</p><p>교차 동작에서는 오른손이 왼쪽 팔꿈치 가까이에 갈 수 있습니다. 아래 소복 장면에서도 가까운 팔꿈치만 따라가면 정상 오른손이 왼손으로 바뀝니다. 이 규칙은 적용하지 않았습니다.</p><img src="evidence/elbow_counterexample_90.jpg" alt="소복 정상 오른손이 반대쪽 팔꿈치에 가까워지는 반례"></section>
<section><h2>0.1초 기다려 다시 확인하는 방법도 검사했습니다</h2><div class="flow"><div><b>손을 보존</b><p>지금 보이는 점은 그대로 남깁니다.</p></div><div><b>같은 손인지 확인</b><p>한 경로로 이어지고 팔 근거가 연속으로 잡히는지 봅니다.</p></div><div><b>불명확하면 미확정</b><p>사라짐·다른 손·중복이 끼면 소급 확정하지 않습니다.</p></div></div><p>전체 검사에서223건이 지연 제안 조건을 만족했지만, 남은9건을 확정한 경우는0건입니다.223건은 정답으로 검증된 수가 아닙니다.1건은 기존 방향을 바꾸려 하지만 양손이 겹쳐 있어 성공으로 세지 않았습니다.</p><p>정상441건과 교차315장면의 기존 배정을 뒤집는 제안은0건이었습니다. 실제 모델이나 웹에는 연결하지 않았습니다.</p></section>
<section><h2>각 사례의 실제 앞뒤 장면</h2><p>이 그림은 이해를 돕기 위해 원본 전체를 보여줍니다. 노란 선 아래는 sensor검출에 입력되지 않았습니다. 녹색/파란색은 raw후보0/1이며 좌우 손 이름이 아닙니다. 표기의0=왼손,1=오른손,-1=배정 보류입니다.</p>'''+cases+'''</section>
<section><h2>이후 보완 방향</h2><p>관측한 손은 보존하고, 손의 연결과 좌우 확실성을 분리합니다. 처음 보인 부분 손이나 중복 후보에 좌우를 강제하지 않습니다. 끊긴 추적은 다른 손으로 자동 이어 붙이지 않습니다.</p><p>남은9건을 더 확정하려면 화면 안 팔과 손의 실제 영상 연결 등 추가 근거를 먼저 검증하는 것이 다음 후보입니다. 가까운 팔꿈치 규칙을 붙이거나 시간·신뢰 기준을 낮추는 방법은 이번 결과만으로 채택하지 않습니다.</p><p>결과를 함께 확인한 뒤 학습 입력 연결과 기존 C 비교 조건을 정합니다. 관측 구조 보완이 곧 인식률 향상인 것은 아닙니다.</p></section>
<section><h2>검증 범위와 보존</h2><p>공개 TRAIN899영상 ×2조건 ·226,362프레임 ·305,602손 후보. 별도 수치 검산 통과, 기존 관절6,292,471점 보존. 새 검사16항목 통과.</p><p>알려진 공개 자료를 사용한 개발 진단이며, 원본 시각검토는 같은 Codex가 수행했습니다. 전체223지연 제안의 사람 정답 검증이나 새 사용자 평가는 아닙니다.</p><p>기존 C·웹·TCN·DTW·사용자 촬영 유지. 학습·튜닝·분류 추론 없음.</p><p><a href="RESULT_KO.txt">상세 보고</a> · <a href="INDEPENDENT.json">별도 수치 검산</a> · <a href="PAIRED_VIEW_DIAGNOSIS.json">전체/잘린 화면 대조</a> · <a href="VISUAL_REVIEW.json">원본 검토 기록</a></p></section></main></html>'''
    (U/'RESULT_KO.html').write_text(page,encoding='utf8')
    for target in re.findall(r'(?:src|href)="([^"]+)"',page):
        if not target.startswith('#'):assert (U/target).exists(),target
    brief='남은9건 공통원인 심층감사완료. 공개TRAIN899×2조건226362프레임/305602손후보.9건 모두sensor조건·핵심주석전·양Pose손목신뢰<.5. 3부분2점/2관측끊김/2다른손전환/1시간내근거부족/1중복갈림. 전체화면동시각에는실제측근거가있지만sensor로복사하지않음. 최근접팔꿈치는9건맞지만정상441중5회귀→미채택. 100ms두현재팔근거+일대일연결지연진단223제안(222동일/1변경),정상441·교차315뒤집힘0,남은9해결0. 새변경1은겹침정답미확정으로개선집계안함;223전체시각정답검증아님. 관측72점/전체6292471점보존,기존배정/운영유지. 계약16·별도전수검산PASS·보호23467핀보존. C/사용자자료유지·새학습/분류/DEV/보류/사용자읽기0. 결과검토후추가독립근거또는학습연결조건결정.'
    paths=list(U.glob('*.py'))+list(U.glob('*.json'))+[U/'RESULT_KO.txt',U/'RESULT_KO.html']+list((U/'evidence').glob('*.jpg'))
    write(U/'FINAL_AUDIT.json',{'status':'PASS','pins':{str(p):sha(p) for p in paths if p.name!='FINAL_AUDIT.json'},'protected_pins':len(c['pins']),'all_links_exist':True,'new_side_arrays_applied':False,'training':0,'execution':'audit exec74976 exit0; verify exec21550 exit0; render exec43913 exit0; no remaining processes'})
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';ctx=p.read_text(encoding='utf8')
    for number,title,txt in [(8,'남은9건 원인·공통 보완 진단 완료',brief),(9,'남은9건 진단 검토·추가근거 및 학습조건 대기',next_plan)]:
        m=re.search(r'^## '+str(number)+r'\..*$',ctx,re.M);assert m;ctx=ctx[:m.end()]+'\n\n### 2026-10-18 '+title+'\nops/'+U.name+'. '+txt+'\n'+ctx[m.end():]
    p.write_text(ctx,encoding='utf8');state=read(O/'STATE.json');state['updated_at']=at;state['current_stage']='WORD100_REMAINING_IDENTITY_AUDIT_VALIDATED_AWAITING_REVIEW';state['next_single_action']=next_plan;state[U.name]=result;write(O/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {at}

### 현재 목표·마지막 완료
ops/{U.name}. {brief}

### 다음 단일 작업
{next_plan}

### 실제 작업과 근거
- audit.py: 공개 캐시의9건 원인/고정100ms 지연 진단/최근접팔꿈치 반례 분석. 기존 assignment/coordinates를 쓰거나 바꾸지 않는 별도 진단이다.
- CONTEXT.json: 실행전고정·23467기존핀. CONTRACTS16PASS. INDEPENDENT:226362프레임/305602후보 별도코드검산PASS.
- REMAINING9/ANNOTATED/PROPOSALS/CROSSING/PAIRED_VIEW_DIAGNOSIS/VISUAL_REVIEW 및15장면+겹침확대근거. RESULT_KO.html/txt·RESULT/NEXT_PLAN·FINAL_AUDIT.
- 구현·원본시각검토 동일Codex;수치검산 별도코드.223제안 전체의 사람정답검증이 아님. 남은9건정정0,정상441/교차315의지연제안회귀0,최근접팔꿈치규칙은정상5회귀로미채택.
- 이전 U 관측보존 후보는 그대로. 이번은 진단이며 지연확인/새좌우배정을운영이나학습에연결하지않음. C/TCN/DTW/웹 변경0. 사용자·멘토·DEV·보류읽기0,학습0,분류0.
- exec74976/21550/43913 모두종료0. 현재실행/잠금없음. st-gcn자동화삭제유지. 원본/W:/sources/과거결과읽기전용.
''',encoding='utf8')
    assert read(O/'STATE.json')[U.name]['remaining_wrong_hints']==9
    print('FINALIZED',str(U/'RESULT_KO.html'))
if __name__=='__main__':main()
