"""Record completed visual review and report, then synchronize current handoff."""
import json,hashlib,re,html
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    assert not Path(p).exists(),p
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
f=read(U/'FROZEN.json');v=read(U/'VERIFY.json');ex=read(U/'EXPORT_VERIFY.json')
fe=read(U/'FEATURE_EFFECT.json');targets=read(U/'REVIEW_TARGETS.json')
assert len(targets)==321 and {a['id'] for a in targets}==set(range(321))
assert v['clips']==36 and v['counts']['added_native_hand_frames']==300 and ex['status']=='PASS'
assert read(U/'CONTRACTS.json')['status']=='PASS'
assert not (U/'RUNNING.json').exists()
for p,h in {**f['pins'],**f['protected'],**ex['pins']}.items():assert sha(p)==h,p
for item in read(U/'INDEX.json'):
    assert sha(item['path'])==item['sha256']
    for p,h in item['outputs'].items():assert sha(p)==h,p

visual=dict(at=at,reviewer='Codex root; same implementer, not a separate human or agent',
    method='Viewed all 33 saved contact sheets in tool image output; source context, raw hand closeup, baseline/new overlay. Larger source evidence also viewed for targets243,252,255.',
    pages_viewed=list(range(1,34)),target_ids_viewed=list(range(321)),
    added_hand_frames_reviewed=300,identity_conflict_decisions_reviewed=21,
    result='No clear new duplicate physical hand or new left/right reversal identified in the 300 added hand-frame source comparisons. This is a visual screening result, not per-joint ground truth.',
    limitations=[
        'Finger joints obscured by own fingers, blur or the other hand cannot be established as anatomically correct from these images.',
        'Two-scale agreement uses the same image and model, so correlated errors remain possible.',
        'Existing source assignments are deliberately preserved, including any old errors. The21 diagnostic conflicts are not21 confirmed errors.',
        'Known normal3 and crossing53 are retained-input controls, not a population accuracy estimate.',
        'TRAIN36 known diagnostic clips, not all100 words or unseen users.'],
    detail_cautions=[
        dict(id=243,note='상처07 f39: hand region exists,16/21 points retained by agreement; precise folded finger shape remains uncertain. Do not label all21 visible or anatomically verified.'),
        dict(id=252,note='유성구주민지원센터09 f35: rotating/partially self-occluded hand. Regional support does not establish every finger joint.'),
        dict(id=255,note='유성구주민지원센터09 f40: two distinct crossing hands supported; underlying occluded palm/fingers remain estimated.')],
    research_gate='MERGE_AND_POST_ROI_VALIDATED_WITH_LIMITATIONS_PENDING_USER_REVIEW',
    automatic_training_ready=False,production_adopted=False)
write(U/'VISUAL_REVIEW.json',visual)
next_action=('전체검출→동일원본손통합→고정ROI 후보 결과를 사용자와 검토한다. '
    '다음 비교는 같은 전체검출 경로의 통합 전/후를 주대조로 하고, 기존 C·T3-F는 고정 참고로 유지한다. '
    '추가손으로 손크기 기준이 변한 영향(35/36영상,최대약7.2%)을 분리할 학습조건을 먼저 고정하고, '
    '21개 좌우 충돌 진단과 정밀 손가락 불확실성을 자동 정답으로 승격하지 않는다. '
    '센서 선크롭 자료 자동혼합 없음. 추가 학습은 사용자 결과 확인 후에만; 사용자/멘토촬영 학습·튜닝·선택 금지.')
audit=dict(at=at,status='RESEARCH_MERGE_AND_ROI_VALIDATED_PENDING_USER_REVIEW',
    scope=dict(public_train_clips=36,words=23,signers=9,frames=4641),
    flow='Full original video detection/assignment cache + fresh same-original-frame redetection; reconcile original coordinates; then fixed ROI and280 features.',
    primary_sensor_training_mix=False,verification=v,export_summary={k:x for k,x in ex.items() if k!='pins'},
    visual_review='VISUAL_REVIEW.json',feature_effect={k:x for k,x in fe.items() if k!='rows'},
    protected_files=len(f['protected']),model_weights_preserved=True,
    training_runs=0,classifier_evaluations=0,user_recording_reads=0,production_changes=0,
    source_redetection='Executed once for36 TRAIN videos; no replay/rerun of completed detector outputs.',
    implementation='Codex local',verification_role='Separate saved arithmetic implementation and source image review by same Codex; no separate reviewer.',
    recovery='Object-valued hand_scales metadata found on export read; original outputs retained, safe numeric NPZ+JSON exports verified without detection rerun.',
    evidence_files=['CONTRACTS.json','VERIFY.json','EXPORT_FIRST_ERROR.json','EXPORT_VERIFY.json','FEATURE_EFFECT.json','VISUAL_REVIEW.json'],
    next_single_action=next_action)
write(U/'FINAL_AUDIT.json',audit)

report="""전체 영상에서 손을 합친 뒤, 상반신 범위만 입력으로 사용하는 후보 검증

결론
요청한 순서로 별도 연구 후보를 구현하고 공개 TRAIN 36개 영상(23개 단어,9명,4,641프레임)에서 비교했다.
전체 원본 영상의 기존 손 검출 → 같은 원본 이미지에서 다시 찾은 손과 통합 → 고정 상반신 범위 밖 관절 제외 → 280개 입력 특징 생성.
기존 C·T3-F·웹·DTW는 그대로다. 학습0, 분류평가0, 사용자/멘토촬영 접근0.
이번 결과는 관측 입력의 보완 결과이며 수어 인식률 향상을 측정한 결과가 아니다.

무엇을 바꿨나
1. 전체 영상에서 처음부터 끝까지 추출해 둔 기존 관절과 좌우 배정을 그대로 사용했다.
2. 같은 원본 프레임을 다시 확인해 손 후보를 구했다. 빠진 쪽은 현재 팔/손목 근처의 작은 이미지로도 다시 확인했다.
   이 작은 이미지는 같은 프레임을 확대 확인하는 용도다. 상반신으로 먼저 자른 sensor 학습 자료를 섞는 것이 아니다.
3. 원본 좌표로 돌려놓은 뒤 같은 실제 손인지, 다른 팔에 연결된 별개 손인지 구분했다.
   같은 손을 다시 찾으면 확인 근거만 붙이고 손을 하나 더 만들지 않는다.
   기존/새 좌우 근거가 충돌하거나 현재 팔 근거가 불분명하면 새 손을 억지로 추가하지 않는다.
   화면의 왼쪽/오른쪽 반으로 나누지 않고 현재 팔 근거를 사용한다. 서로 다른 팔에 연결된 손은 교차해도 보존한다.
4. 기존에 있던 손은 덮어쓰지 않는다. 비어 있던 손에만 새 정보를 추가하고, 두 이미지 크기에서 일치하지 않는 관절은 미관측으로 남겼다.
5. 마지막에만 고정 상반신 경계를 적용한다. 경계 밖 좌표는 제거하고 관측 여부를0으로 둔다.
   영상의 프레임·원래 시간 간격은 그대로이며 중간 장면을 없애거나 추정 관절로 메우지 않았다.
   범위 밖 영상은 앞단의 좌우 판단에 도움이 될 수 있지만, 그 좌표 자체는 최종 입력에 포함하지 않는다.

같은 공개 영상에서 확인한 변화
- 새로 추가된 손·프레임:300. 한 손을10개 프레임에서 찾으면10건이라는 뜻이다.
- 이 중 주석의 본동작 시간 안:81, 주석 시간 밖:219. 주석 밖이라고 모두 불필요하거나 잘못된 손이라는 뜻은 아니다.
- 추가 손에서 상반신 범위 안에 남은 관절·프레임:5,938.
- 손바닥 기준5점이 모두 입력에 남은 손·프레임:기존4,039 → 후보4,329(+290).
- 기존 관측 관절의 원본 좌표·좌우·유효 여부:변경0.
- 기존 정상 배정3개 및 손 교차53개 프레임 대조:기존 관측 보존.
- 기존/새 좌우 근거 충돌:21개 판단을 진단으로 남기고 새 손 추가 안 함.
  21개 모두가 실제 좌우 오류라는 뜻은 아니다. 겹친 손의 근거 충돌도 포함된다.
- 이전 sensor 시험에서 같은 실제 손을 두 손으로 더했던 공항 사례:별도 회귀 계약에서 추가 차단.
  이 한 사례는 규칙 검사에만 사용했고 native 학습 입력에는 섞지 않았다.

검증 범위
계약14개 통과. 저장된9,136개 판단을 후보 구현과 별도로 산술 재구성했다.
추가300개 손시점과 좌우 충돌21개 판단을 원본과 관절 표시33개 시트로 모두 확인했다.
추가300개에서 명백한 신규 동일손 중복이나 신규 좌우 뒤집힘은 이 시각 검토에서 발견하지 못했다.
이는 손가락 관절의 정답 좌표가 있는 정밀 정확도 시험은 아니다.
합친 뒤 범위 밖165,624개 관절·프레임 좌표를 교란해도 최종1,299,480개 입력 수치가 동일했다.
미관측 좌표0/마스크0,280채널,원래프레임ID·시간,유한값,원본·보호파일32개 해시를 확인했다.
구현과 별도 검산 모두 같은 Codex가 수행했다. 다른 사람 또는 에이전트의 독립 검토라고 부르지 않는다.

남은 문제
1. 손이 보인다는 근거와 손가락 하나하나가 정확하다는 근거는 다르다.
   흐림·회전·가림이 있으면 같은 모델이 두 이미지 크기에서 비슷하게 틀릴 수 있다.
   상처07 f39는21점 중16점만 합의했고, 접힌 손가락의 정확한 형태는 확인이 어렵다.
   유성구주민지원센터09 f35/f40도 회전·겹침 구간의 정밀 관절은 완전 검증하지 못했다.
2. 기존 관측을 보존하므로 기존 좌우 오류를 전부 교정한 것은 아니다. 21개 충돌은 여전히 진단 대상으로 남는다.
3. 새 손 정보를 포함하면 영상 전체에서 계산하는 손 크기 기준도 변한다.
   36개 중35개 영상의 한쪽 이상, 총67개 손 기준이 달라졌다. 기준 변화 최대약7.2%.
   기존 원본 좌표와 몸 기준 좌표는 유지됐지만 기존 손 로컬 특징153,442개 값은 달라졌다.
   이것은 새 오류나 인식률 하락의 증거가 아니라, 다음 학습 비교에서 분리해야 할 입력 변화다.
4. 공개 원본에는 있지만 실제 웹캠 밖에 있는 손은 재검출로 복원할 수 없다.
   현재 화면 안에 손의 시각 근거가 있을 때를 보완하는 방법이다.
5. 이번36개는 이미 문제를 조사하던 TRAIN 자료다. 전체100개 단어·새 사람·모든 수어에서의 일반화는 검증하지 않았다.

저장 형식에서 발견한 문제와 복구
초기 channels.npz의 손크기 메타데이터가 객체 배열로 저장돼 allow_pickle=False 전체 읽기가 실패했다.
최초 오류와 원본 산출물을 보존했다. 재검출·학습을 다시 하지 않고 숫자 배열과 JSON 메타데이터를 분리해 safe_channels에 저장했고,
수치 동일성 및 객체 없이 읽기를 검증했다. 실제 후속 입력 검토에는 safe_channels를 사용한다.

다음 계획
먼저 이번 입력 보완 결과를 사용자와 확인한다. 그 뒤 학습 조건을 고정한다.
주대조는 같은 전체검출 경로에서 손 통합 전/후로 두고, 손 크기 기준 변화 효과를 구분한다.
기존 C·T3-F는 고정 비교 기준으로 보존한다. 센서 선크롭 자료는 자동 혼합하지 않는다.
이미 관찰한 충돌을 정답처럼 덮어쓰거나, 사용자 촬영에 맞춰 규칙·학습을 조정하지 않는다.
추가 학습과 운영 적용은 이번에 수행하지 않았다.
"""
(U/'RESULT_KO.txt').write_text(report,encoding='utf8')

gallery=''.join(f'<details><summary>검토 시트 {i:02d} · 대상 {(i-1)*10:03d}–{min(i*10-1,320):03d}</summary><img loading="lazy" src="evidence/check_{i:02d}.jpg" alt="원본과 관절 대조 {i}"></details>' for i in range(1,34))
css="""*{box-sizing:border-box}body{margin:0;background:#f2f5f9;color:#15253b;font:17px/1.7 'Malgun Gothic',sans-serif}
main{max-width:1180px;margin:auto;padding:45px 24px 100px}h1{font-size:38px;line-height:1.35;margin:12px 0 20px}h2{font-size:26px;margin:0 0 14px}h3{font-size:20px;margin:0 0 12px}
section{background:white;padding:30px;margin:22px 0;border:1px solid #dbe3ed;border-radius:16px}p{margin:10px 0}
.tag{color:#277056;font-weight:bold}.lead{font-size:20px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.flow div{background:#eaf2fc;border-radius:10px;padding:18px}.flow strong{display:block;font-size:19px}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.metric{padding:20px;background:#f2f7fd;border-radius:12px}.metric b{font-size:37px;display:block;color:#1652a1}
table{width:100%;border-collapse:collapse}th,td{padding:13px;text-align:left;border-bottom:1px solid #e0e6ed}th{background:#edf3fa}
.warn{background:#fff6e7;border-left:5px solid #d89525;padding:18px}.note{color:#42556d}img{width:100%;height:auto;display:block;border-radius:8px;margin:15px 0}.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}.bar{height:28px;background:#edf2f7;border-radius:7px;overflow:hidden;margin:8px 0}.bar div{height:100%;background:#258467}.bar.old div{background:#708db3}
details{border-top:1px solid #d9e1eb;padding:12px 0}summary{cursor:pointer;font-weight:bold}a{color:#1559ad}li{margin:7px 0}
@media(max-width:760px){.flow,.metrics,.cols{grid-template-columns:1fr}h1{font-size:29px}section{padding:20px}}
"""
page=f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>전체 영상 손 통합 후 상반신 입력 제한 · WordsInSilence</title><style>{css}</style><main>
<div class="tag">공개 자료 입력 후보 검증 · 2026.10.20</div>
<h1>손은 전체 영상에서 찾고,<br>학습 입력만 상반신으로 제한</h1>
<p class="lead">요청한 순서로 별도 후보를 구현하고 검증했다.<br>기존 C·T3-F 유지 · 사용자 촬영 사용 없음 · 추가 학습 없음</p>
<section><h2>이번에 정리한 처리 순서</h2><div class="flow">
<div><strong>① 전체 원본</strong>기존 연속 검출과 좌우 배정을 사용</div>
<div><strong>② 같은 장면 재확인</strong>빠진 손을 현재 이미지와 팔 근거로 확인</div>
<div><strong>③ 손 정보 통합</strong>같은 손은 하나로, 충돌은 보류</div>
<div><strong>④ 상반신 범위</strong>경계 안 관절만 최종 입력에 남김</div></div>
<p>손 주변의 작은 이미지는 같은 장면을 확대 확인하는 용도다. 상반신으로 먼저 잘라 검출한 별도 학습 자료를 섞는 방식이 아니다.</p>
<p>고정 경계는 영상 중간에 바뀌지 않는다. 프레임과 시간 간격도 그대로 보존했다.</p></section>
<section><h2>같은 공개 영상에서 무엇이 늘었나</h2><div class="metrics">
<div class="metric"><b>36개 영상</b>23개 단어 · 9명 · 4,641프레임</div>
<div class="metric"><b>+300</b>추가된 손·프레임<br>이 중 주석 본동작 안 81</div>
<div class="metric"><b>+5,938</b>범위 안에 추가된 관절·프레임</div></div>
<p>한 손을 10개 프레임에서 다시 찾으면 10건이다. <strong>수어 정답 수나 인식률을 뜻하지 않는다.</strong></p>
<p>손바닥 기준 5점이 모두 최종 입력에 남은 손·프레임</p>
<div>기존 4,039</div><div class="bar old"><div style="width:{4039/4329*100:.3f}%"></div></div>
<div>후보 4,329 · 290 증가</div><div class="bar"><div style="width:100%"></div></div></section>
<section><h2>기존 손은 유지하면서 빠진 손 추가</h2>
<p>냉커피의 본동작 장면. 기존 파란 손은 그대로 두고, 같은 원본에서 다시 확인한 손만 초록색으로 추가했다.</p>
<img src="evidence/detail_067.jpg" alt="냉커피 기존 손과 추가된 손">
<p class="note">노란선 아래 좌표는 마지막 입력 생성 단계에서 제외한다. 사진 속 초록색 전체가 모두 최종 입력으로 들어가는 것은 아니다.</p></section>
<section><h2>교차와 중복을 같은 문제로 취급하지 않음</h2><div class="cols">
<div><h3>다른 팔에 연결된 두 손</h3><p>손이 겹쳐 보여도 현재 팔 근거로 구분되면 두 손을 유지한다.</p><a href="evidence/detail_255.jpg"><img src="evidence/detail_255.jpg" alt="서로 다른 교차 손"></a></div>
<div><h3>기존·새 좌우 근거 충돌</h3><p>같은 손일 가능성이 충돌하면 새 손을 하나 더 추가하지 않는다. 기존 오류의 자동 교정은 이번에 하지 않았다.</p><a href="evidence/detail_175.jpg"><img src="evidence/detail_175.jpg" alt="좌우 근거 충돌 보류"></a></div></div>
<table><thead><tr><th>보존·검증 항목</th><th>결과</th></tr></thead><tbody>
<tr><td>기존 관측 좌표·좌우·유효 여부</td><td>변경 0</td></tr>
<tr><td>기존 정상 배정 / 교차 대조</td><td>3개 / 53개 프레임의 기존 관측 유지</td></tr>
<tr><td>좌우 근거 충돌</td><td>21개 판단에서 새 손 추가 안 함</td></tr>
<tr><td>과거 공항의 동일손 중복 반례</td><td>별도 계약 검사에서 추가 차단</td></tr>
</tbody></table><p>21개 충돌이 모두 실제 오류라는 뜻은 아니다. 과거 sensor 반례는 검사에만 썼고 이번 native 입력에는 섞지 않았다.</p></section>
<section><h2>경계 밖 관절이 들어가지 않는지 확인</h2>
<p>손 전체를 찾은 뒤에도 경계 안의 관절만 남긴다. 아래 예시는 21점 중 2점만 최종 범위에 남는 장면이다.</p>
<img src="evidence/detail_015.jpg" alt="학습 경계 밖 손가락은 제외">
<p>합친 뒤 범위 밖 165,624개 관절·프레임 좌표를 다른 값으로 바꿔도, 최종 1,299,480개 입력값은 동일했다.</p>
<p>이 검사는 <strong>통합 이후 입력 누출 방지</strong> 검사다. 앞단의 좌우 판단에 전체 영상의 근거를 쓰는 것은 의도한 동작이다.</p></section>
<section><h2>남은 문제와 해석의 한계</h2>
<div class="warn"><strong>손을 다시 찾았다고 손가락까지 정답으로 확인된 것은 아니다.</strong>
<p>흐림·회전·가림에서는 같은 검출기가 두 이미지 크기에서 비슷하게 틀릴 수 있다. 관측 추가량을 손 모양 정확도로 발표하지 않는다.</p></div>
<img src="evidence/detail_243.jpg" alt="손 영역은 있지만 손가락 정밀도는 불확실한 사례">
<p>상처 07 f39: 두 크기에서 확인된 16점만 추가했다. 접힌 손가락의 정확한 각도까지 검증한 것은 아니다.</p>
<ul><li>추가 300개와 충돌 21개를 원본으로 대조했다. 명백한 신규 동일손 중복·좌우 전도는 이 검토에서 발견하지 못했다. 정밀 관절 정답 좌표에 의한 정확도 평가는 아니다.</li>
<li>기존 관측을 보존하므로, 기존에 있던 좌우·형상 오류까지 해결한 것은 아니다.</li>
<li>실제 웹캠 화면 밖에 있는 손은 다시 찾을 수 없다. 화면 안에 시각 근거가 있는 경우를 보완한다.</li>
<li>검증한 범위는 이미 문제를 조사하던 TRAIN 36개다. 전체 100단어와 새 사용자 성능을 보장하지 않는다.</li></ul></section>
<section><h2>학습 전에 분리해야 할 변화</h2>
<p>손을 더 찾으면 손 크기를 계산하는 자료도 늘어난다. 기존 좌표를 그대로 두어도, 크기를 맞춘 뒤의 손 로컬 입력은 달라질 수 있다.</p>
<div class="metrics"><div class="metric"><b>35 / 36</b>손 크기 기준이 달라진 영상</div><div class="metric"><b>최대 7.2%</b>손 크기 기준의 상대 변화</div><div class="metric"><b>0회</b>이번에 실행한 학습·인식률 평가</div></div>
<p>원본 좌표와 기존 몸 기준 입력은 유지됐다. 손 로컬 특징 153,442개 값은 달라졌다. 이것만으로 개선이나 악화를 판단할 수 없다.</p></section>
<section><h2>다음 단계</h2><ol>
<li>이번 통합 결과와 남은 불확실성을 함께 확인한다.</li>
<li>같은 전체검출 경로에서 <strong>통합 전 / 통합 후</strong>를 비교하고, 손 크기 기준 변화가 섞이지 않도록 조건을 정한다.</li>
<li>학습 계획을 확인한 뒤 실행한다. C·T3-F는 고정 비교 기준으로 보존한다.</li></ol>
<p>이번에는 별도 연구 후보만 완성했다. 운영 웹·DTW·C·T3-F 변경 및 사용자 촬영의 학습·튜닝은 없다.</p></section>
<section><h2>검증 근거</h2><p>계약 14개 통과 · 저장 판단 9,136개 별도 산술 검산 · 321개 시각 대조 · 원본 및 보호 파일 32개 유지.</p>
<p>검산은 후보 구현과 별도로 작성했지만, 수행자는 같은 Codex다. 다른 검토자의 독립 검토를 받은 것으로 표현하지 않는다.</p>
<p>입력 저장 시 메타데이터가 객체 배열로 들어간 문제를 발견했다. 원본 결과와 최초 오류는 보존하고, 숫자 배열과 JSON을 분리한 <code>safe_channels</code>의 동일성을 확인했다. 재검출·학습 재실행은 없었다.</p>
<p><a href="RESULT_KO.txt">전체 설명</a> · <a href="FINAL_AUDIT.json">최종 기록</a> · <a href="VERIFY.json">수치 검산</a> · <a href="VISUAL_REVIEW.json">시각 검토</a> · <a href="FEATURE_EFFECT.json">크기 기준 영향</a> · <a href="EXPORT_VERIFY.json">입력 내보내기 검증</a></p>
<details><summary>전체 시각 대조 33개 시트 보기</summary><p>파랑=기존, 초록=재검출, 작은 자홍점=확대 확인. 제목의 ‘추가’는 후보 관절이 입력에 추가된 경우다. ‘충돌’의 초록은 추가하지 않은 진단 표시다.</p>{gallery}</details>
</section></main></html>"""
(U/'RESULT_KO.html').write_text(page,encoding='utf8')

# Only update root documents after source, results, and current-document checks.
docpaths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
for p in docpaths:assert sha(p)==sha(U/'before'/p.name),'Root document changed concurrently: '+str(p)
title='2026-10-20 전체 영상 손 통합 후 ROI 후보 검증 완료 — 기존 관측 보존·추가300, 학습 전 검토'
summary=f"""### {title}
- 단위 `ops/{U.name}`. `RESULT_KO.html`/`FINAL_AUDIT.json`. 전체 원본 검출·좌우 기존자료+같은 원본 이미지 재검출을 원본좌표에서 통합한 뒤 고정ROI·280입력. sensor 미래학습 자동혼합 없음. C/T3-F/DTW/웹 보존.
- 공개 TRAIN36영상/23단어/9명/4641프레임. 새 손300손·프레임(주석본동작81),ROI안5938관절·프레임. 손바닥5점유효4039→4329(+290). 인식률아님.
- 기존관측XY/좌우/유효값보존. native정상3/교차53프레임대조 유지. 기존/새좌우근거충돌21판단은새손추가안함(21실제오류로해석금지). 공항과거sensor동일손중복반례는별도계약에서차단,입력에sensor섞지않음.
- 계약14PASS·저장판단9136별도검산·추가300+충돌21원본시각검토33시트. 명백한신규중복/좌우전도는이검토에서못찾음. 정밀손가락좌표정답시험아님. 같은모델두크기합의는가시성/해부학적진실아님;형상·가림불확실성유지. 기존오류자동교정아님.
- 통합후ROI밖165624관절교란에도1299480입력값동일. 원래프레임ID/시간/280차원/미관측0유지. 후보추출완료,재추출금지. VERIFY의pending_visual은당시상태이며최종VISUAL_REVIEW참조.
- 손추가로35/36영상·67손크기기준변화(최대약7.2%),기존손로컬153442값변경. 원본·기존몸기준좌표불변. 인식회귀증거아니며향후크기기준효과분리필요.
- channels.npz객체메타데이터읽기실패최초증거보존;원본출력불변,숫자safe_channels+JSON별도저장·동일성확인. 모델재평가/학습/사용자촬영접근0. 입력후보만검증,운영미채택.
- 원본262핀·보호32파일해시유지. Codex구현+별도산술검산+이미지검토(동일수행자,다른에이전트독립검토아님). 실행종료/잠금없음. 결과확인후학습조건검토.

"""
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf8')
text=text.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+summary,1)
text=text.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n### '+title+'\n'+next_action+'\n\n',1)
p.write_text(text,encoding='utf8')
state=read(O/'STATE.json');state['updated_date_kst']='2026-10-20'
state['current_stage']='WORD100_FULLFRAME_HAND_MERGE_VALIDATED_PENDING_USER_REVIEW'
state['next_single_action']=next_action
state[U.name]={**audit,'report':str(U/'RESULT_KO.html')}
(O/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f"""# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 전체 원본 영상에서 관절을 검출·통합한 뒤, 학습 입력만 고정 상반신 범위로 제한한다.
마지막 완료: 공개 TRAIN36 입력 통합 후보·저장 산술검산·원본 시각대조·최종 입력 검증.
다음 한 작업: {next_action}

{summary}
- 실제 기본운영/웹 C·T3-F는 이전대로이며 이번 후보를 사용하지 않는다. 기존 모델을 수정했다고 말하지 않는다.
- 이전 C는 masked/sensor 번갈아 학습, B는masked전용. 이번후보는native만. 원본검출과학습마스크분리가사용자명시의도.
- 36개 영상 전체순서의 기존 검출캐시 재사용, 같은프레임 IMAGE재검출은이번에1회완료. 원본영상 전체를재학습한것아님. 전체1199입력준비완료도아님.
- 모든추가/충돌이미지는 evidence/check_01~33.jpg. 상세좋은예 detail_067,교차255,보류175,경계015,불확실243/252.
- `safe_channels`를 후속 검토의 수치자료로 사용. 최초outputs객체메타데이터파일을덮어쓰거나allow_pickle우회하지않는다.
- 불확실한손가락을정밀정답으로표현하지않고,실제웹캠밖손복원가능주장금지. 300은손×프레임수,성능정답수아님.
- `FEATURE_EFFECT.json`:동일한원본좌표라도손크기정규화가변해입력차이가난다는분리필요. 추가학습은사용자확인전금지.
- 보고서정적링크/JSON확인예정. 브라우저자동미리보기는별도미확인;정책차단우회없음.
""",encoding='utf8')
print('REPORT_AND_HANDOFF_WRITTEN',at)
