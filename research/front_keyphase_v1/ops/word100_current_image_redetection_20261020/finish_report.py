"""Close out saved public re-detection comparison; never performs inference."""
import json, hashlib, html
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
f=read(U/'FROZEN.json');v=read(U/'VERIFY.json');t=read(U/'REVIEW_TARGETS.json')
assert len(t)==308
for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
for item in read(U/'INDEX.json'):assert sha(item['path'])==item['sha256']
assert not (U/'RUNNING.json').exists()
assert not (U/'FIRST_ERROR.json').exists()
s=read(U/'SUPPLEMENTAL_COUNTS.json')
shape={141:'Right hand located, but wrist/palm elongated toward forearm.',
       142:'Right hand located, but wrist/palm elongated toward forearm.',
       159:'Both methods show elongated wrist/palm despite correct visible hand region.',
       227:'Visible left hand found, but finger landmarks collapse into a narrow line.'}
reviews=[]
for a in t:
    reviews.append(dict(id=a['id'],clip_id=a['clip_id'],index=a['index'],side=a['side'],
        evidence=f"evidence/added_{a['page']:02d}.jpg",
        visible_hand_region='visually_plausible_same_anatomical_hand',
        merge_status='duplicate_existing_physical_hand' if a['id']==307 else 'no_clear_duplicate_observed_in_contact_review',
        detailed_geometry=shape.get(a['id'],'not_pointwise_ground_truth_verified'),
        note='Current image detections are model estimates, not certified visible joints.'))
visual=dict(at=at,status='ALL_308_ADDED_HAND_FRAMES_VISUALLY_REVIEWED_WITH_COUNTEREXAMPLES',
    reviewer='Same Codex as implementation; no independent human/agent/coordinate ground truth.',
    sheets_reviewed=31,overlap_review_ids=s['overlap_review_ids'],
    detail_review_ids=s['overlap_review_ids']+[141,142,159],
    duplicate_ids=[307],shape_counterexample_ids=list(shape),rows=reviews,
    conclusion='Detection proposals cover visible hands, but one existing wrong-side detection duplicates a newly found hand. Shape distortion and partial fingers remain. Not approved for merging or model input.')
write(U/'VISUAL_REVIEW.json',visual)
bykey=defaultdict(set)
for a in read(U/'ADDITIONS.json'):bykey[(a['clip_id'],a['index'],a['side'])].add(a['mode'])
overlap=dict(both=sum(x=={'image','crop'} for x in bykey.values()),
             image_only=sum(x=={'image'} for x in bykey.values()),
             crop_only=sum(x=={'crop'} for x in bykey.values()),unique=len(bykey))
audit=dict(at=at,status='COMPLETE_CURRENT_IMAGE_DETECTION_GAIN_WITH_MERGE_AND_SHAPE_FAILURES_NOT_ADOPTED',
    scope=dict(videos=36,words=23,people=9,frames=4641,split='TRAIN only; same known diagnostic sample'),
    arithmetic='VERIFY.json PASS at saved-check stage; pending_visual superseded by VISUAL_REVIEW.json',
    contracts='11 PASS',decisions_verified=4820,
    proposals=dict(image=294,crop=303,image_core=76,crop_core=77,overlap=overlap),
    failures=dict(new_duplicate_physical_hand=1,duplicate_target=307,
        shape_counterexamples=list(shape),partial_joint_outside_sensor=s['partial_joint_outside_sensor'],
        known_baseline_identity_errors_unchanged=12),
    controls=dict(normal42_preserved=True,crossing54_preserved=True,baseline_overwritten=0),
    initial=dict(full_native_first_frame_two_candidates=36,sensor_first_frame_zero_candidates=35,sensor_first_frame_one_candidate=1,
                 known12_errors_in_frame0=1,known12_errors_after_frame0=11,
                 caveat='Two raw candidates are not certified left/right identity. Source and sensor paths must be distinguished.'),
    speed=dict(clips=3,words=2,frames=365,factors=[1,.75,.5,1],count_label_xy_changed=0,
               method='Same sequential pixels, timestamp scaling only, last1 repeat-control; no interpolation or frame duplication.'),
    timing=dict(full_image_median_ms=v['latency_ms']['image']['median'],
                full_image_p95_ms=v['latency_ms']['image']['p95'],
                crop_calls=s['crop_calls'],crop_mean_per_call_ms=s['crop_per_call_mean_ms'],
                limitations='Detector only, this machine; not end-to-end FPS. Full image measured on all frames; crops only when eligible.'),
    protected_files=32,existing_C_T3F_unchanged=True,user_recording_reads=0,training=0,
    classifier_inference=0,downloads=0,production_changes=0,
    generality='Word-independent conditions, evaluated on23 known public words only. No claim of all-sign or new-user improvement.',
    role='Codex implementation/execution plus separate saved reconstruction and visual review by same Codex.',
    decision='Research candidate only; do not merge directly into C, DTW or training.',
    next='Review result. Next proposed unit: duplicate physical-hand reconciliation and per-joint observation quality before input-level comparison; no training until user review.')
write(U/'FINAL_AUDIT.json',audit)

report="""현재 이미지에서 손 다시 찾기 — 공개 자료 비교 결과

결론
보이는 손을 놓친 순간에 현재 이미지를 다시 분석하면 추가 손 후보를 얻을 수 있었다.
그러나 기존 오배정과 합쳐 같은 손을 두 손으로 세는 반례1건, 손목·손가락 좌표 왜곡,
화면 밖 손가락 문제가 확인돼 학습·운영에는 적용하지 않았다. 수어 인식률 향상은 이번에 측정하지 않았다.

1. 처음부터 두 손이 보이는데 왜 틀렸나?
이전 설명에서 원본 전체 화면과 실제 웹캠을 흉내 낸 제한 화면을 충분히 분리하지 못했다.
저장된36원본은 첫 프레임에서 모두 손 후보2개가 있었다. 제한 화면은35개에서0개,1개에서1개였다.
우리가 확인하던 알려진12좌우 오류는 sensor(제한 화면) 경로이며, 그중 첫 프레임1건, 이후11건이다.
원본 정지 자세의 두 손을 처음부터 전부 못 찾았던 상황이라는 설명은 부정확하다.
전체화면 검출 후 ROI 안만 쓰는 masked 경로도 그대로 있고, 기존 C 학습은 masked와 sensor를 섞었다.
sensor는 웹캠에 아예 찍히지 않는 손의 정보를 이용하지 못하게 만든 비교 조건이다.
원본에서 후보2개를 찾았다고 좌우와21개 관절 모두 정확했다는 뜻은 아니다.
좌우 라벨은 매번 영상에서 추정하는 값이고, 교차·가림·손목 근거 부족 때 바뀔 수 있다.

2. 0.75배속/0.5배속 추출
실제 빠른 동작은 프레임 사이 이동과 흔들림 때문에 검출을 어렵게 만들 수 있다.
하지만 이미 촬영된 영상의 재생시간만 늘려도 흐린 손이 선명해지거나 놓친 순간의 사진이 생기지는 않는다.
현 오프라인 추출은 원래도 각 프레임을 순서대로 처리하며, 컴퓨터가 느리다는 이유로 재생시간에 맞춰 생략하지 않는다.
공개3영상·2단어·365프레임을 같은 픽셀로1배/0.75배/0.5배/1배재시험했다.
손 개수·좌우 라벨·좌표가 모두 같았고 최대 좌표차0이었다.
이는 timestamp만 바꾼 시험이다. 프레임 반복/보간/재인코딩은 하지 않았고 모든 빠른 동작에 대한 결론은 아니다.
실제 촬영 FPS·셔터/노출·프레임별 지연을 별도로 확인하는 것이 우선이다. 원래 촬영시간 정보는 보존해야 한다.

3. 같은 공개36영상·23단어·9명·4641프레임 비교
기존 C의 센서 관측을 보존하고, 기존 좌우 슬롯이 비어 있으며 현재 팔 근거가 있는 경우만 새 후보를 제안했다.
전체 현재 이미지 재검출:294손시점 후보, 그중 주석 본동작76.
현재 팔로 찾은 작은 영역 재검출:303손시점 후보, 그중 주석 본동작77.
두 방식의 중복을 합친 고유 검토 대상308손시점(289공통/전체만5/국소만14).
같은 손을 여러 프레임에서 찾으면 여러 건이며, 영상294개나 정확히 복구한 관절294개라는 뜻은 아니다.
주석 구간 밖에는 준비/마무리 등이 포함된다. 본동작 의미 경계의 정답을 새로 검증한 것은 아니다.
전체 화면 IMAGE 모드는 이전 추적 상태에 의존하지 않고 현재 이미지에서 다시 검출한다.
별도 새 검출 모델 다운로드 없이 같은 기존 MediaPipe 가중치를 사용했다.
좌우 수용은 현재 팔꿈치·손목 연결을 사용하고, 단어 이름이나 사용자의 결과는 조건에 없다.

4. 정상 유지와 남은 문제
기존 정상 배정42개와 교차54프레임의 기존 출력은 그대로 유지됐다. 원래 값을 덮어쓰지 않는 설계의 검산이다.
알려진 기존12오배정도 그대로다. 이번 후보가 좌우 오류까지 고쳤다고 볼 수 없다.
공항09 f41: 기존 실제 오른손을 왼손으로 배정한 값이 남은 채 새 오른손이 추가된다.
중심이 다르게 추정돼 거리 기반 중복 방지를 통과했으며, 합치면 같은 손을 두 손으로 취급한다.
추출09 f81: 보이는 손을 찾았지만 손가락 좌표가 좁은 선으로 찌그러졌다.
대지05 f83/f84 및 대지06 f66: 손 영역은 맞아도 손목·손바닥이 팔 쪽으로 길게 잡힌 예가 있다.
전체 재검출26/국소32손시점은 손바닥이 안에 있어도 손가락 일부가 화면 밖이다(서로 중복 가능).
따라서 손 존재/좌우/중심/손가락 각도 유효성을 별개로 다뤄야 한다.
308추가 손시점31시트를 모두 확인하고, 기존 손과 겹치는8개 및 모양 반례3개를 확대 비교했다.
검토는 동일 Codex이며 정밀 관절 정답이나 독립 전문가 판정은 아니다.
검출 추가와 단어 인식률 향상은 다른 결과다. 이번에는 학습/TCN분류/새사용자 성능 측정이 없다.

5. 처리 비용
현재 장비에서 전체 이미지 재검출 자체 중앙22.28ms/95백분위24.31ms.
국소 검출1091호출 평균11.91ms/호출. 기존 Pose/영상읽기/TCN 비용은 포함하지 않는다.
매 프레임 둘 다 실행하는 운영 방식을 채택한 것이 아니다. 누락 시 제한적으로 재검출하는 구성이 후속 후보이다.

6. 다음 제안
첫째, 기존 손과 새 손이 같은 실제 손인지 확인한다. 충돌 시 억지로 양손에 채우지 않고 보류한다.
단순히 두 손이 가까우면 모두 제거하는 방식은 정상 교차를 해치므로 사용하지 않는다.
둘째, 손을 찾은 근거와 손목·각 손가락 좌표가 믿을 만한 근거를 분리한다.
잘 안 보이는 손가락을 검출기의21점 출력만 보고 실제 관측으로 확정하지 않는다.
셋째, 같은 공개 대조에서 정상 교차/새 중복/누락/처리시간을 비교한 뒤 입력 연결 여부를 판단한다.
C·T3-F는 그대로이고 사용자 촬영 읽기/학습/튜닝/후보선택0. 추가 학습은 결과를 사용자와 확인한 다음이다.

검증
계약11PASS, 저장 판단4820건 별도 산술 검산, 원본/동결핀/32보호파일 해시 유지.
최초 오류 없음, 실행 완료, RUNNING 없음. 같은 Codex의 별도 검산과 시각 검토이며 다른 검토자 검증이 아니다.
FINAL_AUDIT.json이 최종 판정이다. VERIFY.json의 pending_visual은 산술 검산 당시 기록으로 보존한다.
"""
(U/'RESULT_KO.txt').write_text(report,encoding='utf8')
sections=[
('처음 손이 보이는데도 틀렸다는 설명, 바로잡기',"""
<p>네가 짚은 부분이 맞아. <b>이번 36개 원본은 첫 프레임부터 모두 손 후보 2개를 검출했다.</b>
문제 사례는 주로 <b>웹캠처럼 아래쪽이 안 보이도록 제한한 별도 경로</b>에서 나왔다.</p>
<div class="pair"><article><h3>원본 전체 → 관절 검출 → 범위 안만 사용</h3><p>masked 경로. 처음부터 보이는 손을 추적하는 방식은 유지되어 있다.</p><strong>첫 프레임 두 손 후보 · 36 / 36</strong></article>
<article><h3>제한 화면만 → 관절 검출</h3><p>sensor 경로. 실제 웹캠에 찍히지 않는 손을 모르는 조건을 재현한다. C 학습은 두 경로를 혼합했다.</p><strong>첫 프레임 손 0개 · 35 / 36</strong></article></div>
<p>알려진 sensor 좌우 오류 12건 중 1건만 첫 프레임이며, 11건은 이후 손이 등장하거나 움직이는 순간이다.
두 후보가 있다는 사실만으로 좌우·손가락까지 모두 정확하다고 단정할 수는 없다.</p>
<img src="evidence/initial.jpg" alt="원본 전체 화면과 노란 경계 아래를 제외한 제한 화면 비교">
<p class="caption">왼쪽: 원본과 노란 하단 경계. 오른쪽: 같은 프레임의 제한 화면. 확대율이 아닌 입력에 포함되는 픽셀의 차이.</p>"""),
('느리게 재생하면 더 잘 찾을까?',"""
<p><b>실제 빠른 움직임은 영향을 줄 수 있지만, 이미 찍은 영상을 느리게 재생하는 것만으로는 정보가 늘지 않는다.</b>
흐릿한 사진을 오래 보더라도 새로운 선명한 사진이 생기는 것은 아니라는 뜻이다.</p>
<div class="pair"><article><h3>이번에 실제 비교한 것</h3><p>3개 영상 · 2개 단어 · 365프레임의 같은 픽셀을 1배 / 0.75배 / 0.5배 / 1배 재시험.
프레임 시간표만 변경했고 반복·보간은 하지 않았다.</p></article><article><h3>결과</h3><strong>손 개수 · 좌우 · 좌표 변화 0</strong>
<p>우리 오프라인 코드는 원래도 프레임을 순서대로 모두 처리한다. 촬영 당시의 FPS·노출·흔들림과 실시간 처리 지연을 별도로 보는 것이 우선이다.</p></article></div>"""),
('놓친 손을 현재 이미지에서 다시 찾은 결과',"""
<p>공개 학습용 자료 36영상 · 23단어 · 9명 · 4,641프레임. 같은 기존 MediaPipe 모델의 IMAGE 모드로 현재 화면을 다시 분석했다.
새 후보는 현재 팔 연결을 확인한 뒤 제안하며 기존 좌표는 덮어쓰지 않았다.</p>
<table><thead><tr><th>별도 비교 방법</th><th>추가 후보 손시점</th><th>주석 본동작 안</th><th>손가락 일부 화면 밖</th></tr></thead>
<tbody><tr><td>현재 제한 화면 전체에서 재검출</td><td>294</td><td>76</td><td>26</td></tr>
<tr><td>현재 팔이 가리키는 작은 영역에서 재검출</td><td>303</td><td>77</td><td>32</td></tr></tbody></table>
<p><b>1손시점 = 특정 프레임의 한쪽 손.</b> 같은 손의 연속 프레임도 각각 센 수이며, 검출 정확도나 단어 정답률이 아니다.
두 방식에 겹친 후보 289개, 전체만 5개, 국소만 14개로 고유 308손시점을 시각 검토했다.</p>
<img src="evidence/detail_025.jpg" alt="기존 손은 유지하면서 다른 손을 현재 이미지로 다시 찾은 교차 사례">
<p class="caption">가래떡 · 두 손이 가까운 장면. 파랑: 기존 손, 초록/자홍: 다시 찾은 손. 손의 존재를 확인하는 데 도움이 되지만 정밀 손가락 각도의 정답을 뜻하지 않는다.</p>"""),
('바로 적용하지 않은 이유',"""
<div class="callout"><b>중요한 반례: 같은 실제 손을 두 손으로 세는 경우 1건</b>
<p>기존 오른손이 왼손 슬롯에 잘못 들어간 상태에서 새 오른손이 추가됐다.
추정된 손바닥 중심이 서로 멀어 거리 검사만으로 중복을 막지 못했다.</p></div>
<img src="evidence/detail_307.jpg" alt="공항의 동일한 오른손에 기존 및 신규 검출이 겹치는 반례">
<div class="pair"><article><h3>잘되던 배정은 유지</h3><p>기존 정상 42개 · 교차 54프레임의 출력 변화 0. 기존 관측을 덮어쓰지 않는 설계로 보존했다.</p></article>
<article><h3>기존 오류 해결은 별개</h3><p>알려진 좌우 오류 12건은 그대로다. 여기에 새 후보를 붙이는 것만으로 오배정을 고칠 수는 없다.</p></article></div>
<h3>손을 찾아도 손 모양이 정확하지 않을 수 있다</h3>
<img src="evidence/detail_227.jpg" alt="보이는 손에서 손가락 좌표가 좁은 선으로 찌그러진 반례">
<p>추출의 손가락 선이 한쪽으로 뭉친 예. 대지에서도 손목·손바닥이 팔 방향으로 길어지는 반례가 있다.
또한 손바닥은 보여도 손가락 일부는 화면 밖인 경우가 있어 <b>손 존재·좌우·중심·각 손가락의 유효성을 분리</b>해야 한다.</p>"""),
('다음에 보완할 순서',"""
<ol><li><b>중복과 좌우 충돌부터 확인.</b> 같은 실제 손을 양손 슬롯에 동시에 넣지 않는다. 정상 교차는 유지하고, 근거가 엇갈리면 추가를 보류한다.</li>
<li><b>손 찾기와 정밀 좌표 판단 분리.</b> 위치를 찾았어도 손목·손가락이 불명확하면 각도 비교에 그대로 넣지 않는다.</li>
<li><b>같은 공개 자료에서 대조.</b> 추가 관측뿐 아니라 새 오류·교차 보존·처리시간을 함께 확인한 후 연결 여부를 판단한다.</li></ol>
<p>현재 장비의 검출기 자체 비용은 전체 이미지 중앙 22.28ms, 국소 영역 평균 11.91ms/호출이다.
영상 읽기·Pose·TCN 비용은 별도다. 운영에서는 누락 시에만 제한적으로 호출할 방안을 검토한다.</p>
<p><b>C·T3-F / 운영 DTW / 웹은 그대로.</b> 사용자 촬영 접근·학습·튜닝·후보선택 0. 이번 연구 후보는 미채택이며 추가 학습은 결과 확인 후에 진행한다.</p>
<p>공통 조건으로 구현했지만 확인 범위는 알려진 23단어의 공개 진단 자료다. 모든 수어·새 사람의 인식률 향상은 아직 입증되지 않았다.</p>""")]
gallery=''.join(f'<option value="{i:02d}">{i} / 31</option>' for i in range(1,32))
body=''.join(f'<section><div class="eyebrow">{i:02d}</div><h2>{title}</h2>{b}</section>' for i,(title,b) in enumerate(sections,1))
page="""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WordsInSilence · 현재 이미지 재검출 비교</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f9;color:#17263b;font-family:'Malgun Gothic',sans-serif;line-height:1.75}
main{max-width:1160px;margin:auto;padding:40px 24px 80px}header{background:#152b43;color:white;border-radius:22px;padding:42px}
h1{font-size:36px;line-height:1.3;margin:12px 0 22px}h2{font-size:26px;line-height:1.4;margin:6px 0 20px}h3{font-size:19px;line-height:1.5}p{margin:12px 0}
.tag{font-size:13px;letter-spacing:2px;color:#a9cee9}.badge{display:inline-block;background:#d4e9ff;color:#183754;padding:7px 13px;border-radius:7px;font-weight:bold}
section{background:white;border:1px solid #dfe5ed;border-radius:18px;margin-top:24px;padding:32px}.eyebrow{font-size:14px;color:#2b6b99;font-weight:bold}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:20px 0}.pair article{background:#f2f6fa;padding:20px;border-radius:12px}.pair h3{margin-top:0}
img{display:block;max-width:100%;height:auto;border-radius:10px;margin:20px auto}.caption{font-size:14px;color:#486074}
table{width:100%;border-collapse:collapse;margin:20px 0}th,td{padding:14px;text-align:left;border-bottom:1px solid #dfe5ed}th{background:#edf3f9}
.callout{padding:20px 24px;background:#fff3e2;border-left:5px solid #d48a2c;border-radius:8px}a{color:#17659e}li{margin:12px 0}
select,button{font:inherit;padding:8px 16px;border:1px solid #b8c8d8;border-radius:7px;background:white}footer{font-size:14px;color:#53667d;padding-top:24px}
@media(max-width:700px){main{padding:15px}header,section{padding:22px}.pair{grid-template-columns:1fr}h1{font-size:28px}table{font-size:12px}th,td{padding:8px}}
</style><main><header><div class="tag">WORDS IN SILENCE · PUBLIC VIDEO DIAGNOSTIC</div>
<h1>화면에 보이는 손,<br>현재 이미지에서 다시 찾기</h1><p>보이는 손을 추가로 찾을 가능성은 확인했다.<br>중복 손과 좌표 왜곡이 남아, 자동 적용은 보류했다.</p>
<span class="badge">공개 자료 비교 완료 · 추가 학습 없음</span></header>"""+body+"""
<section><div class="eyebrow">06</div><h2>추가 후보 전체 확인</h2>
<p>308손시점 · 31시트. 각 항목 왼쪽은 장면, 오른쪽은 손 확대. 초록은 전체 재검출, 자홍은 국소 재검출이다.
‘구간밖’은 제공된 주석 구간 밖이라는 뜻이며 잘못된 수어라는 뜻이 아니다.</p>
<label for="sheet">시트 선택 </label><select id="sheet">"""+gallery+"""</select>
<img id="gallery" src="evidence/added_01.jpg" alt="추가 손 후보 연락 시트">
<p>같은 Codex가 시각 검토했으며 독립 전문가 판정이나 정밀 관절 정답이 아니다. 전체 검토와 중복·형상 반례는 아래 근거 파일에 분리 기록했다.</p></section>
<footer><p>근거: <a href="RESULT_KO.txt">전체 설명</a> · <a href="FINAL_AUDIT.json">최종 판정</a> ·
<a href="VERIFY.json">저장 결과 검산</a> · <a href="VISUAL_REVIEW.json">시각 검토</a> · <a href="INITIAL_AUDIT.json">초기 검출 감사</a> · <a href="FROZEN.json">고정 비교 조건</a></p>
<p>기술 근거: <a href="https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker">MediaPipe Hand Landmarker 개요</a> ·
<a href="https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python">공식 Python 모드 설명</a></p>
<p>현재 결과는 검출 보조의 개발 비교이며 수어 정확도 평가가 아니다. 원본과 기존 연구 결과는 보존했다.</p></footer></main>
<script>document.getElementById('sheet').addEventListener('change',e=>{document.getElementById('gallery').src='evidence/added_'+e.target.value+'.jpg';});</script></html>"""
(U/'RESULT_KO.html').write_text(page,encoding='utf8')
print(audit['status'])
