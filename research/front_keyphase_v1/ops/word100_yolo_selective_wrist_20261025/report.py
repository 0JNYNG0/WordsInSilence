from common import *
import html
s=read(U/'SUMMARY.json');audit=read(U/'FIVE_SCENE_AUDIT.json');adds=read(U/'ADDITIONS.json')
conclusion='누락 순간에만 YOLO를 호출하는 부품을 검증했다. MediaPipe 재검출의 기존 관측을 보존하면서 기존 평가 표본의 손목 지원이 1장면 늘었다. 손목만 추가한 6프레임은 정밀 정답이 없어 정확도 향상으로 확정하지 않는다. 보류5장면은 안전하게 복원되지 않아 그대로 남았다. 운영 적용과 추가 학습은 하지 않는다.'
nextstep='이번 선택 호출과 위치/모양 분리 코드는 연구 후보로 보관한다. 다음 우선 작업은 검출 누락과 검출됐지만 위치가 잘못된 경우를 나눠, 손목·보이는 손가락의 독립 정답을 가진 공개 표본에서 보완 이득을 평가하는 것이다. 현재 두 크기의 검출 일치를 정답으로 간주하거나, 보류5장면을 통과시키려고 기준을 낮추지 않는다. 중복 후보 통합은 같은 실제 손이라는 별도 근거가 있을 때만 설계한다. C·T3-F/운영과 사용자 자료를 유지하며 정밀 검증 전 학습·운영교체하지 않는다.'
notes={
 'R01':'손목은 일부 확대 후보와 2.1px·9.0px 차이로 가까웠다. 그러나 같은 팔에 연결되는 확대 후보가 2개이고, 손바닥·손가락 배치가 크게 다르다. 겹친 주먹과 펼친 손의 정보가 섞인 듯한 왜곡도 영상에서 보인다. 손목만 선택해도 어느 후보를 믿을지 확정하기 어려워 보류했다.',
 'R02':'확대 검출에서는 손을 찾지 못했다. 또한 MediaPipe 재검출이 이미 손을 채운 상태인데 그 손목이 평가 범위를 벗어나 있다. 이번 누락 전용 보조는 기존 손을 덮어쓰지 않으므로 이 오류는 남는다. 미검출과 잘못 검출한 경우를 별도로 처리해야 함을 보여준다.',
 'R14':'확대 후보 2개의 손목 차이는 15.9px·20.5px로 기존 허용폭 안이다. 하지만 손가락 배열이 서로 다르고 후보가 중복된다. 손목 위치 근거 일부가 있어도 손 모양까지 확정할 수 없어 보류했다.',
 'R21':'전체 화면과 확대 영상의 손목이 32.6px 차이로 기존 허용폭 약24.1px를 넘는다. 손목부터 어긋나고 손가락 배치도 달라 손목만 복원할 근거도 충분하지 않다.',
 'R22':'두 크기의 손목이 36.5px 차이로 기존 허용폭 약24.2px를 넘는다. 손을 옆으로 펼친 모습에서 관절 배치가 다르다. 손목과 모양 모두 불확실한 상태다.'}
review=[]
for i,a in enumerate(adds):
    note='손목이 해당 손 주변에 표시되고 기존 반대손은 유지된다. 정확한 손목 마디 위치의 정밀 판독은 보류한다.'
    if i==0:note='기존 R24의 손목 범위로 추가 지원을 확인한 장면. 양쪽 크기에서 손 모양이 비슷하더라도 실제 손가락21점의 정확성은 보증하지 않는다.'
    if i in (3,4):note='손이 겹친 상태에서 추가 손목이 주먹 쪽에 표시된다. 실제 손목과 주먹 표면의 위치 구별은 이 화면과 기존 주석만으로 확정하기 어렵다. 정밀 위치 정확성 미확인.'
    review.append({'id':f'A{i+1:02d}',**a,'visual_note':note,'classification':'qualitative_only_not_ground_truth'})
write(U/'VISUAL_REVIEW.json',{'reviewer':'Codex, predictions visible, not independent or blind','rows':review,'five_scene_notes':notes,'precise21GT':'not_available','no_new_obvious_empty_background_hands_at_display_scale':True,'accuracy_claim_for_six_position_only':False})
write(U/'FINAL_NOTES.json',{'conclusion':conclusion,'next':nextstep,'limitations':['Known diagnostic clips and coarse AI wrist boxes, not general accuracy','Six wrist-only additions lack independent location ground truth','Additional07 cohort856frames had only1trigger and0additions: preservation check, not recovery proof','Fallback cannot fix existing false-positive hands or partially observed fingers','CPU hybrid component estimate uses previousM1 and newYOLO; called-frame overhead can be large','No classifier/DTW evaluation or model training']})
def e(x):return html.escape(str(x))
rows=''
names={'M1':'MediaPipe 재검출','S':'재검출 후 YOLO · 손 전체 일치','P':'같은 선택 호출 · 손목/모양 분리'}
for c in names:
    a=s['scoring']['R24'][c];b=s['scoring']['V18'][c]
    rows+=f'<tr><td>{names[c]}</td><td>{a["counts"].get("inside_both",0)} / {a["scorable"]}</td><td>{b["counts"].get("inside_both",0)} / {b["scorable"]}</td></tr>'
auditcards=''
for a in audit:
    sid=a['id'];cropinfo=[]
    for c in a['crop_candidates']:
        ds=c['wrist_delta_pixels'];ad=[v for v in c['angle_differences_degrees'] if v is not None]
        cropinfo.append(f"확대 후보 {c['index']}: 연결 팔 {c['arm']}, 손목 차이 {ds:.1f}px" if ds is not None else f"확대 후보 {c['index']}: 손목 좌표 불충분")
    auditcards+=f'<article><h3>{sid} · {e(a["label"])} · {a["frame_id"]}프레임</h3><p>{notes[sid]}</p><p class="muted">{e(" / ".join(cropinfo) or "대상 확대 후보 없음")} · 기존 허용폭 {a["shoulder_scale_pixels"]*.08:.1f}px</p><img src="audit_visual/{sid}.jpg" loading="lazy"></article>'
addcards=''.join(f'<article><h3>{q["id"]} · {e(q["label"])} · {q["frame_id"]}프레임</h3><p>{q["visual_note"]}</p><img src="additions_visual/{q["id"]}.jpg" loading="lazy"></article>' for q in review)
lat=s['latency'];ratio=100*(1-139/4781)
text=f'''선택적 YOLO 손목 보조 검증 — 2026-10-25
결론: {conclusion}
범위: 이전39영상4781프레임 + 추가공개7영상856프레임 =46영상5637프레임. 기존TRAIN 공개자료만 사용.
처리: 전체영상MP→MP재검출→전체원영상에서도손이전부누락되고현재팔손목이영역안일때YOLO→기존좌표보존→마지막고정ROI.
비교 S: 기존 손바닥 전체 일치 규칙. P: 같은 규칙에서 손목만 일치할 경우 손목만 별도 사용, 손가락 각도 근거로 쓰지 않음. 임계값0.08어깨폭 유지, 단어/정답 입력 없음.
R24의23판독가능: M1 12/23→S13/23/P13/23. V18의16판독가능: 12/16유지. 새 지원R24 1개, 기존지원손실0, 새범위밖/애매추가0(42개평가장면에한정).
전체영상 추가: 손전체근거1손-프레임+손목만6손-프레임. 6개는독립정답없어정확도개선확정금지.
기존M1관절101680관절-프레임정확보존, 기존교차35프레임보존. 프레임/시간/비교분모삭제없음.
호출: 알려진39영상에서139/4781프레임에만YOLO전체영상호출({ratio:.1f}%감소), 대상확대141회추가. 새07영상1/856프레임호출/추가0.
속도: 새YOLO추가비용 전프레임평균{lat['new_selective_yolo_over_all_frames']['mean_ms']:.2f}ms, 호출프레임평균{lat['new_selective_yolo_on_called_frames']['mean_ms']:.1f}ms,p95 {lat['new_selective_yolo_on_called_frames']['p95_ms']:.1f}ms.
이전M1측정+새YOLO측정 부품합산추정 {lat['M1_prior_components']['mean_ms']:.1f}→{lat['M1_plus_selective_components']['mean_ms']:.1f}ms/프레임. 실제웹캠FPS/동일세션end-to-end측정 아님.
5장면 상세: {json.dumps(notes,ensure_ascii=False)}
새영상856프레임에서는호출조건이1회뿐이라 정상보존은확인했으나 새로운복원효과는검증하지못함.
손목범위는기존AI의대략적인범위, 전문가21관절정답/수어인식률/피드백정확도아님.
다음: {nextstep}
새학습0,TCN추론0,사용자/멘토촬영접근0,운영변경0. 구현Codex/별도저장산술검산Codex, 신규독립전문가검토아님.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>누락 때만 YOLO · 손목과 손 모양 분리</title><style>
body{{margin:0;background:#eef3f8;color:#1c2f43;font:16px/1.8 "Malgun Gothic",sans-serif}}main{{max-width:1120px;margin:auto;padding:36px 24px}}section{{background:white;border-radius:18px;padding:28px;margin:22px 0}}h1{{font-size:34px;line-height:1.35}}h2{{font-size:24px}}h3{{font-size:20px}}.tag,.muted{{color:#526b82}}.call{{padding:18px;background:#fff4dc;border-left:5px solid #db9b27}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}.metrics>div{{padding:18px;background:#edf5fb;border-radius:12px}}.metrics b{{display:block;font-size:30px;color:#146b93}}table{{width:100%;border-collapse:collapse}}td,th{{padding:14px;text-align:left;border-bottom:1px solid #dce7ef}}th{{background:#eff6fb}}img{{width:100%;height:auto;border-radius:10px}}article{{margin:28px 0;padding-top:18px;border-top:1px solid #dae4ec}}pre{{white-space:pre-wrap;font-size:13px}}.flow{{padding:20px;background:#e8f3ef;border-radius:12px;font-weight:bold}}@media(max-width:700px){{.metrics{{grid-template-columns:1fr}}main{{padding:16px}}section{{padding:18px}}}}</style><main>
<p class="tag">공개 자료 검증 · 손 검출 부품 · 기존 C/T3-F 보존</p><h1>놓쳤을 때만 YOLO를 부르면<br>손을 더 안전하게 찾을 수 있을까?</h1>
<section><h2>결론: 작은 보완 효과, 정밀 복원은 아직 미확인</h2><p>{conclusion}</p><div class="metrics"><div><b>+1장면</b>기존 평가 범위의 손목 지원</div><div><b>139 / 4,781</b>기존영상 YOLO 호출프레임</div><div><b>기존값 보존</b>101,680관절-프레임 / 교차35프레임</div></div><p class="call">기존 보류5장면은 해결되지 않았습니다. 추가1장면은 지난 비교에서도 YOLO의 손목 근거가 확인됐던 R24입니다. 새 검출 능력을 발견한 것과 누락 전용 흐름에 연결한 효과를 구분합니다.</p></section>
<section><h2>바뀐 처리 방식</h2><p class="flow">원본 전체 화면 MediaPipe → MediaPipe 재검출 → 그래도 비어 있는 손만 YOLO → 기존 손 보존 → 마지막에 고정 상반신 범위 적용</p><p>내려놓은 손이 상반신 범위 밖에 있다는 이유만으로 호출하지 않습니다. 전체 원영상 관측에서도 해당 손이 없고, 현재 팔·손목이 범위 안에 보일 때만 호출합니다. 손목 위치와 손가락 모양의 근거를 따로 저장합니다.</p><p>전체 화면과 확대 화면의 손목만 일치하면 손목1점만 후보로 남깁니다. 나머지20점을 채우거나 손가락 굽힘을 맞았다고 계산하지 않습니다. 좌우·중복이 애매하면 그대로 보류합니다.</p></section>
<section><h2>같은 장면에서 비교한 결과</h2><table><tr><th>조건</th><th>누락 문제 표본</th><th>기존 점검 표본</th></tr>{rows}</table><p>각각 23개와16개 판독 가능한 장면의 손목 범위 일치 수입니다. 판독불가3개는 별도 보류했습니다. <b>단어 인식률이나 손가락 정확도는 아닙니다.</b></p><p>전체 영상에서 손 전체 근거1프레임과 손목만6프레임이 추가됐습니다. 손목만6개는 기존 평가 주석이 없어 정확도 개선으로 세지 않았습니다. 새 범위 밖 추가0·기존 성공 손실0은 주석이 있는42장면 안의 결과입니다.</p></section>
<section><h2>처리 부담</h2><table><tr><th>측정 범위</th><th>결과</th></tr><tr><td>기존39영상에서 YOLO 전체 화면 호출</td><td>4,781회 → 139회 ({ratio:.1f}% 감소), 확대검출141회 별도</td></tr><tr><td>새 YOLO 비용 · 전체프레임 평균</td><td>{lat['new_selective_yolo_over_all_frames']['mean_ms']:.2f}ms 추가</td></tr><tr><td>실제로 호출한 프레임의 추가비용</td><td>평균 {lat['new_selective_yolo_on_called_frames']['mean_ms']:.1f}ms / p95 {lat['new_selective_yolo_on_called_frames']['p95_ms']:.1f}ms</td></tr><tr><td>이전MP재검출 + 새YOLO 부품합산 추정</td><td>70.0 → {lat['M1_plus_selective_components']['mean_ms']:.1f}ms/프레임</td></tr></table><p class="call">평균 비용은 작아졌지만 호출한 순간에는 추가 지연이 큽니다. CPU 부품 시간이며 실제웹캠FPS나 GPU 성능이 아닙니다.</p></section>
<section><h2>정상 손과 교차는 유지됐는가?</h2><p>전체46영상5,637프레임의 MediaPipe 재검출 좌표101,680개를 그대로 보존했습니다. 기존 교차35프레임, 시간축과 전체 프레임 수도 유지했습니다. 손이 안 보인다고 비교 분모나 영상 구간을 줄이지 않았습니다.</p><p>이번 YOLO 비교에 새로 추가한 촬영자07의7영상856프레임에서는1프레임만 호출했고 추가 손은 없었습니다. 이 결과는 보존 확인이며 새 사람의 복원 정확도를 입증하지 않습니다. 기존 좌표 보존은 기존 오류까지 모두 고쳤다는 뜻도 아닙니다.</p></section>
<section><h2>보류됐던5장면: 무엇이 불확실한가?</h2><p>녹색은 전체화면 YOLO, 주황색은 확대검출입니다. 아래 확대는 결과를 보여주기 위한 것입니다. 검출은 원본 전체화면을 먼저 사용하며 좌표는 원본 크기로 되돌립니다. 두 검출이 일치해도 정답을 보증하지 않습니다.</p>{auditcards}</section>
<section><h2>이번에 추가된7개 손-프레임 전체</h2><p>1개는 손전체 비교근거,6개는 손목만 남겼습니다. 육안 검토는 예측을 본 상태에서 한 보조 확인으로 독립 정답을 대신하지 않습니다. 특히 겹친 손에서 손목점이 주먹 표면 쪽에 치우쳤는지는 정밀 검증이 남아 있습니다.</p>{addcards}</section>
<section><h2>다음에 확인할 것</h2><p>{nextstep}</p><p>현재 C·T3-F와 웹/DTW는 교체하지 않았습니다. 사용자·멘토 촬영을 열거나 학습·튜닝에 사용하지 않았습니다. 학습과 TCN 추론도 하지 않았습니다.</p><p><a href="VERIFY_SAVED.json">저장 결과 별도 검산</a> · <a href="FROZEN.json">결과 전 고정 조건</a> · <a href="FIVE_SCENE_AUDIT.json">5장면 수치</a></p><details><summary>상세 기록</summary><pre>{e(text)}</pre></details></section></main></html>'''
(U/'RESULT_KO.html').write_text(page,encoding='utf8');print('REPORT_READY')
