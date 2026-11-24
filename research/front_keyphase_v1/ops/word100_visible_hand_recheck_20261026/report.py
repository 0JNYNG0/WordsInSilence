import prepare as R
from pathlib import Path
from html import escape as e
import json
U=R.U
def main():
 a=R.read(U/'IMAGE_VERIFIED.json');p=R.read(U/'PIXEL_GUARD_VERIFIED.json');v=R.read(U/'PIXEL_GUARD_RESULT.json');f=R.read(U/'PREPARED_VERIFIED.json');review=R.read(U/'VISUAL_REVIEW.json');assert a['clips']==p['clips']==100
 c=a['counts'];d=p['counts'];idx=R.read(U/'visual_100/INDEX.json');pct=lambda x,y:f'{x/y*100:.2f}%' if y else '해당 없음'
 lines=[
 '보이는 손을 다시 확인하는 공개 100단어 시험',
 'MediaPipe 기본 관측은 그대로 두고, 손이 재등장·부분 관측·겹침·급격한 좌표 변화로 재확인이 필요한 순간을 공통 규칙으로 골랐다. MediaPipe 전체 장면 및 손 주변 재검출을 먼저 확인하고, 손가락 비교 근거가 남지 않는 부분에만 손 전용 YOLO를 추가했다.',
 f'준비: 공개 TRAIN899(기록ID1~9)로 요청 기준만 고정. 단어 정답·인식 점수는 요청 규칙에 넣지 않았다. 각 단어 1영상씩 고정 해시로 선택한 100영상, {c["frame_targets"]:,}장면, 요청된 손-프레임 {c["hand_frames"]:,}건을 비교했다. 여러 ID가 섞인 개발 표본이며 새 독립 평가가 아니다.',
 f'요청된 손에서 원래 계산 가능한 각도는 {c["native_angles"]:,}개였다. MediaPipe 재확인과 유사변환을 적용한 기존 형태 비교가 통과한 각도 {c["mp_corroborated"]:,}개, 선택적 YOLO까지 함께 본 값은 {c["fused_corroborated"]:,}개. YOLO가 추가한 {c["added_by_yolo"]}개와 충돌 등으로 보류된 {c["held_after_yolo"]}개를 따로 남겼다. 이 수치는 정답률이 아니다.',
 '중요한 문제: 손 모양 비교에서 한 검출 결과를 이동·회전·확대하여 맞추면, 실제 이미지에서 작게 뭉쳐 잡힌 손도 일부 비슷한 모양으로 처리될 수 있었다. 따라서 예측 간 모양 일치만으로 정밀 손가락 교정에 사용하면 안 된다.',
 f'별도 보완 검산: 같은 원본 장면에서는 좌표를 이동·회전·확대하지 않고 원래 영상 위치에서 기존/전체 재검출/손 주변 재검출 세 쌍의 차이를 검사했다. 원래 모양 근거는 별도로 보존했다. 직접 이미지 근거가 남은 MediaPipe 각도 {d["pixel_mp"]:,}개, YOLO 각도 {d["pixel_yolo"]:,}개, 결합 {d["pixel_fused"]:,}개. 추가 YOLO 근거는 {d["pixel_yolo_added"]}개. 적어진 근거를 인식률 하락이나 사용자 오류로 해석하지 않는다.',
 '정상 손과 교차: 모든 원본 좌표·좌우 배정·범위·시점·손 크기 기준을 변경하지 않았다. 겹침 자체는 오류가 아니라 재확인 요청이다. 불확실성을 별도 표시하며 정상 동작을 틀렸다고 판정하지 않는다. 수치 보존 검산은 해부학적 정답 검증과 다르다.',
 f'시각 검토: 공개 장면 {len(review["reviewed_images"])}개를 직접 확인했다. 이는 동일 Codex의 영상 판독이며 독립 사람의 정확한 21점 정답 라벨은 아니다. 영상에서 숨겨진 손가락을 맞혔다고 주장하지 않는다.',
 '처리 시간: 새 검출 호출과 과거 동일 결과 재사용을 분리했다. 아래 시간은 이 PC의 개별 호출 시간이며 웹캠 전체 FPS가 아니다.',
 '결론: 보이는 손도 재확인할 수 있게 되었고, 위치·크기 차이를 모양 정규화가 숨기는 문제를 새로 드러냈다. YOLO는 선택적 연구 보조이며 모든 결과를 정답으로 채택하지 않는다. 기존 C·T3-F·운영 웹/DTW는 그대로다. 이번 시험의 신규 학습과 TCN 재채점은 0회다. 사용자·멘토 촬영 및 추가199/200 자료를 접근하거나 사용하지 않았다.',
 '다음: 기존 순간 차이는 유지하면서 재확인 미실행/관측 없음/예측 불일치/추가 근거 있음 상태를 연구용 DTW에 연결한다. 현재 이미지 보완이 실제 다른 사람과의 비교에서 충분한 근거를 남기는지 확인한다. 정밀 손가락 피드백 전에는 보이는 관절의 영상 판독 근거와 비교하고, 숨겨진 부분은 추정으로 남긴다. 단어 인식 개선은 별도 고정 비교에서 확인해야 한다.'
 ]
 for k,t in a['latency'].items():lines.append(f'{k}: 새 호출 {t["n"]:,}회, 평균 {t["mean_ms"]:.2f}ms, 중앙값 {t["median_ms"]:.2f}ms, 95백분위 {t["p95_ms"]:.2f}ms')
 (U/'RESULT_KO.txt').write_text('\n\n'.join(lines),encoding='utf8')
 rows=''
 for name,key in [('전체 요청 손','all'),('수어 구간','core'),('균등 시점 대조','control'),('손 영역 겹침 요청','crossing_request')]:
  g=v['groups'][key];rows+=f'<tr><td>{name}</td><td>{g["native_angles"]:,}</td><td>{g["old_mp"]:,}</td><td>{g["pixel_mp"]:,}</td><td>{g["pixel_fused"]:,}</td></tr>'
 cards=''.join(f'<details><summary>{x["number"]:02d} {e(x["label"])} · {e(x["category"])} · 프레임 {x["index"]}</summary><img loading="lazy" src="visual_100/{Path(x["image"]).name}" alt="공개 원영상과 검출 결과 비교"><p>{e(x["clip_id"])} / {"왼손" if x["side"]==0 else "오른손"} · 표시용 확대이며 원본 전체 화면 검출을 유지했다.</p></details>' for x in idx['scenes'])
 document='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>보이는 손 재확인 · 공개100단어</title><style>*{box-sizing:border-box}body{margin:0;background:#101824;color:#e8eef5;font:17px/1.75 system-ui,"Malgun Gothic",sans-serif}main{max-width:1120px;margin:auto;padding:45px 28px}h1{font-size:36px;line-height:1.3}h2{margin-top:38px;font-size:24px}p{max-width:1000px}b{color:#67e1cf}.tag{color:#8fb4df}section,details{background:#182637;border:1px solid #304357;border-radius:14px;padding:22px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:12px;border-bottom:1px solid #3c4b5d;text-align:right}th:first-child,td:first-child{text-align:left}summary{cursor:pointer;font-weight:700}img{max-width:100%;display:block;margin:14px auto}small{color:#bccadc}.warning{border-color:#bb8c4d}a{color:#88c9ff}@media(max-width:650px){main{padding:22px 14px}h1{font-size:28px}table{font-size:12px}td,th{padding:5px}}</style><main><div class="tag">연구용 비교 · 기존 C / T3-F 보존 · 운영 미적용</div>'''
 document+=f'<h1>{e(lines[0])}</h1><p>{e(lines[1])}</p><section><b>실행 완료</b><p>{e(lines[2])}</p></section><h2>재확인을 했다고 정답이 되는 것은 아니다</h2><p>{e(lines[3])}</p><section class="warning"><p>{e(lines[4])}</p></section><p>{e(lines[5])}</p>'
 document+='<h2>같은 입력에 남는 비교 근거</h2><p>각도 × 손 × 시점의 개수. 정확도·사람 수·올바른 동작 비율이 아니다. 대조와 겹침 범주는 일부 중복된다.</p><table><thead><tr><th>범위</th><th>원래 계산 가능</th><th>MP 모양 비교 통과</th><th>MP 이미지 위치 검사</th><th>MP+YOLO 이미지 검사</th></tr></thead><tbody>'+rows+'</tbody></table>'
 document+=''.join(f'<p>{e(t)}</p>' for t in lines[6:10])+''.join(f'<p>{e(t)}</p>' for t in lines[11:])+f'<h2>다음 작업</h2><p>{e(lines[10])}</p><h2>실제 공개 장면 비교</h2><p>목표 단어에 맞춰 관절을 변경하지 않았다. 원본/기존 좌표/MP 전체·손 주변/YOLO 전체·손 주변을 따로 표시한다.</p>'+cards+'<h2>검증 근거</h2><p>IMAGE_VERIFIED.json: 별도 복소수 기하·각도·좌우 연결 계약·원본 해시 검산.<br>PIXEL_GUARD_VERIFIED.json: 원본 이미지 좌표의 별도 산술 검산.<br>같은 Codex의 별도 코드 검산이며 독립적인 사람 정답 검증은 아니다.</p></main></html>'
 (U/'RESULT_KO.html').write_text(document,encoding='utf8');assert 'nan' not in document.lower() and len(idx['scenes'])>=18
 R.write(U/'REPORT_CHECK.json',dict(at=R.now(),html_sha256=R.sha(U/'RESULT_KO.html'),text_sha256=R.sha(U/'RESULT_KO.txt'),all_image_links_exist=all(Path(x['image']).exists() for x in idx['scenes']),status='SOURCE_AND_IMAGE_QA',browser_render_verified=False));print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
