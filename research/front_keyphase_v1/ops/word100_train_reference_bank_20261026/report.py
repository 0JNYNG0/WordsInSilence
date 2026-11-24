import prepare as R
import html,json
from pathlib import Path

def main():
 a=R.read(R.U/'ANALYSIS.json');assert R.read(R.U/'ANALYSIS_VERIFIED.json')['analysis_sha256']==R.sha(R.U/'ANALYSIS.json');R.check(a['source_pins']);iv=R.read(R.U/'INPUTS_VERIFIED.json');im=R.read(R.U/'VERIFIED_IMAGES.json');c=iv['counts'];names={'point_angle':'순간별 손가락 각도','sequence_angle':'앞·중간·뒤 손가락 변화','point_position':'순간별 손·팔 위치','sequence_position':'앞·중간·뒤 손·팔 위치 변화'}
 normal=[];bars=[]
 for key in ['point_angle','sequence_angle','point_position','sequence_position']:
  q=a['normal'][key];s=q['states'];within=100*s['4']/q['domain'];beyond=100*s['5']/q['domain'];unknown=100-within-beyond
  normal.append(f"<tr><td>{names[key]}</td><td>{q['sufficient_percent']:.2f}%</td><td>{q['within_observed_percent_of_supported']:.2f}%</td><td>{'이미지 재추정 일치 근거' if key.endswith('angle') else '관절 위치 관측 근거'}</td></tr>")
  bars.append(f'<div class="barrow"><b>{names[key]}</b><div class="bar"><span style="width:{within}%" class="within"></span><span style="width:{beyond}%" class="beyond"></span><span style="width:{unknown}%" class="unknown"></span></div><small>자료 범위 안 {within:.1f}% · 범위 밖 {beyond:.1f}% · 판단 근거 부족 {unknown:.1f}%</small></div>')
 labels={'speed075':'0.75배 속도','speed125':'1.25배 속도','delete3':'중간 3시점 삭제','missing_L':'왼손 전체 관측 제거','missing_R':'오른손 전체 관측 제거','gap3_L':'왼손 3시점 관측 제거','gap3_R':'오른손 3시점 관측 제거'}
 control=[];deletion=[]
 for kind in labels:
  q=a['controls'][kind];v=q['cell_transitions'];base='sequence_position';fraction=100*v[base+'_within_to_beyond']/max(1,v[base+'_domain']);lost=100*v[base+'_supported_lost']/max(1,v[base+'_domain']);control.append(f'<tr><td>{labels[kind]}</td><td>{q["summary"]["queries"]}</td><td>{q["summary"]["pairs"]}</td><td>{fraction:.2f}%</td><td>{lost:.2f}%</td></tr>')
 for key in ['point_angle','sequence_angle','point_position','sequence_position']:
  s=a['controls']['delete3']['summary'];get=lambda k:s.get(key+'_'+k,0)
  deletion.append(f'<tr><td>{names[key]}</td><td>{get("new_local_excess")}</td><td>{get("new_local_uncertainty_only")}</td><td>{get("no_new_local_signal_with_some_baseline_support")}</td><td>{get("baseline_target_already_insufficient")+get("no_original_target_correspondence")}</td></tr>')
 text=f'''공개 100단어·899영상 다중 참고 검증

이번에는 여러 사람이 수행한 같은 단어를 각각 유지하면서 비교 기준을 만들었다. 시험하는 기록 ID는 기준 계산에서 제외했다. 나머지 참고 중 같은 시간·항목에 최소5명의 비교 근거가 있을 때만 경험적 차이 범위를 계산했다. 이는 올바른 수어의 허용 범위나 일반 사용자 정답률이 아니다.

완료: 기존300영상 재사용+추가599영상 이미지 재확인,899원본/근거 입력 검산,7184서로 다른 수행자 비교,5593관측 수준 속도·삭제·손누락 통제 및별도산술검산. 사용자자료/모델학습/TCN재추론/운영변경은0. 원본좌표·손배정·비교분모 보존.

이미지 근거: 원본각도 {c['raw_angles']:,}칸 중 {c['corroborated_angles']:,}칸에 재추정 일치 근거. 새로 {c['added_angles']:,}칸 추가, 기존근거철회 {c['withdrawn_angles']}칸. 같은 MediaPipe를 전체/확대 이미지에서 다시 실행한 결과와의 일치이며, 독립적인 해부학 정답이 아니다. 위치와 팔 좌표는 raw 관측 근거로 구분.

원본 비교 범위를 만들 수 있었던 비율:
'''+''.join(f"{names[k]}: {v['sufficient_percent']:.2f}%\n" for k,v in a['normal'].items())+'''
위 비율은 기준영상의 핵심시간×양손×항목으로 만든 비교 칸 기준이다. 같은 영상이 여러 참고와 비교되어 중복되므로 독립 사람 수나 단어 인식률로 읽으면 안 된다. 손내림·한 손 동작의 안 보이는 손은 오류로 판정하지 않는다. 비교 가능한 다른 손/팔은 별도 보존한다.

짧은삭제 시험은 입력의3시점을 없앤 구조 시험이며 수어의 필수 동작을 지웠다는 언어적 정답 주석이 아니다. 새차이 없음과 이미근거부족을 구분해서 보고한다. 속도 시험은 이미추출한 좌표의 시간 표본을 바꾼 것이며 빠른동작의 MediaPipe 검출성능을 새로 검사한 것은 아니다.

기존 C·T3-F·E3 및 운영 기본은 보존했다. 정상 속도 변형에서도 연속 위치 비교 칸의 4.93%·5.53%가 새 범위 밖으로 바뀌었다. 자동 오류 경고는 보류하고 시간 정렬 변화와 표본 변화의 영향을 분리해 검증한다. 현재 결과만으로 어떤 손가락을 몇 도 고치라고 자동 지시하지 않는다. 실제 사용자에게 적용하기 전 연구 입력 연결과 알려진 한계를 함께 검증해야 한다.
'''
 for kind in labels:text+='\n'+labels[kind]+': '+json.dumps(a['controls'][kind]['summary'],ensure_ascii=False)+'\n'
 document='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공개 다중 참고 범위 검증</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f7;color:#183044;font:17px/1.7 system-ui,"Malgun Gothic",sans-serif}main{max-width:1180px;margin:auto;padding:42px 24px}h1{font-size:36px;line-height:1.3}h2{font-size:25px;margin-top:0}section{background:white;padding:30px;border-radius:18px;margin:22px 0;border:1px solid #dce4ec}.eyebrow{color:#326c84;font-weight:700;letter-spacing:.08em}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}.metric{background:#eaf2f8;padding:16px;border-radius:12px}.metric strong{display:block;font-size:30px}.note{padding:15px;border-left:4px solid #c98a31;background:#fff5e4}.bar{display:flex;height:24px;background:#e3e8ed;border-radius:5px;overflow:hidden}.within{background:#3c9c88}.beyond{background:#dc9751}.unknown{background:#b3bdc8}.barrow{margin:20px 0}.barrow b{display:block;margin-bottom:6px}small{color:#536778}table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;padding:12px 9px;border-bottom:1px solid #e0e8ee;vertical-align:top}th{background:#eff5f8}img{width:100%;height:auto}a{color:#1b6b90}.scroll{overflow:auto}.legend{display:flex;gap:18px;flex-wrap:wrap}.legend i{display:inline-block;width:13px;height:13px;margin-right:5px}.footer{font-size:14px;color:#607482}@media(max-width:700px){.grid{grid-template-columns:repeat(2,1fr)}h1{font-size:28px}section{padding:18px}}
</style><main><div class="eyebrow">WORDS IN SILENCE · 공개자료 연구 검증</div><h1>여러 사람의 참고를 모으면<br>어디까지 믿고 비교할 수 있을까?</h1><p>관측한 차이, 자료에서 본 차이 범위, 확인하지 못한 부분을 구분했다. 기존 인식 모델을 바꾸거나 사용자 촬영에 맞춘 실험이 아니다.</p><div class="grid"><div class="metric"><strong>100</strong>공개 단어 종류</div><div class="metric"><strong>899</strong>원본 학습 영상</div><div class="metric"><strong>7,184</strong>수행자 간 비교</div><div class="metric"><strong>5,593</strong>속도·누락 통제 비교</div></div>
<section><h2>1. 무엇을 바꿨나</h2><p>기존 관절 좌표와 좌우 손 배정은 그대로 두었다. 손가락 좌표가 실제 영상의 재추정 결과와 일치하는지 확인하는 근거를 늘리고, 같은 단어의 여러 참고와 각각 비교했다. 시험할 사람은 참고 범위를 계산할 때 제외했다.</p><p>각 참고를 다른 사람들과 비교해 나타난 차이의 최댓값을 해당 시점·항목의 경험적 범위로 사용했다. 최소 5명의 근거가 있어야 만들며, 사람들의 골격을 평균내거나 가장 잘 맞는 참고만 고르지 않는다.</p><div class="note">“자료 범위 안”은 이 공개 자료에서 관측한 차이와 비슷하다는 뜻이다. 수어가 올바르다는 판정이나 일반 사용자의 허용 오차 인증이 아니다.</div></section>
<section><h2>2. 비교 기준을 만들 수 있는 부분과 부족한 부분</h2><div class="legend"><span><i class="within"></i>자료 범위 안</span><span><i class="beyond"></i>자료 범위 밖</span><span><i class="unknown"></i>판단 근거 부족</span></div>'''+''.join(bars)+'''<div class="scroll"><table><thead><tr><th>항목</th><th>전체 칸 중 기준·관측 충분</th><th>충분한 칸 중 범위 안</th><th>근거 종류</th></tr></thead><tbody>'''+''.join(normal)+'''</tbody></table></div><p>회색 부분을 “잘 맞았다” 또는 “틀렸다”로 처리하지 않는다. 손가락이 불확실해도 확인 가능한 손 위치·팔 움직임은 별도로 남긴다. 전체 분모는 참고 핵심시간×양손×항목이며, 한 영상이 여러 참고와 비교되므로 사람 수나 인식률과 다르다.</p></section>
<section><h2>3. 속도 차이와 손 누락을 따로 시험</h2><p>중요한 미해결 문제: 정상 속도 변형에서도 연속 위치 비교 칸의 약 4.93%·5.53%가 새로 범위 밖으로 바뀌었다. 이를 그대로 동작 오류 경고에 사용하는 것은 보류한다. 아래는 연속 위치 비교에서 기존 범위 안이 새 범위 밖으로 바뀐 칸과, 원래 가능했던 비교가 불확실해진 칸이다. 불확실해진 것을 개선으로 세지 않았다.</p><div class="scroll"><table><thead><tr><th>입력 조건</th><th>원본 대상 영상</th><th>참고별 비교</th><th>범위 안→밖 / 전체 칸</th><th>비교 가능→부족 / 전체 칸</th></tr></thead><tbody>'''+''.join(control)+'''</tbody></table></div><p>좌표를 추출한 뒤 시간을 바꾸거나 관측을 제거한 통제 시험이다. 실제 빠른 영상의 새 관절 검출 시험 또는 수어 오류 정답 데이터는 아니다. 각 시험에서도 원래 참고 범위와 비교할 전체 시간대를 유지했다.</p></section>
<section><h2>4. 중간 3시점을 없애도 변화가 드러나는가</h2><p>단어마다 고정한 한 원본, 총 100개에서 핵심시간 중앙의 3시점을 제거했다. 각 원본의 모든 참고 결과를 확인했다. 한 참고라도 새 차이가 나오면 첫 칸, 차이는 없고 비교 근거만 줄면 두 번째 칸이다.</p><div class="scroll"><table><thead><tr><th>항목</th><th>새 국소 차이</th><th>새 불확실성만</th><th>기존 근거 있으나 새 신호 없음</th><th>원래부터 대상 근거 부족</th></tr></thead><tbody>'''+''.join(deletion)+'''</tbody></table></div><div class="note">3시점 제거가 반드시 의미 있는 수어 단계 누락인 것은 아니다. 따라서 위 수치를 오류 검출 정확도라고 부르지 않는다. 새 신호가 없는 사례도 감추지 않고 다음 분석 대상으로 남긴다.</div></section>
<section><h2>5. 실제 영상에서 확인한 이유</h2><p>왼쪽은 원본 영상, 오른쪽은 보존한 관절 좌표다. 관절 점이 존재해도 실제 손에서 벗어나거나 겹친 손의 근거가 불명확할 수 있다. 아래 4장면의 시각 확인은 전체 손가락의 해부학적 정답 검증을 대신하지 않는다.</p><img src="review_visual/public_audit.jpg" alt="공개 원본 4장면과 보존된 관절 표시"><p>전체 영상과 손 주변 확대 영상에 같은 MediaPipe를 적용해 확인한 일치는 품질 근거이며, 독립된 정답 확인이 아니다. 영상 확대는 검토 화면 표시용으로도 사용했으며 원래 학습 영역을 변경하지 않았다.</p></section>
<section><h2>6. 현재 의미와 남은 작업</h2><p>원래 차이 수치를 보존하면서 비교 근거가 부족한 부분을 분리하고, 여러 수행자 사이에서 관측한 차이 범위를 붙일 수 있는 연구 자료를 준비했다. 손 위치·팔 위치와 손가락 각도의 근거 수준도 구분했다.</p><p>자동으로 “올바른 수어” 또는 “이 손가락을 몇 도 고치라”고 지시할 수 있다는 증거는 아니다. 정상 속도에서 새로 생긴 차이와 짧은 삭제에서 아무 신호가 없었던 사례를 검토한 뒤, 검증된 범위만 연구용 피드백 연결에 반영한다.</p><p>기존 C·T3-F·E3, 운영 웹, 원본, 사용자 촬영은 그대로다. 이번 신경망 학습·TCN 새 추론·새 YOLO 추론은 0회다. 정상 손과 교차의 좌표를 유지한 것과 그 좌표가 모두 정확하다고 입증한 것은 구분해야 한다.</p></section>
<section><h2>검증 근거</h2><p><a href="INPUTS_VERIFIED.json">899개 입력</a> · <a href="ALIGNMENT_VERIFIED.json">7,184개 시간 정렬</a> · <a href="RANGE_VERIFIED.json">사람별 제외와 범위</a> · <a href="CONTROLS_VERIFIED.json">5,593개 통제 검산</a> · <a href="ANALYSIS.json">전체 집계</a> · <a href="VISUAL_REVIEW.json">원본 4장면 검토</a> · <a href="RESULT_KO.txt">텍스트 보고</a></p><p class="footer">구현 및 별도 수식 검산: Codex. 독립 수어 전문가·기관의 정답 검증이 아니다. 이 화면은 로컬 연구 보고이며 운영 모델 교체가 아니다.</p></section></main></html>'''
 for name,value in [('RESULT_KO.txt',text),('RESULT_KO.html',document)]:
  with (R.U/name).open('x',encoding='utf8') as fp:fp.write(value)
 R.write(R.U/'REPORT_MANIFEST.json',dict(at=R.now(),analysis_sha256=R.sha(R.U/'ANALYSIS.json'),report_html_sha256=R.sha(R.U/'RESULT_KO.html'),report_text_sha256=R.sha(R.U/'RESULT_KO.txt'),source_pins=a['source_pins'],rendered_browser=False,goal_complete=False));print('PUBLIC_REFERENCE_REPORT_WRITTEN',flush=True)
if __name__=='__main__':main()

