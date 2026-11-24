from pathlib import Path
import json,hashlib,html,datetime
T=Path(__file__).resolve().parent;S=T.parent/'word100_reference_speed_cause_20261026';U=T.parent/'word100_train_reference_bank_20261026'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert read(T/'BASELINE_EQUIVALENCE.json')['pairs']==799;v=read(T/'VERIFIED.json');d=read(T/'DONE.json');assert v['result_sha256']==sha(T/'DONE.json');assert read(S/'VERIFIED.json')['result_sha256']==sha(S/'DONE.json');counts=d['counts'];c=read(S/'DONE.json')['counts'];normal=read(U/'ANALYSIS.json')['normal'];rows=[];textrows=[]
 labels={'original':'원본 여러 수행자','speed075':'0.75배 — 동일 관측 반복','speed125':'1.25배 — 일부 시점 생략','delete3':'중간 3시점 삭제','missing_L':'왼손 전체 미관측','missing_R':'오른손 전체 미관측','gap3_L':'왼손 3시점 미관측','gap3_R':'오른손 3시점 미관측'}
 for kind,label in labels.items():
  x=counts[kind];key='sequence_position';den=x[key+'_domain'];old=x[key+'_old_new_excess'];new=x[key+'_candidate_new_excess'];lost=x[key+'_old_supported_now_unknown'];rows.append(f'<tr><td>{label}</td><td>{x["pairs"]:,}</td><td>{100*old/den:.2f}%</td><td>{100*new/den:.2f}%</td><td>{lost:,}</td></tr>');textrows.append(f'{label}: 비교{x["pairs"]}쌍, 새 범위 밖 {old}/{den}({100*old/den:.2f}%) → {new}/{den}({100*new/den:.2f}%), 기존 비교 가능→부족 {lost}칸.')
 slow=counts['speed075'];fast=counts['speed125'];old=slow['sequence_position_old_new_excess'];new=slow['sequence_position_candidate_new_excess'];reduction=old-new
 text='''시간 정렬 보완 — 실제로 고쳐진 범위와 남은 문제

목표는 어떤 수어인지 맞추는 기능과, 확인 가능한 동작 차이를 설명하는 기능을 함께 만드는 것이다. MediaPipe는 몸·손 관절의 기본 관측, YOLO는 기본/재검출이 놓친 경우 연구한 선택적 보조다. YOLO 결과 자체를 정답으로 믿거나 모든 프레임에 새로 실행하는 것은 아니다. 이번 작업은 새 YOLO/신경망 학습/TCN 추론 없이 DTW의 시간 비교를 보완했다.

문제: 원래 동일한 동작의 속도만 바꿨는데도 새 차이가 나타났다. 느린 시험은 같은 관측을 반복 저장하고, 빠른 시험은 일부 관측을 건너뛴다. 수행자가 수어를 두 번 반복했다는 뜻이 아니다.
원인 검산: 원래 대응 시점을 알려주면 느린 시험의 새 차이는 사라졌고 빠른 시험도 크게 줄었다. 다만 이 정답 시점은 실제 사용자에게는 없으므로 원인 진단으로만 사용했다.

실제 후보: 좌표 특징·각도·관측 여부·이미지 재확인 상태·핵심 구간 여부가 모두 같은 연속 입력만 시간 탐색에서 묶는다. 조금 다른 동작, A→B→A 반복, 관측 상태 변경은 묶지 않는다. 이후 원래 모든 시간과 관측 구간으로 되돌려 비교하므로 손이 안 보인다고 비교 범위를 줄이지 않는다. 원본 시점 정답을 이 후보에 입력하지 않았다.

원본899개에서 해당 중복 행은0이었다. 따라서 원본7184비교와 빠른속도/삭제/미관측 대조는 새로 계산하지 않고 저장 결과를 정확히 재사용했다. 새로 정렬한 것은 느린속도799비교다. 총12777비교의 원본 재사용과 새799경로를 별도 수식으로 검산했다.

연속 손·팔 위치 비교 — 원본에서 범위 안이던 칸이 새로 범위 밖이 된 비율:
'''+ '\n'.join(textrows)+f'''

느린 시험을 기존 속도변형 결과와 비교하면 순간각도10124칸·연속각도4124칸·순간위치1378칸·연속위치1318칸은 비교 가능에서 근거 부족으로 바뀌었다. 이를 개선으로 세지 않았다. 후보의799개 느린 시험 상태·지원여부는 변형 전 원본과 정확히 같았고 원본 차이는2e-6 허용오차 내에서 같았다. 즉 기존 속도 정렬이 다른 관측으로 연결했던 효과도 있다.\n\n느린 중복 표본 시험에서 새 차이 {reduction:,}칸이 감소했다. 이것은 단어 인식률 상승이나 실제 사용자 관절 정확도 향상이 아니다. 자연스럽게 느리게 움직이는 새 RGB 영상에서의 일반화도 아직 검증되지 않았다. 빠른 샘플에서 사라진 세부 동작은 이 보완만으로 복구되지 않으며, 중간3시점 삭제나 가림을 구별하는 기존 한계도 그대로 남는다.

다중 참고 자체도 한계가 있다. 순간 손가락 각도는 전체 비교칸 중 {normal['point_angle']['sufficient_percent']}%, 연속 각도는 {normal['sequence_angle']['sufficient_percent']}%에서만 현재 관측과 최소5명의 참고 근거가 모두 충분했다. 위치 근거와 손가락 근거를 분리해야 한다. 참고들의 관측 잔차 최댓값은 수어 정답 허용범위 인증이 아니므로 범위 밖을 그대로 교정 명령으로 바꾸지 않는다.

다음: 검증된 정확중복 처리만 격리된 연구 피드백 경로에 연결하고, 빠른 입력의 표본 손실과 서로 다른 사람의 시간 대응 문제를 따로 다룬다. 기존 순간 차이·관측 부족 시간대는 유지한다. 정밀 손가락 교정, 실사용 속도와 정상 오류 구분, 운영 웹 적용은 완료하지 않았다. 기존C/T3-F/E3·원본·운영 기본을 보존했고 사용자/멘토 자료 및 추가199/200은 사용하지 않았다.

구현 및 별도 산술검산: Codex. 독립 수어 전문가의 정답 검증은 아니다. 전체 개선 목표는 계속 진행한다.
'''
 document='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>시간 정렬 보완 검증</title><style>body{background:#eff3f7;color:#183349;font:17px/1.75 system-ui,"Malgun Gothic",sans-serif;margin:0}main{max-width:1050px;margin:auto;padding:40px 22px}h1{font-size:36px;line-height:1.3}h2{font-size:24px}section{background:white;border:1px solid #d8e3eb;border-radius:16px;padding:26px;margin:22px 0}.note{background:#fff2d8;border-left:5px solid #d29a35;padding:16px}.good{background:#e3f1ed;padding:16px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:15px}th,td{padding:12px 8px;text-align:left;border-bottom:1px solid #dfe7ee}th{background:#edf4f8}a{color:#126b98}small{color:#5d6f7e}</style><main><small>WORDS IN SILENCE · 공개 자료 연구 후보</small><h1>같은 동작인데<br>비교 시점이 달라 생긴 차이 줄이기</h1><p>손 좌표나 모델 점수를 바꾸지 않고 시간 정렬의 한 가지 원인을 보완했다.</p><section><h2>무엇을 해결하려 했나</h2><p>느리게 만든 영상 입력에는 같은 관측이 여러 번 들어간다. 기존 DTW가 이를 다른 경로로 맞추면서 같은 동작도 조금 다른 자세끼리 비교하는 문제가 있었다. 수어 동작 자체를 여러 번 반복했다는 뜻은 아니다.</p><p>이번 후보는 완전히 같은 연속 관측만 탐색에서 묶고, 계산 후 원래 모든 시간과 관측 근거로 되돌린다. 손 모양·위치·관측 상태가 달라지거나 A→B→A로 움직이면 그대로 남긴다.</p><div class="note">원본의 정답 시점을 알려준 진단과, 정답 시점을 쓰지 않는 실제 후보를 따로 비교했다. 두 결과를 같은 개선으로 세지 않았다.</div></section><section><h2>실제 후보 비교 결과</h2><p>원본100단어·899영상의 여러 수행자 비교7,184건과 속도·누락 통제5,593건을 확인했다. 바뀔 필요가 없는 저장 결과는 그대로 재사용했다.</p><div class="scroll"><table><thead><tr><th>조건</th><th>비교 쌍</th><th>기존 새 범위 밖</th><th>후보 새 범위 밖</th><th>기존 가능→부족</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div><p>비율의 분모는 각 참고의 핵심시간×양손×위치 항목이다. 같은 영상이 여러 참고와 비교되므로 사람 수나 단어 정답률이 아니다. ‘범위 밖’도 실제 수어 오류 판정과 다르다.</p></section><section><h2>좋아진 점과 남은 한계를 분리</h2><div class="good">동일 관측이 반복 저장된 느린 입력의 정렬 흔들림을 줄이는 공통 처리를 검증했다. 원래 정상 영상들의 좌표·좌우 배정·비교 결과는 유지했다.</div><p>이전 느린 정렬 대비 근거 부족으로 바뀐 칸도 있다(순간각도10,124·연속각도4,124·순간위치1,378·연속위치1,318). 이를 개선으로 세지 않았다. 후보의 느린시험799개 상태·지원은 변형 전 원본과 정확히 같다.</p><p>빠른 입력에서 실제 시점이 빠진 경우, 자연스러운 수행 속도 차이, 손 가림이나 손가락의 정확한 모양은 이 처리 하나로 해결되지 않는다. 원본 자료의 반복 행이 없었으므로 이번 보완이 원본 인식률을 높였다고 말할 수도 없다.</p><p>다중 참고 범위도 충분한 관측이 있는 항목에만 표시한다. 확인할 수 없는 손가락은 ‘근거 부족’으로 남기며 잘 맞았다고 표시하지 않는다.</p></section><section><h2>프로젝트에서의 역할</h2><p>MediaPipe가 몸과 손을 기본 관측하고, 과거 검증한 YOLO 보조는 필요한 경우의 후보 근거로 취급한다. TCN은 단어를 분류하고 DTW는 연습 목표의 참고들과 동작 차이를 비교한다. 이번 단계에서는 새YOLO·TCN추론·학습을 실행하지 않았다.</p><p>기존C·T3-F·E3, 운영 웹과 사용자 촬영은 그대로다. 검증한 부분만 격리 연구 경로에 연결하고 남은 시간 정렬 문제를 계속 다룬다.</p></section><section><h2>재현 근거</h2><p><a href="VERIFIED.json">전수 검산</a> · <a href="PREFLIGHT.json">변화·누락 보존 사전시험</a> · <a href="../word100_reference_speed_cause_20261026/VERIFIED.json">원인 분리 검산</a> · <a href="../word100_train_reference_bank_20261026/RESULT_KO.html">899영상 다중 참고 전체 결과</a> · <a href="RESULT_KO.txt">상세 텍스트</a></p><small>같은 Codex가 별도 수식으로 검산했으며 독립 사람 정답 검증은 아니다. 전체 프로젝트 목표는 아직 진행 중이다.</small></section></main></html>'''
 for name,value in [('RESULT_KO.txt',text),('RESULT_KO.html',document)]:
  with (T/name).open('x',encoding='utf8') as fp:fp.write(value)
 links=['VERIFIED.json','PREFLIGHT.json','../word100_reference_speed_cause_20261026/VERIFIED.json','../word100_train_reference_bank_20261026/RESULT_KO.html','RESULT_KO.txt'];assert all((T/p).is_file() for p in links)
 manifest=dict(at=datetime.datetime.now().astimezone().isoformat(),html_sha256=sha(T/'RESULT_KO.html'),text_sha256=sha(T/'RESULT_KO.txt'),source_sha256=sha(T/'VERIFIED.json'),links_exist=True,browser_rendered=False,goal_complete=False)
 (T/'REPORT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print('REPEAT_SAMPLE_REPORT_READY')
if __name__=='__main__':main()

