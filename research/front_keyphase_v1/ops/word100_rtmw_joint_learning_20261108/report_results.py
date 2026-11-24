"""Report completed fixed experiments; never retrain, re-score or select a seed."""
from common import *
import numpy as np

def pct(x):return f'{100*x:.2f}%'
def spread(xs,digits=2):return f'{min(xs):.{digits}f}~{max(xs):.{digits}f}'
def stage(cases,key):
 return [c[key]['correct'] for c in cases]

def main():
 assert not (U/'RESULT_KO.txt').exists()
 verified=read(U/'FINAL_VERIFICATION.json');assert verified['status']=='PASS'
 ext=read(U/'EXTRACTION_COMPLETE.json');inp=read(U/'INPUT_VERIFICATION.json');tc=read(U/'TCN_COMPARISON.json');tr=read(U/'TCN_TRAINING_COMPLETE.json')
 fb=read(U/'feedback_viewfix/FEEDBACK_COMPARISON.json');upper=read(U/'feedback_viewfix/FEEDBACK_UPPER_VIEW.json');initial=read(U/'FEEDBACK_FULL_VIEW.json')
 control=read(U/'feedback_viewfix/FEEDBACK_CONTROLS.json');ft=read(U/'feedback_viewfix/FEEDBACK_TRAINING_COMPLETE.json');stream=read(U/'TRAINED_STREAM_COMPLETE.json');mt=read(U/'MODEL_TIMING.json')
 lines=['RTMW 관절 입력으로 TCN·세부피드백 모델 학습 및 비교',
 '공개 자료 준비·학습·고정 비교·별도 검산 완료. 운영 모델과 웹8812는 교체하지 않았다.',
 '', '1. 무엇을 했나',
 '공개 영상 → RTMW 관절 추출(한 번) → 같은 관절을 TCN과 관절전용 피드백 신경망에 학습했다.',
 '실행 시험: 촬영과 함께 RTMW 추출 → 저장한 관절값 → TCN 단어 후보 + 손모양·단계 + 학습한 패턴과 DTW 정렬.',
 '이 실행 경로에서는 SignRep 이미지 모델과 참고 영상 9개 재분석을 사용하지 않는다.',
 'TCN은 기존 C 복사 후 미세조정 / 동일 구조 새 학습을 비교. C가 과거 최고라는 뜻은 아니다. 원본 C·T3-F는 보존했다.',
 '피드백은 시간합성곱+GRU이며 새 학습 / 기존 피드백 시간층·속성·설명·패턴층 복사 후 학습을 비교했다. SignRep 백본을 복사한 것이 아니다.',
 '', '2. 어떤 자료인가',
 f'100단어 공개1199영상: 학습899(기록ID01~09), 개발비교300(ID11~13). 원본 {ext["frames"]:,}프레임.',
 f'동일 원본 캐시 {ext["reused"]}영상 재사용. 입력 실패 {len(ext["failures"])}개도 평가 분모에 포함. 원본 SHA/CRC·중복·시각 연결 확인.',
 '피드백은 이 중4단어(조상·이집트·순종·잘하다), 학습36/개발12영상이다. TCN100단어 전체 피드백 완성을 뜻하지 않는다.',
 '손모양·양손높이 7개 지표, 설명에 연결된13단계. 개발36중심프레임의 확인 가능한194개 지표와92단계지점을 비교했다.',
 '7개 지표: 오른검지만 펴기,왼/오른 네손가락 펴기,왼/오른 네손가락 굽히기,오른손/왼손이 더 높은지. 네손가락 굽힘이 엄지까지 올바른 주먹임을 보장하지 않는다.',
 '설명은 단계와 지표로 구조화해 학습에 연결했다. 문장 전체를 자유롭게 이해하는 언어모델을 만든 것이 아니다. 목표단어 정보는 피드백의 설명/패턴 선택에만 쓰고 TCN에는 넣지 않는다.',
 '왼손 펴기 긍정 개발사례는1개이며 오른손 검지만 펴기는 긍정 개발사례0개라 균형평균에서 제외된다.',
 '직접 시각검토 정답은 이전 Codex의 화면검토이며 한국수어 전문가 정답이 아니다. DTW로 옮긴 다른 학습영상 구간은 낮은 가중치의 추정 주석이다.',
 '개발자료는 이미 여러 번 본 자료다. 새 사람을 처음 평가한 최종시험이 아니며, 현재 차이가 재현될지는 추가 독립검증이 필요하다.',
 '', '3. TCN 단어 맞추기: 300영상 중 맞힌 수',
 '구분 | 입력 화면 | 1위 정답 | 3위 안 정답']
 for b in tc['baseline']:
  lines.append(f'기존 {b["condition"]} | {b["view"]} | {b["metrics"]["top1"]}/300 | {b["metrics"]["top3"]}/300')
 for name,views in tc['summary'].items():
  for view,r in views.items():
   lines.append(f'RTMW {name} | {view} | {r["top1"]}/300, 평균 {r["top1_mean"]:.2f} | {r["top3"]}/300, 평균 {r["top3_mean"]:.2f}')
 lines += ['새 후보 대괄호 숫자는 고정3시드 결과 전체이며 가장 높은 것만 채택하지 않았다.',
 'upper_mask는 전체 영상 검출 후 기존 상반신 경계 밖 좌표를 제외한 조건이다. 기존 masked와 같은 공개영상으로 비교했다.',
 'full은 사용자의 실제 카메라 화면 전체를 쓰려는 조건이다. 기존 sensor는 이미지를 잘라 검출했던 별도 조건이어서 full과 검출기만의 차이라고 단정하지 않는다.',
 '', '4. 세부피드백: 무엇을 잘 구분했나',
 '조건 | 손모양·관계 균형평균 | 직접 단계 정답/92(3시드) | DTW 단계 정답/92(3시드)']
 for name,r in fb['summary'].items():
  lines.append(f'{name} | {pct(r["attribute_balanced_mean"])} | {r["direct_stage_correct"]} | {r["dtw_stage_correct"]}')
 lines += ['기존 projection_finetune은 RGB+관절의 저장 결과(상반신 조건), 새2조건은 전체화면 입력 결과다. 완전히 같은 화면 조건 비교는 아래 상반신 진단도 함께 본다.',
 '같은 상반신 입력으로 확인한 수정 후보:']
 for name in ['pose_fresh','pose_transfer']:
  rr=[r for r in upper['rows'] if r['condition']==name]
  lines.append(f'{name}: 지표 평균 {pct(np.mean([r["attributes"]["macro_balanced_accuracy"] for r in rr]))}, 직접단계 {[r["stage_direct_correct"] for r in rr]}/92, DTW단계 {[r["stage_dtw_correct"] for r in rr]}/92.')
 lines += ['이 수치는 손이 펴졌는지·어느 손이 높은지 등의 물리 특징과 단계 구분 성능이다. 수어 전체의 정답률이나 사용자 교정문장의 정확도가 아니다.',
 '단어별 저장 결과(같은 개발영상의3시드 합계이며 영상 수가3배가 된 것이 아님):']
 for r in read(U/'FEEDBACK_BREAKDOWN.json')['rows']:
  lines.append(f'{r["condition"]} {r["label"]}: 확인지표 {r["known_feature_correct"]}/{r["known_feature_n"]}, 직접단계 {r["direct_correct"]}/{r["stage_n"]}, DTW단계 {r["dtw_correct"]}/{r["stage_n"]}.')
 lines += ['조상에서 지원하는 손모양·관계 지표는 개선되었지만, 다른 단어와 DTW 단계는 모두 개선된 것이 아니다. 특히 엄지의 실제 위치 정확도나 모든 교차 상황을 이7개 지표만으로 보장할 수 없다.',
 '', '5. 작업 중 발견하여 고친 입력 문제',
 '첫 피드백 학습은 예전 상반신 마스크 입력만 사용했다. 화면 전체를 넣으면 단계 구분 성능이 떨어졌다.',
 '특정 단어 예외 없이 모든 학습영상에 전체화면/상반신 입력을 번갈아 사용하도록 공통 조건을 바로잡았다. 정규화는 학습자료만 사용하고 주석은 바꾸지 않았다.',
 '첫6개 모델·점수는 보존하고, 수정6개 모델을 별도 폴더에 학습했다. 개발진단을 보고 입력 문제를 찾았다는 이력도 남겼다.',
 '수정 전 전체화면 직접단계 → 수정 후 전체화면 직접단계:']
 for name in ['pose_fresh','pose_transfer']:
  before=[r['stage_direct_correct'] for r in initial['rows'] if r['condition']==name]
  lines.append(f'{name}: {before} → {fb["summary"][name]["direct_stage_correct"]} /92')
 lines += ['', '6. 속도·누락·순서 변화 확인',
 '0.75/1.25 속도,2프레임/30프레임 손 누락,주요동작 역순,5프레임 삭제를 고정한432개 공학 제어를 비교했다.',
 '이는 좌표열을 바꾼 시험이며 실제 카메라의 빠른 동작·흐림이나 수어 학습자의 잘못된 행동 정답은 아니다.',
 '조건 | 시험 | 확인지표 정답/분모 | 직접단계 정답/분모 | DTW단계 정답/분모']
 groups={}
 for r in control['rows']:groups.setdefault((r['condition'],r['kind']),[]).append(r)
 controls=[]
 for (name,kind),rr in groups.items():
  z=dict(condition=name,kind=kind,rows=len(rr),feature_correct=sum(q['correct'] for r in rr for q in r['center_labels']),feature_n=sum(q['known'] for r in rr for q in r['center_labels']),direct_correct=sum(q['prediction']==q['truth'] for r in rr for q in r['stage_labels']),dtw_correct=sum(q['dtw_prediction']==q['truth'] for r in rr for q in r['stage_labels']),stage_n=sum(len(r['stage_labels']) for r in rr))
  controls.append(z);lines.append(f'{name} | {kind} | {z["feature_correct"]}/{z["feature_n"]} | {z["direct_correct"]}/{z["stage_n"]} | {z["dtw_correct"]}/{z["stage_n"]}')
 lines += ['속도는 대체로 유지되지만 순서를 뒤집으면 특히 DTW 단계 대응이 무너진다. 역순을 정확한 수정문장으로 검출했다는 뜻은 아니다.',
 '긴 미관측은 정상으로 확정하지 않고 별도 부족표시. 짧은 누락도 실제 관측과 시간 흐름 추정을 구별한다.',
 'RTMW가 가려진 손에도 좌표를 추정할 수 있으므로 좌표 존재·확신점수만으로 손이 실제 보였다고 판단할 수 없다.',
 '', '7. 걸린 시간',
 f'공개 {ext["frames"]:,}프레임 입력 준비: {ext["wall_seconds"]/60:.1f}분. 사용자 한 번의 대기시간이 아니다.',
 f'TCN 학습1회(80epoch): {spread([r["training_seconds"] for r in tr["runs"]],1)}초, 총6회.',
 f'수정 피드백 학습1회(120epoch): {spread([r["training_seconds"] for r in ft["runs"]],1)}초, 총6회.',
 f'관절이 준비된 뒤 두신경망+DTW 계산: 반복중앙값 범위 {spread([r["warm_total_p50_ms"] for r in mt["rows"]])}ms. 카메라·관절추출·화면표시는 제외.',
 '공개영상 실시간 재생+녹화+촬영 중 RTMW+학습된 두모델의 실제 연결 시험:']
 timing=[]
 for r in stream['runs']:
  first=r['conditions'][0];timing.append(dict(clip_id=r['clip_id'],frames=r['frames'],post_last_capture_ms=first['post_last_capture_ms'],post_actual_stop_ms=first['post_actual_stop_ms'],condition=first['condition']))
  lines.append(f'{r["clip_id"]}: {r["frames"]}프레임 모두보존, 마지막 예정프레임→분석완료 {first["post_last_capture_ms"]/1000:.3f}초 / 실제종료→완료 {first["post_actual_stop_ms"]/1000:.3f}초.')
 lines += [f'RTMW 초기준비 {stream["engine_compile_seconds"]+stream["engine_warm_seconds"]:.2f}초와 신경망4조합 준비 {stream["neural_load_seconds"]:.2f}초는 촬영 전에 별도.',
 '위 실측은 결과를 보기 전 고정한 첫조합 finetune_C+pose_transfer다. 나머지3조합 시간은 공통처리+각 계산시간의 추정으로 별도 저장했다.',
 '동일 노트북의 공개재생 시험이며 실제 웹캠·Unity·브라우저 렌더링까지의 사용자 대기시간을 검증한 것은 아니다.',
 '', '8. 완료 범위와 남은 문제',
 '완료: RTMW 공통입력 준비,100단어 TCN 두 학습방법 비교,4단어 관절전용 피드백 신경망 학습,입력오류 수정,누락·속도·순서 제어,촬영 중 재사용 연결과 속도검사.',
 '남음: 중요한 실제 오동작과 허용되는 차이를 구분할 검증 정답,짧은 반복/누락의 교정 정확도,이마·코·광대 등 자세한 부위관계,다른 사용자·촬영환경,웹 화면 연결과 RTMW 조건의 손내림 자동종료.',
 '현재 출력은 손모양/단계 관측과 정렬 결과다. normal_or_wrong은 미확정으로 두며 아직 검증되지 않은 정답/오답이나 교정문장을 자동 확정하지 않는다.',
 '다음 단일 작업: 이번에 고정한 관절전용 후보를 실제 촬영 전달·저장시간축과 연결하고,4단어의 중요한 손모양·순서 차이를 시간별 정답과 대조해 교정 가능한 범위를 좁게 검증한다. 새 단어 확대나 운영 교체보다 먼저 한다.',
 '', '9. 보존과 검산',
 f'별도 저장값 계산·해시·시각 검산 {verified["checks_total"]}건 PASS. 같은 Codex가 수행한 별도 코드 검산이며 독립 사람/전문가 검증을 뜻하지 않는다.',
 '사용자·멘토 영상 접근/학습/튜닝/후보선택0. 카메라/새다운로드/외부전송/Git0. 원본 C·T3-F 및 기존8812 보존.',
 '특정 단어 예외나 개발 점수로 고른 시드·임계값은 적용하지 않았다. 개선과 퇴행을 모두 보존했다.',
 '', '근거: SOURCE_AUDIT.json, FROZEN.json, INPUT_VERIFICATION.json, TCN_COMPARISON.json, feedback_viewfix/FEEDBACK_COMPARISON.json, MODEL_TIMING.json, TRAINED_STREAM_COMPLETE.json, FINAL_VERIFICATION.json.']
 (U/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 write(U/'RESULT.json',dict(status='COMPLETE_RESEARCH_COMPARISON_NOT_DEPLOYED',tcn=tc['summary'],feedback_full=fb['summary'],feedback_controls=controls,actual_stream=timing,
  input_clips=1199,input_frames=ext['frames'],input_failures=ext['failures'],verified_checks=verified['checks_total'],production_changed=False,educational_corrections_validated=False,
  pins={str(p.relative_to(U)):sha(p) for p in [U/'FROZEN.json',U/'TCN_COMPARISON.json',U/'feedback_viewfix/FEEDBACK_COMPARISON.json',U/'TRAINED_STREAM_COMPLETE.json',U/'FINAL_VERIFICATION.json']}))
 write(U/'CODE_MANIFEST.json',{str(p.relative_to(U)):sha(p) for p in U.rglob('*.py')})
 event('final_report_written',report=str(U/'RESULT_KO.txt'))
if __name__=='__main__':main()
