from datetime import datetime
from common import *

def main():
 assert not (U/'COMPLETION.json').exists()
 v=read(U/'review/VERIFY_DEMO.json');g=read(U/'review/VERIFY_GAP_CONTEXT.json');ui=read(U/'UI_VERIFICATION.json')
 assert v['status']=='PASS' and g['status']=='PASS' and ui['status']=='PASS'
 assert sha(U/'index.html')==ui['html_sha256'] and sha(U/'app.js')==ui['js_sha256']
 pins();root=O.parents[2];at=datetime.now().astimezone().isoformat();server=read(U/'SERVER_STATUS.json')
 current=('### 2026-11-01 공개 4표현 비교 데모 완료 — 짧은 관측 누락의 제한 평가와 화면 통합\n'
 '`word100_public_feedback_demo_20261101` 완료. 기존 96영상·288시점의 저장 예측을 관절 지원별로 분리 평가했다. 같은 표현의 다른 기록ID4 개발 자료에서 부족한 11항목은 3영상/3시점이며, 세 seed 정답 10/11·9/11·10/11(평균87.88%, 범위81.82~90.91%). 해당 native 손 관측은 실제 1~2프레임 0점이었다. 작은 표본·특징 항목 성적이며 단어/교육 정확도나 RGB 단독 효과가 아니다. 추가학습0, 기존 지도학습 모델 재사용. 원3항목6252값/48센터 보호, 추가4항목의 새1844센터만1회추론, 경계192센터는 채우지 않았다. 순종·잘하다·조상·이집트 16영상/12쌍/39구간(출처설명 연결12)의 영상·관절·추정·DTW·설명을 localhost:8801에 통합. 분석기록 재생 데모이며 새촬영 분석/교육교정 미완료. 별도Codex 저장검산200310+보충134 PASS, 브라우저12쌍/39구간 재생·가독성 PASS. 컴퓨터제어 kernel 실패 후 일반환경 별도headless Chrome으로 검증. C/T3F·운영·개인자료 보존.\n\n')
 nxt=('### 2026-11-01 다음 단일 작업 — 데모의 차이 후보를 중요한 동작 근거와 연결\n'
 '공개 비교 데모는 http://127.0.0.1:8801/ 에서 기존 분석기록을 재생한다. 원래 목표는 손모양·관계·동작 순서의 중요한 차이와 정상 변이를 구분하는 피드백이다. 다음은 현재4표현의 출처 설명/확인된 특징을 실제 차이 사례 및 같은 단계 대응 근거와 연결해 안내 유용성을 평가한다. 39구간 중 출처 연결12와 관찰설명27을 혼동하지 않고, 반복회차 미확정은 정답으로 채우지 않는다. 이번11항목/3시점 성적을 문턱조정·seed선택·보류승격에 쓰지 않는다. 완료집계·1844추론 재실행X. 새 FIT 정답이나 구체적인 입력/목표 오류가 확인될 때만 필요한 판정부 학습; 기존DEV 오답을 FIT로 이동X. 사용자자료/운영 교체/새 다운로드 자동 확대X. 영상전체 대기시간은 캐시 재생시간과 분리해 향후 측정한다.\n\n')
 cp=root/'WordsInSilence_PROJECT_CONTEXT.md';text=cp.read_text(encoding='utf-8-sig')
 for heading,addition in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',current),('## 9. 다음 단일 작업',nxt)]:
  assert text.count(heading)==1;text=text.replace(heading,heading+'\n\n'+addition,1)
 cp.write_text(text,encoding='utf8')
 entry=dict(status='COMPLETE_BOUNDED_RESEARCH_REPLAY_NOT_EDUCATIONAL_TUTOR',at=at,unit=str(U),
  active_training_or_inference=None,steps_executed=['saved gap-stratum evaluation','conditional training decision: reuse supervised model','new-time predictions and comparison integration','independent data and browser verification'],
  public_audit_clips=96,label_centers=288,known_cells_per_seed=1441,unknown_cells_per_seed=575,
  development_gap_known_cells=11,development_gap_correct=[10,9,10],development_gap_mean_accuracy=29/33,
  development_gap_clips=3,development_gap_recorded_people=[4],development_gap_native_runs=[2,1,2],
  new_training=0,new_native_center_predictions=1844,reused_centers=48,protected_original_values=6252,
  new_rgb_extraction=0,new_detector_calls=0,new_dtw_paths=0,
  demo_expressions=['순종','잘하다','조상','이집트'],demo_clips=16,demo_pairs=12,demo_stage_cards=39,source_linked_stage_cards=12,
  educational_accuracy=None,recognition_gain=None,raw_video_latency_seconds=None,
  verification=dict(separate_codex_saved_data_checks=200310,separate_gap_checks=134,browser_pairs=12,browser_stage_seeks=39),
  replay_server=server,report=str(U/'RESULT_KO.txt'),url='http://127.0.0.1:8801/',production_changed=False)
 state=read(O/'STATE.json');state['current_stage']='PUBLIC_FEEDBACK_DEMO_COMPLETE_RESEARCH_REPLAY'
 state['next_single_action']='4표현 데모의 차이 후보를 출처 특징·같은 동작 단계·실제 대비 근거와 연결해 안내 유용성 평가. 완료추론/집계 재실행·DEV튜닝·운영교체 금지.'
 state[U.name]=entry;(O/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 (root/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+current+nxt+
  '실행 상태: 학습/추론/검산 종료. 일반 실행환경의 읽기 전용 데모 서버 PID '+str(server['pid'])+' / 8801 유지. 카메라·업로드·추론 endpoint 없음. 새 대화는 /health와 PID를 먼저 확인하고 중복 시작하지 않는다.\n'
  '원source/완료 실험은 읽기전용. 기존 C·T3F·운영 DTW/웹 유지, 사용자·멘토 자료 접근·학습·튜닝·선택0.\n'
  '구현/통합/UI: root. 별도 데이터/보충검산: semantic_feedback. 동일 Codex이며 독립 사람 KSL 정답 아님. 신규 forward 자체 재현은 하지 않았고 저장 lineage/범위/출력 연결 검산이다.\n'
  '최초실패보존: UI_FIRST_FAILURE_*.json(브라우저/crashpad·제한localhost), GAP_CONTEXT_FIRST_FAILURE.txt(과거80개 색인 lookup). 후속검증 PASS.\n'
  '기록 위치: Docs/PROJECT_DIRECTION_HISTORY_KO.txt, RESULT_KO.txt, FROZEN.json, AUDIT.json, DENSE_MANIFEST.json, DEVELOPMENT_GAP_CONTEXT.json, review/VERIFY_*.json, UI_VERIFICATION.json.\n',encoding='utf8')
 hist=root/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
 with hist.open('a',encoding='utf8') as f:
  f.write('\n\n2026-11-01 — 공개 영상 소규모 비교 데모 연결 완료\n'
  '문제: 관절이 조금만 빠져도 모든 분석을 멈추는 것처럼 보였고, 연구 부품들이 흩어져 실제 피드백 목표에 얼마나 가까운지 알기 어려웠다.\n'
  '방향: 완벽한 좌표 복원을 먼저 요구하지 않고, 관측 가능한 중요한 특징과 모델의 추정을 구분하면서 몇 개 표현을 끝까지 비교해 보는 데모로 연결했다. 원래의 수어 방법·순서 이해와 중요한 차이 피드백 목표는 유지했다.\n'
  '검증: 기존 공개96영상288중앙시점의 저장 예측을 관절 지원별로 나눴다. 같은 표현/다른 수행자 개발 자료의 관절 부족11항목에서 세seed는10/11,9/11,10/11을 맞혔다. 이3시점은 해당 손의 native 관측점이 실제1~2프레임 없었던 경우였다. 다만3영상·한 기록ID에 모인 표본이며 교육 정확도가 아니다. 미학습 표현의 부족26항목에서는19/26,19/26,18/26으로 더 낮았다.\n'
  '학습 결정: 기존 판정부는 이미 지도학습돼 있었다. 이번에는 새 학습 정답이나 입력 오류가 확인되지 않았으므로 반복 재학습하지 않고 고정 모델을 사용했다. 학습을 추가하지 않은 사실을 숨기지 않았다.\n'
  '구현: 순종·잘하다·조상·이집트의 공개16영상/12쌍에서 두 영상·관절·7특징 추정·저장 DTW 대응·동작 설명을 나란히 볼 수 있게 했다. 기존3출력6252값과48센터를 보호하고 추가4특징의 새1844센터만 계산했다. 실제13프레임 문맥이 없는 경계는 채우지 않았다.\n'
  '이전 대비: 보류된 관절 근거와 남아 있는 이미지·시간 모델 추정을 한 화면에서 볼 수 있게 됐다. 출처 설명 연결12구간과 물리적 관찰설명만 있는27구간을 구분했다. 전체 수어 정오/중요오류 교정 모델이 완성됐거나 인식률이 올랐다고 주장하지 않았다.\n'
  '검증/한계: 별도 Codex 저장 검산200310+보충134 PASS, 브라우저12쌍39구간의 동영상 재생·버튼·가독성 확인. 기록 재생 데모이므로 반응속도를 원본 영상 분석 대기시간으로 말할 수 없다. 실제 학습자의 교육적 오류 정답과 반복회차 대응은 여전히 제한적이다.\n'
  '다음: 새 검출기를 늘리기보다, 현재 데모에서 명확히 확인할 중요한 차이와 같은 단계 대응을 함께 평가한다. 필요 근거가 갖춰진 부분에 한해서 학습을 추가한다. 사용자 촬영은 학습·튜닝·모델 선택에서 계속 제외한다.\n'
  '근거: '+str(U/'RESULT_KO.txt')+'\n')
 (U/'CURRENT_EXECUTION.json').write_text(json.dumps(dict(status='RESEARCH_COMPLETE_REPLAY_SERVER_RUNNING',at=at,
   inference_running=False,training_running=False,server=server),ensure_ascii=False,indent=2),encoding='utf8')
 artifacts=[p for p in U.iterdir() if p.is_file() and p.name not in ['COMPLETION.json','SERVER_STATUS.json','server_stdout.txt','server_stderr.txt']]
 artifacts += list((U/'predictions').glob('*.npz'))+list((U/'review').glob('*.json'))+list((U/'review').glob('*.py'))
 docs=[cp,root/'NEW_CHAT_HANDOFF.md',O/'STATE.json',hist]
 save('COMPLETION.json',dict(**entry,artifact_pins={str(p):sha(p) for p in artifacts},document_pins={str(p):sha(p) for p in docs},
  completion_claim='Bounded saved-input research replay integrated and verified. No new training, raw-video online feedback or validated educational correction.'))
 assert read(O/'STATE.json')[U.name]['status']==entry['status']
 for p in docs:assert p.exists()
 print('COMPLETE: public demo, evidence/limitations, independent checks and project history synchronized.')
if __name__=='__main__':main()
