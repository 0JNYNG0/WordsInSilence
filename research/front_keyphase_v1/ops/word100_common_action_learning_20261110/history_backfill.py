"""Backfill from archived local reports, with a searchable evidence index."""
from common_action import *
def main():
 target=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt';old=target.read_text('utf-8');assert '2026-11-10 과거 기록 대조·보완' not in old
 index=read(U/'HISTORY_REPORT_INDEX.json');rows=index['rows']
 for r in rows:assert sha(r['report'])==r['sha256']
 entries=[
 ('2026-10-25 — RGB·관절 보완 후보를 실제로 학습했으나 보간보다 좋아지지 않음',
  '단순 재검출만으로 부족한 누락을 영상과 앞뒤 관절로 보완하려 했다. 공개61영상에서 두 후보를 학습·비교했지만 대리좌표 오차는 선형 보간보다 커졌고 단어 인식 추가 이득도 없었다. 연구 후보로 보존했으며 기본 모델 교체는 하지 않았다. 추정 관절과 실제 관측을 구분하는 원칙이 여기서 중요해졌다.',
  ['word100_temporal_refiner_autonomous_20261025','word100_recovery_image_benchmark_20261025']),
 ('2026-10-25 — YOLO 손 모델은 실제 네 조건 시험 후 제한적 보조로 판단',
  '기존 관측·MediaPipe 재검출·YOLO 단독·결합을 같은 공개 영상으로 비교했다. YOLO 단독은 이미 맞던 표본을 잃었고 결합도 재검출보다 충분한 이득을 보이지 않았다. 이후 누락 순간만 호출해 손목 근거를 조금 추가했으나 손가락 정확도까지 입증한 것은 아니었다. 운영 교체나 YOLO 추가 학습은 하지 않았다. 현재 RTMW 앞의 사람 영역 YOLOX와 이 손 전용 YOLO 시험은 서로 다르다.',
  ['word100_yolo_hand_fourway_20261025','word100_yolo_selective_wrist_20261025']),
 ('2026-10-25~26 — MANO·HaMeR에서 손 복원과 같은 손 중복의 문제를 분리',
  'MANO는 영상 인식기 자체가 아닌 손 형상 모델이고, HaMeR 영상 추정 후보를 공개 자료에서 비교했다. 기존 손과 같은 손을 중복 추가한7프레임을 막았지만 숨은 손을 복구한 결과는 아니었다. 팔 연결·앞뒤 영역 추적을 추가해 위치 근거와 손가락 모양 근거를 나눴다. 좌우 충돌10프레임은 미해결이었고 새 좌표 강제 채택·웹 교체는 하지 않았다.',
  ['word100_mano_threeway_20261025','word100_mano_duplicate_guard_20261025','word100_mano_identity_recheck_20261026','word100_mano_position_shape_split_20261026']),
 ('2026-10-26 — AI Hub 좌표를 오답노트로 쓰기 전 품질과 영상 대응을 확인',
  '제공 좌표는 OpenPose 계열 처리·보정 자료였지만 영상 프레임과의 대응, 누락, 위치 불일치가 있었다. 원본 RGB에서 보이는 위치 범위와 비교했으며 일정 이동 보정이 일부 도움을 줘도 정상 사례를 악화시키기도 했다. 그래서 모든 제공 좌표를 손가락 학습 정답으로 승인하지 않았다. 관측 도구끼리 일치하는 것을 독립 정답으로 여기지 않는 방향을 유지했다.',
  ['word100_aihub_coordinate_quality_20261026','word100_aihub_coordinate_alignment_20261026','word100_visible_coordinate_reference_20261026']),
 ('2026-10-26 — TCN 인식 개선 수치와 평가 자료 중복 정리',
  '기존 T3-F 세 모델의 확률을 동일 비중으로 평균한 E3를 고정 비교했다. 추가 공개199영상에서 기준 T3F13은196/199, E3는197/199였다. 199는 기록ID14·15의 영상 수이며100단어 전부의 모든 사람 수나 피드백 정확도가 아니다. REAL16 일부가 과거 DEV11과 파일 해시까지 같아 점수 확인 전에 ID16전체를 격리했다. 새 학습·원본 C/T3-F 교체 없이 비교한 결과다.',
  ['word100_autonomous_recognition_20261026']),
 ('2026-11-02 후반 — 기록이 준비 단계에 멈춰 있던 이후의 학습·실패도 연결',
  '기존 이력 마지막 문장은 행동 구간 정답을 준비하는 중이었다. 이후에는 실제 과정 주석, 단계/순서/현재 모습 분리, 과정 판정부와 관측 보존을 여러 고정 조건으로 학습·비교했다. 일부 지표 개선과 정상 또는 역순 회귀가 함께 있었고 미채택 후보도 남겼다. 아래 근거 색인에는 중간의 개별 완료 보고서를 빠짐없이 연결했다. 각각 다른 과제·분모이므로 하나의 성능 상승 곡선으로 합치지 않는다.',
  ['word100_action_episode_supervision_20261102','word100_learned_action_process_20261102']),
 ('2026-11-03 — 학습된 시간별 움직임을 실제 전체 분석에 연결',
  '새 공개 영상에서 관절·RGB 관측→작은 시간별 판정부→학습된 흐름 DTW를 연결했다. 여러 참고 영상을 새 촬영마다9회 분석하는 방식은 아니었다. 움직임이 보이는 것과 올바른 순서로 완수했다는 것을 분리했다. 실제 잘못된 손모양 반례의 부족도 확인했다. 전체 분석12.82초 중 입력 준비11.34초였고 RGB는 병렬 실행되므로 단순 합산하지 않는다. 기존 웹 기본을 전부 교체한 결과는 아니다.',
  ['word100_local_event_learning_20261103','word100_local_event_runtime_20261103','word100_capture_parallel_20261103','word100_feedback_discrimination_20261103']),
 ('2026-11-04 — 긴 글 중심 결과에서 영상·색상 중심 화면으로 전환',
  '사용자가 직관적인 시각 피드백을 원한다고 재확인했다. 저장된 공개 원본과 순서변경·삭제·가림 시험을 나란히 보여주는8811 화면을 만들었다. 주황은 차이 후보, 회색은 확인 어려움이며 실제 사용자 오류 정답률이 아니다. 이어8812에는3초 카운트다운, 웹캠 촬영, 손내림 자동 종료, TCN 가로3후보, DTW 동시 재생과 구간 색상을 연결했다. 화면 구현 완료와 교정 AI 완성은 구분했다.',
  ['word100_public_visual_feedback_20261104','word100_integrated_test_20261104']),
 ('2026-11-08 — 멘토 제안 검토 후 RTMW와 촬영 중 관절 재사용으로 전환',
  'MediaPipe와 RTMW-l을 같은 공개 자료에서 비교해 손 전체 누락의 차이를 확인했다. RTMW도 엄지 오차·겹침·가짜 손이 남아 좌표가 모두 정답이라고 간주하지 않았다. 촬영하면서 RTMW를 실행하고 종료 뒤 저장 관절을 재사용하는 경로를 구현했다. SignRep RGB를 계속 실행하는 대신 관절 전용 시간 합성곱+GRU를 별도 학습하는 방향을 택했다. 이는 SignRep 백본 가중치가 관절 모델로 자동 이전된다는 뜻이 아니다.',
  ['word100_rtmw_mediapipe_compare_20261108','word100_rtmw_stream_reuse_20261108','word100_rtmw_browser_capture_20261108']),
 ('2026-11-08 — TCN100단어와 세부 피드백4표현의 학습 범위를 구분',
  'RTMW 공개1199영상(학습899/개발300), 원본150,476프레임을 준비했다. TCN은 C 복사 후 미세조정과 같은 구조의 새 학습을 비교했다. 피드백은 그중4표현의36학습/12개발 영상,7특징과13단계다. 이후8814 연구 화면으로 연결했다. 화면 밖 반대 손의 잘못된 좌표를 공통 입력 검사로 보류해 공개 상반신22영상의 TCN1위14→22를 확인했다. 이22는 새 단어 수나 피드백 정오 정확도가 아니다. C/T3-F 원본은 보존했다.',
  ['word100_rtmw_joint_learning_20261108','word100_action_feedback_demo_20261108','word100_webcam_recording_fix_20261108','word100_observation_guard_20261108']),
 ('2026-11-09 — 손모양 긍정 근거·이동 방향 확인은 웹 적용, 확장 학습은 별도',
  '차이 점수가 낮다는 이유만으로 맞았다고 하지 않고 손모양 일치 근거를 요구했다. 팔꿈치·손목 기준 이동 방향도 연결해 정방향과 역방향을 구분했다. 이 공통 보완은8814에 적용했다. 반면 추가 시간 합성곱+GRU 미세조정, 공통 움직임 판정부,100단어 보조 학습은 실제로 실행했지만 단계·자동 구간 회귀가 있어 새 가중치를 웹 기본으로 교체하지 않았다.100단어 영상 학습과100단어 모든 세부 정답 학습은 다르다.',
  ['word100_positive_direction_20261109','word100_gru_semantic_finetune_20261109','word100_motion_components_20261109','word100_shared_motion_learning_20261109']),
 ('2026-11-09~10 — 모양 오류가 비교에서 빠지거나 미완료를 완료로 보는 문제',
  '모양을 포함한 특징만으로 구간을 고르면 잘못된 손모양 구간 자체를 놓칠 수 있어, 몸·팔·손목 움직임으로 찾는 공통 후보 경로를 추가 학습했다. 단계 전체 교체는 정상 보류를 늘려 채택하지 않았다. 이후 기존 창을 보존하고 분명한 다른 사건 근거만으로 완료를 보류하는 분리 검사로 좁혔다. 역사적 공개 삭제6/관절가림6조건의 거짓 전체비슷함은 각각2→0이었다. 새 기록ID10의20조건에서는 추가 이득이 없었다. 이것은 실제 사용자 오류 전체를 해결했다는 결과가 아니다. 완료 검사는 격리 경로이며 웹 기본 미교체다.',
  ['word100_attempt_event_supervision_20261109','word100_motion_event_candidates_20261109','word100_event_completion_20261109','word100_completion_fullpath_20261110']),
 ]
 block=['\n\n2026-11-10 과거 기록 대조·보완 (이번에 확인한 완료 보고서 기준)','이 절은 과거 작업 날짜에 새 일을 했다는 뜻이 아니라, 빠졌던 이력을 현재 시점에 보완한 기록이다. 원래 실패·미채택·보류 결과는 고치지 않았다.']
 evidence=[]
 for title,body,units in entries:
  block+=['\n'+title,body]
  for unit in units:
   p=O/unit/'RESULT_KO.txt';assert p.exists();content=p.read_text('utf-8-sig');evidence.append(dict(unit=unit,path=str(p),sha256=sha(p)))
   block.append('근거: '+str(p))
 # Do not let an earlier snapshot masquerade as current runtime.
 old=old.replace('현재 기능의 역할과 적용 상태\n','2026-11-01 당시 기능의 역할과 적용 상태 (과거 스냅샷)\n',1)
 old=old.replace('웹 기본은 현재 코드 기준 M0다.','이 절을 처음 작성한 2026-11-01의 웹 기본은 당시 코드 기준 M0다.',1)
 current='''\n2026-11-10 현재 상태를 읽는 기준\n최신 연구 웹 계보는8814: RTMW 관절 재사용 → TCN 단어 후보와 관절 전용 시간 합성곱+GRU 피드백 → 학습 기준 흐름 DTW 표시이다. SignRep RGB·YOLO 손 전용·HaMeR가 현재 기본 분석에서 함께 돌아간다고 설명하면 안 된다. 웹 적용된 공통 관측 보호·모양/방향 확인과, 아직 격리된 새 학습·완료 판정 후보를 구분한다. 최신 작업 상태는 프로젝트 맥락 §8·§9와 STATE를 우선한다. 서버가 현재 실행 중이라는 뜻은 아니다.\n발표 흐름: 단어 인식 → 좌표·시간 비교 → 가림 보완 모델 시험 → 설명과 동작 연결 → 학습된 시간별 피드백 → 느린 RGB 경로 개선 → RTMW 촬영 재사용·관절 신경망 → 모양·관계·순서·완료를 나눠 검증. 목적은 유지했고 이를 이루는 방법과 검증 범위를 조정해 왔다.\n'''
 old=old.replace('작성: 2026-11-01 / 용도:', '작성: 2026-11-01 / 기록 대조 보완: 2026-11-10 / 용도:',1)
 insert=old.index('\n\n')+2;old=old[:insert]+current+'\n'+old[insert:]
 target.write_text(old+'\n'.join(block)+'\n','utf-8')
 ledger=['WordsInSilence 과거 연구 완료보고서 근거 색인 — 2026-11-10','범위: 현재 웹 연구 ops 바로 아래의 RESULT_KO.txt '+str(len(rows))+'개. Unity의 모든 과거 산출물이나 비공개 기록까지 전수 감사했다는 뜻이 아니다.','이력에 작업 폴더명이 직접 없던 '+str(sum(not r['explicitly_named_in_history'] for r in rows))+'개는 인용 누락 후보다. 내용 자체가 모두 빠졌다는 뜻은 아니다.','각 보고서는 당시 결과다. 다음 계획·보류·실제 적용 여부를 제목만으로 판정하지 말고 원문을 확인한다. 현재 서버 실행 여부나 최신 상태를 대신하지 않는다.','']
 for r in sorted(rows,key=lambda r:(r['unit'][-8:],r['unit'])):ledger += [r['unit']+' | '+r['title'],r['report'],'SHA256 '+r['sha256'],'']
 path=ROOT/'Docs/RESEARCH_REPORT_INDEX_KO.txt';path.write_text('\n'.join(ledger),'utf-8')
 write(U/'HISTORY_BACKFILL_VERIFICATION.json',dict(reports_indexed=len(rows),explicit_citation_gap_candidates=sum(not r['explicitly_named_in_history'] for r in rows),milestone_sections_added=len(entries),narrative_evidence=evidence,history=str(target),history_sha256=sha(target),index=str(path),index_sha256=sha(path),old_snapshot_relabelled=True,original_reports_modified=False,scope='Archived evidence only; pending plans are not rewritten as accomplishments. Main history previously ended mid 2026-11-02; major missing Oct25-26 and Nov02-10 milestones backfilled.'))
 print('History backfill:',len(entries),'milestones;',len(rows),'report references')
if __name__=='__main__':main()
