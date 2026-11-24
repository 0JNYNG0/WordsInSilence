from common import *

def main():
 d=read(U/'COMPARISON.json');v=read(U/'SCORES_TRACE_INDEPENDENT.json');assert v['status']=='PASS' and v['comparison_sha256']==sha(U/'COMPARISON.json')
 s=d['summary'];passed=d['gate']['pass_to_next_review']
 lines=['손 관측 근거 추가 TCN 비교 결과',f'완료 시각: {now()}',f'판정: {"후속 검토 조건 통과 — 자동 적용 아님" if passed else "추가 후보 미채택 — 기존 C 유지"}','',
 '1. 무엇을 바꿨는가',
 '기존 C의 좌표·관측 여부·시간 정보280개는 그대로 두고, 손 위치·영역 경계·양손 영역 교차·최근 근접 단서30개를 별도로 추가했다. 실제 손가락 좌표를 복원하거나 기존 동작 프레임을 잘라내지 않았다.',
 'C0는 같은310입력 구조에서 새30개를 모두0으로 둔 대조 모델, C1은 실제30개 단서를 넣은 후보다. 같은 원초기 상태에서 각각3회 새로 학습했다. 기존 C의 학습된 가중치를 이어 학습한 것은 아니다.',
 '', '2. 자료와 검증',
 'AI Hub100단어, TRAIN899영상(촬영자1~9), DEV300영상(촬영자11~13). 사용자·멘토 촬영 및 보류14~18은 읽거나 사용하지 않았다. 기존 웹/모델/DTW 변경 없음.',
 '1199입력 준비 완료. 기존280채널77,910,560값 완전보존, 추가손상태556,504개 독립 계산 일치. 숨긴6,192,157점 교란에도 masked추가단서 불변. 기존68영상 진단17,774상태 재현 및 원래C의 경계처리 차이165손상태 별도 기록.',
 '기존 정상대조40의 실제C관측 보존:38완전/2부분관측. 두 부분관측은 원래C에서 이미 화면밖 점을 제외했던 사례이며, 과거 진단의 완전관측 표시와 구분했다.',
 '새 TRAIN96영상 대표장면16페이지 시각검토 완료; 나머지4단어는 기존9명전원 검토 근거 재사용. 표본시각검토이며 모든 프레임의 수어 의미를 판정한 것은 아니다.',
 '구현·검산: Codex 로컬. 구현과 다른 계산 코드로 확인했으며 별도 사람/에이전트 검토를 받았다고 주장하지 않는다.',
 '', '3. 개발 검증 정확도 (각 조건300영상,3회 평균)']
 for view,title in [('sensor','상반신 영상에서 실제 검출한 입력(sensor,주지표)'),('masked','전체 영상 검출 후 상반신만 남긴 입력(masked)')]:
  lines.append(title)
  for arm in ('C','C0','C1'):
   q=s[arm][view];m=q['top1_correct'];rr=m['percent_range']
   lines.append(f"  {arm}: Top1 {m['percent_mean']:.2f}% / 범위{rr[0]:.2f}~{rr[1]:.2f}% / 각회정답{m['counts']} / Top5 {q['top5_correct']['percent_mean']:.2f}%")
  for old in ('C','C0'):
   pp=s['C1'][view]['top1_correct']['percent_mean']-s[old][view]['top1_correct']['percent_mean']
   lines.append(f'  C1 - {old}: {pp:+.2f}퍼센트포인트')
 lines.extend(['','sensor 고정 미관측 그룹 (언어학적 한손수어 라벨 아님)'])
 for g in s['C']['sensor']['by_observation_pattern']:
  vals=[s[a]['sensor']['by_observation_pattern'][g] for a in ('C','C0','C1')]
  lines.append(f"  {g}, 각회{vals[0]['total_per_seed']}영상: C {vals[0]['top1_percent_mean']:.2f}% / C0 {vals[1]['top1_percent_mean']:.2f}% / C1 {vals[2]['top1_percent_mean']:.2f}%")
 lines.extend(['','4. 개선과 새 오답'])
 for b in ('C','C0'):
  for seed,p in d['paired_changes'][f'{b}_to_C1'].items():
   q=p['sensor'];lines.append(f"  seed{seed}, {b}→C1: 새로맞힘{len(q['gained'])}, 새로틀림{len(q['regressed'])}, 순변화{q['net_correct']:+d}/300")
   for k,title in [('gained','새로맞힘'),('regressed','새로틀림')]:
    lines.append('    '+title+': '+(', '.join(f"{r['label']}({r['clip_id']})" for r in q[k]) or '없음'))
 fixed='20261013';lines.append('사전고정 웹시험용seed20261013도 전체3회와 별도로 확인(이번에는 웹적용 안함).')
 for a in ('C','C0','C1'):lines.append(f"  {a}: sensor {s[a]['sensor']['top1_correct']['counts'][0]}/300 = {s[a]['sensor']['top1_correct']['counts'][0]/3:.2f}%")
 lines.extend(['','5. 판정과 남은 문제'])
 failed=[k for k,z in d['gate']['checks'].items() if not z]
 lines.append('사전고정 조건 불통과: '+(', '.join(failed) or '없음'))
 lines.extend([
 '진단 정보가 더 자세해진 것과 TCN 인식률이 좋아지는 것은 별개다. 이번 비교에서 관측되지 않은 손가락 자체가 생긴 것은 아니다. 화면안 미검출, 손/팔/얼굴 가림, 실제 손윤곽과 Pose4점 추정 차이는 여전히 남는다.',
 '새 단서가 유용한 정보뿐 아니라 센서의 흔들림이나 불확실한 추정도 전달할 수 있다는 것이 가능한 설명이다. 이번 결과만으로 특정 단서가 원인이라고 확정하거나 ST-GCN/TCN 전체 방법의 실패로 일반화하지 않는다.',
 'C0는 기존C보다 입력 투영 크기가 커서 실행경로/수치계산 차이가 있다. 공통 초기 출력은3seed 모두 차이0이었지만 학습 후 완전동일을 전제하지 않았다. C1 대 C0와 C1 대 저장C를 모두 보고한다.',
 'DEV300은 모델 선택에도 반복 사용한 개발 검증 자료다. 3seed와 두입력보기를 독립된1800명/영상으로 세지 않는다. 실제 웹캠 사용자 정확도·새 사람 일반화·손모양 정확도는 이번에 검증하지 않았다.',
 '조건과 계수를 결과에 맞추어 바꾸거나 추가 재학습하지 않았다. 다음 단일 단계는 이 결과를 사용자와 검토하는 것이다. 운영 C는 보존한다.',
 '', '6. 검산 및 실행 기록',
 f"새 학습6회, 총{sum(t['epochs'] for t in d['trace'])}epoch / 실제 배치{v['batches']}개 / 학습영상 노출로그{v['logged_video_exposures']}건(같은899영상 반복, 신규영상 수 아님).",
 '선택epoch·중단규칙·배치별원본ID/보기·저장checkpoint 연결 확인. 선택된 저장logits5400행을 독립 재집계(기존C3회+새6회,각2보기). 모델 재추론0. 다른epoch의 CE는 저장이력의 선택규칙을 검산했으며 전체epoch logits를 재계산한 것은 아니다.',
 f"순수6회 학습 총{sum(read(U/f'{a}_{seed}_RESULT.json')['seconds'] for a in ('C0','C1') for seed in (20261013,20261014,20261015))/60:.2f}분. 입력 준비/검산/시각검토/코드작성 시간 별도.",
 '근거: READY.json, INPUTS_INDEPENDENT.json, VISUAL_REVIEW.json, TRAIN_CONTEXT.json, COMPARISON.json, SCORES_TRACE_INDEPENDENT.json. 최초 오류는 FIRST_*와 RECOVERY 기록/attempt01에 보존.'
 ])
 (U/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 write(U/'STATUS.json',{'at':now(),'stage':'COMPLETE_VERIFIED','decision':'PASS_NEXT_REVIEW' if passed else 'NOT_ADOPTED_KEEP_C','comparison':str(U/'COMPARISON.json'),'report':str(U/'RESULT_KO.txt'),'training_runs':6,'process_running':False,'production_changed':False})
 print('\n'.join(lines[:38]))
if __name__=='__main__':main()
