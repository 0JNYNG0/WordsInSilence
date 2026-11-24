from common import *
from sync_status import sync

NAMES={'C':'기존 C','C0':'추가 정보 모두 제외 C0','C1':'추가 정보 모두 사용 C1','minus_position':'위치·관측 상태 제외','minus_extent':'손 영역 경계 제외','minus_intersection':'현재 양손 겹침 제외','minus_proximity':'최근 근접 단서 제외'}

def main():
 d=read(U/'COMPARISON.json');iv=read(U/'SCORES_TRACE_INDEPENDENT.json')
 assert iv['status']=='PASS' and iv['comparison_sha256']==sha(U/'COMPARISON.json')
 assert not (U/'TRAIN_RUNNING.json').exists()
 interpretation=(U/'INTERPRETATION_KO.txt').read_text(encoding='utf-8')
 s=d['summary'];rows=[];effect={}
 for arm in ARMS:
  effect[arm]={}
  for view in ('sensor','masked'):
   vals=s[arm][view]['top1_correct']['counts'];ref=s['C1'][view]['top1_correct']['counts']
   delta=[a-b for a,b in zip(vals,ref)]
   effect[arm][view]={'net_correct_by_seed_vs_C1':delta,'better_seeds':sum(x>0 for x in delta),'same_seeds':sum(x==0 for x in delta),'worse_seeds':sum(x<0 for x in delta),'pp_vs_C1':sum(delta)/9,'pp_vs_C':s[arm][view]['top1_correct']['percent_mean']-s['C'][view]['top1_correct']['percent_mean'],'pp_vs_C0':s[arm][view]['top1_correct']['percent_mean']-s['C0'][view]['top1_correct']['percent_mean']}
   rows.append({'arm':arm,'view':view,**effect[arm][view]})
 write(U/'GROUP_EFFECTS.json',{'effects':effect,'interpretation':'Conditional removal effect under fixed training protocol; not universal or additive importance. DEV reused; no final blind confirmation.'},new=True)
 write(U/'NEXT_PLAN.json',{'status':'PROPOSED_NOT_EXECUTED_AWAITING_REVIEW','goal':'Audit the cause of added position-state disagreement on public TRAIN before proposing any new representation or training.','keep':'Existing C and web/DTW unchanged. No best-seed replacement.','scope':['Use saved public TRAIN observations/evidence and original frames; preselect by observation condition without classification errors or user data.','Separate real observation differences, admissible extrapolation differences, and inconsistent decisions given comparable evidence.','Specify minimal distinction between observed Hand21 information and inferred Pose/history evidence. Do not force equal labels or leak hidden raw points.','Only propose later training if state-reliability evidence supports a minimal revision; no simultaneous epoch/window/smoothing changes.'],'user_mentor_heldout_used':False,'new_training_this_plan':0,'no_promised_accuracy_gain':True},new=True)
 lines=['손 관측 정보 네 묶음 제외 비교 결과',f'완료 시각: {now()}','',interpretation,'',
 '1. 실제 수행한 비교',
 '공개 AI Hub 100단어, 학습 영상 899개(촬영자 ID 1~9), 개발 검증 300개(ID 11~13)를 사용했다. 같은 네 조건을 각 3개 초기 난수 조건으로 학습하여 총 12회를 완료했다.',
 '좌표·관측 마스크·시간의 기존 280채널과 영상 구간은 보존했다. 310채널 구조에서 지정한 추가 묶음만 양손 모두 0으로 고정했다. 나머지 세 묶음, 초기화, 시간/보기 일정, 최적화, 최대 80회/조기종료 12회 규칙은 동일하다.',
 '기존 C의 학습된 가중치를 이어서 학습한 것이 아니라 각 조건을 같은 초기 상태에서 새로 학습했다. 기존 C/C0/C1은 저장된 점수만 재사용했다.',
 '사용자·멘토 촬영과 보류 촬영자는 읽거나 사용하지 않았다. 실제 웹 C·DTW·운영 코드·원본을 변경하지 않았다.',
 '', '2. 개발 검증 정확도 — 정답이 1위인 비율, 3회 평균과 범위']
 for view,title in [('sensor','상반신 잘린 영상에서 실제 관절 검출(sensor, 주지표)'),('masked','전체 검출 후 상반신 밖 정보 가림(masked)')]:
  lines.append(title)
  for a in ('C','C0','C1')+ARMS:
   q=s[a][view];m=q['top1_correct'];lo,hi=m['percent_range']
   lines.append(f"  {NAMES[a]}: {m['percent_mean']:.2f}% (범위 {lo:.2f}~{hi:.2f}%), 300개 중 각회 {m['counts']}; Top5 {q['top5_correct']['percent_mean']:.2f}%; 평균 교차엔트로피 {q['mean_ce']:.6f}")
  for a in ARMS:
   q=effect[a][view]
   lines.append(f"  {NAMES[a]}: 전체정보 C1 대비 {q['pp_vs_C1']:+.2f}%p / 기존 C 대비 {q['pp_vs_C']:+.2f}%p / C0 대비 {q['pp_vs_C0']:+.2f}%p; C1 대비 각회 정답 변화 {q['net_correct_by_seed_vs_C1']}")
 lines.extend(['','3. 사람별·미관측 유형별 결과 (sensor)'])
 for field in ('by_signer','by_observation_pattern'):
  for k in s['C']['sensor'][field]:
   n=s['C']['sensor'][field][k]['total_per_seed']
   lines.append(f'{field} {k} / 각회 {n}영상: '+', '.join(f"{NAMES[a]} {s[a]['sensor'][field][k]['top1_percent_mean']:.2f}%" for a in ('C','C0','C1')+ARMS))
 lines.append('미관측 유형은 사전에 정한 센서 관측 분류이며 언어학적으로 한손 수어/양손 수어를 판정한 라벨은 아니다.')
 lines.extend(['','4. 새로 맞힌 것과 새로 틀린 것 (sensor, 같은 영상·seed 쌍 비교)'])
 for a in ARMS:
  for b in ('C','C0','C1'):
   pp=d['paired_changes'][f'{b}_to_{a}'];gain=sum(len(x['sensor']['gained']) for x in pp.values());loss=sum(len(x['sensor']['regressed']) for x in pp.values())
   lines.append(f'  {NAMES[b]} → {NAMES[a]}: 새로 맞힘 {gain}, 새로 틀림 {loss}, 순변화 {gain-loss:+d} (3회 반복 포함)')
 lines.extend(['','5. 결과 확인 전 고정한 진행 조건'])
 for a in ARMS:
  g=d['gates'][a];failed=[k for k,v in g['checks'].items() if not v]
  lines.append(f"  {NAMES[a]}: {'후속 검토 조건 통과' if g['pass_to_next_review'] else '조건 미충족'}")
  if failed:lines.append('    미충족: '+', '.join(failed))
 lines.extend(['자동 운영 적용은 없다. C1보다 좋아졌더라도 기존 C/C0와 비교한 개선·사람별/미관측별/Top5 보존 조건을 함께 확인했다.',
 '', '6. 검증과 한계',
 '1199개 기존 입력을 재사용했다. 학습용 네 일정과 검증용 전체 구간에서 제외한 채널만 0인지, 나머지 값·패딩·라벨·입력 원본이 보존되는지 검사했다. 세 초기화의 원래 출력 동등성과 제외 채널의 기울기 0도 확인했다.',
 '초기 검증 도구가 DEV에 학습 전용 핵심 구간을 요청해 실패했다. 최초 오류와 코드를 보존하고 DEV는 원래 전체 구간만 검사하도록 수정했다. 데이터/학습 코드/조건 변경 없이 복구했으며 오류 당시 학습은 시작 전이었다.',
 f"새 학습 12회, 총 {sum(t['epochs'] for t in d['trace'])} epoch. 실제 배치 {iv['batches']}개와 영상 노출 {iv['logged_video_exposures']}건을 원본 ID·보기·순서로 확인했다. 노출은 같은 899개 영상의 반복이며 새 표본 수가 아니다.",
 '새 12회 및 기존 9회, 두 보기의 저장 점수 12,600행을 별도 계산 코드로 재집계했다. 선택 epoch·조기종료·저장 모델 연결과 제외 채널의 최종 투영 가중치 0을 확인했다. 검산을 위한 모델 재추론은 없다.',
 '구현과 검산은 Codex가 서로 분리한 코드로 수행했다. 다른 사람이나 에이전트의 독립 검토로 표현하지 않는다.',
 'DEV300은 epoch 및 후보 비교에 반복 사용한 개발 검증 자료다. 3회 반복을 900개의 독립 영상이나 새 사람으로 세지 않는다. 실제 사용자 정확도와 최종 보류 촬영자 성능은 이번 결과가 아니다.',
 '이 비교는 나머지 세 정보가 존재할 때 한 묶음을 빼는 효과다. 묶음 간 상호작용, 초기화에 따른 학습 경로 차이가 있으므로 네 효과를 더하거나 정보의 보편적인 중요도 순위로 해석할 수 없다.',
 f"순수 학습 총 {sum(read(U/f'{a}_{seed}_RESULT.json')['seconds'] for a in ARMS for seed in (20261013,20261014,20261015))/60:.2f}분. 준비·검산 시간 별도.",
 '근거: TRAIN_CONTEXT.json, INPUTS_INDEPENDENT.json, IMPLEMENTATION_REVIEW.json, COMPARISON.json, SCORES_TRACE_INDEPENDENT.json, GROUP_EFFECTS.json.'])
 (U/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 checkpins(read(U/'TRAIN_CONTEXT.json')['pins']);checkpins(read(SRC/'PREP_CONTEXT.json')['pins'])
 body='ops/'+U.name+'. 4조건×3seed 총12회 학습·저장점수/실제배치/선택checkpoint 검산 완료. '+ '; '.join(f"{NAMES[a]} sensor{s[a]['sensor']['top1_correct']['percent_mean']:.2f}% / masked{s[a]['masked']['top1_correct']['percent_mean']:.2f}%" for a in ARMS)+'. 기존 C 보존. 모든 결과는 반복 개발 검증이며 실제 사용자/보류 정확도 아님. '+interpretation.split('\n\n')[0]
 next_body='RESULT_KO.txt/GROUP_EFFECTS.json의 비교·회귀·한계와 NEXT_PLAN.json의 공개 TRAIN 위치상태 불일치 원인 감사 제안을 사용자와 확인한다. 새 감사·추가 학습·운영 교체는 실행하지 않았다. 완료된 12회/점수 검산은 반복하지 않는다. 사용자·멘토 자료 학습/튜닝/후보 선택 금지 유지.'
 sync('2026-10-18 네 정보 묶음 제외 12회 비교·검산 완료',body,'2026-10-18 묶음별 제외 결과와 다음 보완 방향 검토',next_body,'WORD100_HAND_EVIDENCE_ABLATION_COMPLETE',{'status':'COMPLETE_VERIFIED','running':False,'new_runs':12,'summary':s,'gates':d['gates'],'report':str(U/'RESULT_KO.txt'),'comparison':str(U/'COMPARISON.json'),'user_mentor_heldout_used':False,'production_changed':False})
 write(U/'STATUS.json',{'stage':'COMPLETE_VERIFIED_AWAITING_REVIEW','at':now(),'runs':12,'production_changed':False,'report':str(U/'RESULT_KO.txt')})
 execution=read(U/'CURRENT_EXECUTION.json');execution.update(stage='COMPLETE_VERIFIED',at=now(),running=False,exit_code=0);write(U/'CURRENT_EXECUTION.json',execution)
 write(U/'FINAL_AUDIT.json',{'status':'PASS','at':now(),'source_and_model_hashes_preserved':True,'documents_synchronized':True,'report_sha256':sha(U/'RESULT_KO.txt'),'comparison_sha256':sha(U/'COMPARISON.json')},new=True)
 print('Complete: 12-run report, protected hashes and current documents.')

if __name__=='__main__':main()
