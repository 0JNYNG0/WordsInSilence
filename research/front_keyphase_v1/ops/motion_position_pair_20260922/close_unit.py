"""Record completed comparison and one bounded next task; no new model work."""
from pair_io import *
from datetime import datetime,timezone,timedelta

def main():
    p=policy();audit=read(H/'RESULT_AUDIT.json');e=read(R/'evaluation.json');t=read(R/'training.json')
    assert audit['status']=='PASS' and e['decision']=='REJECTED_BY_PREDECLARED_GATES'
    now=datetime.now(timezone(timedelta(hours=9)));stamp=now.isoformat();human=now.strftime('%Y-%m-%d %H:%M KST')
    next_plan=dict(status='NEXT_TASK_FIXED_NOT_EXECUTED',candidate=p['historical_reference'],
      purpose='Frozen post-public-selection compatibility and regression check on existing web-domain evaluation inputs; not further model selection or a fresh generalization test.',
      selection_basis='Existing head-only public result691/699;whole74/74;publicMP7 per window;public reference/missing/apology all retained. Partial encoder and new motion paths not adopted.',
      reuse=dict(user_plan=str(L/'ops/frozen_public_candidate_validation_20260921/plan_LOCAL_ONLY.json'),
                user_existing_results=str(L/'ops/output_retention_candidate_20260922/frozen_user_replay'),
                public_two_results=str(L/'ops/ksl77_confirmed_pair_20260922')),
      procedure=['Pin candidate and existing data/feature evidence before any new score. Verify unchanged encoder before cache reuse.',
                 'Reuse existing features and old b8/d1 predictions; compute only new head scores for same36 user records and2 previously confirmed public expressions.',
                 'Report full93/project18 separately; retain all regressions and missing evidence. Independently recalculate scores/ranks without new encoder extraction.'],
      prohibitions=['no training or threshold/seed/step search using these scores','no candidate reselection using these scores','no automatic default promotion','no additional34 KSL clips or reserved data','no new capture,UI,download,inquiry,backup,delete,Unitybuild'],
      limitation='One development-exposed user at two existing sessions and two public expressions with unresolved signer identity; not multiple independent people or final heldout evaluation.',
      new_task_already_executed=False)
    write(H/'NEXT_SINGLE_TASK.json',next_plan)
    checks=read(H/'IMPLEMENTATION_AUDIT.json')
    record=dict(at_kst=stamp,status='COMPLETE_MOTION_COMPARISON_REJECTED_NO_DEFAULT_PROMOTION',result=str(H/'RESULT.md'),
       closeout=str(H/'CLOSEOUT.json'),next_plan=str(H/'NEXT_SINGLE_TASK.json'),
       native_correct={a:e['summary']['annotation_dev'][a]['intact']['full93']['correct'] for a in ['historical_reference','control','treatment']},
       native_n=699,native_macro_recall={a:e['summary']['annotation_dev'][a]['intact']['full93']['macro_recall'] for a in ['historical_reference','control','treatment']},
       gains_vs_position=2,regressions_vs_position=5,failed_gates=audit['failed_gates'],
       implementation_checks=checks['count'],result_checks=audit['count'],new_training_runs=2,updates_each=660,
       user_arrays_read=0,reserved_arrays_read=0,default_promoted=False,
       project_complete=False)
    state_path=L/'ops/STATE.json';s=read(state_path)
    s['updated_at']=stamp;s['updated_date_kst']=now.strftime('%Y-%m-%d')
    s['motion_position_pair_20260922']=record
    s['current_stage']='MATCHED_POSITION_MOTION_COMPARISON_COMPLETE_MOTION_REJECTED'
    s['next_single_action']='Freeze existing head-only83212868 candidate and reuse cached features for one post-selection web-domain regression evaluation; no training,automatic promotion,new camera session or source search.'
    s['autonomous_workers_current_unit']=dict(unit=H.name,claude_exit_code=0,implementation_verification_exit_code=0,training_exit_code=0,evaluation_exit_code=0,independent_audit_exit_code=0,all_current_unit_workers_finished=True,work_paused_by_user=False)
    s['autonomous_goal_status'].update(status='active',at_kst=stamp,tool_status_observed='active',project_complete=False,user_requested_pause=False,
         progress_this_turn='Matched position/motion comparison actually trained,evaluated,and independently verified;motion candidate rejected;next fixed web-domain replay identified.',fresh_blocked_audit_count=0)
    s['goal_status']='INCOMPLETE: TCN recognition research continues; current new motion candidate rejected. Multi-person18-target web generalization,valid execution-difference assessment,and integration of the validated path into preserved Unity remain incomplete.'
    state_path.write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    root=L.parents[1]/'WordsInSilence_PROJECT_CONTEXT.md';text=root.read_text(encoding='utf-8-sig')
    lines=text.splitlines();lines[2]='최종 갱신: '+human;text='\n'.join(lines)+'\n'
    current=f'''### {human} 최우선 — 위치/시간변화 동일 용량 비교 실행 완료, 시간변화 후보 미채택

- 실제 두 후보를 같은 공개2122+문맥224/표집/660회로 학습했다. 추가17856가중치+분류층6045만 학습, 기존 특징 블록/통계 고정. 위치 대조690/699·시간변화687/699, 기존 분류층 후보691/699다. 평균재현율98.474%/98.337%/98.611%. 시간변화는 위치 대비2개선/5회귀, 사전11기준 중3실패로 미채택; 기본모델 변경0.
- 전체74·MP각7·시범18/누락5/사과학습9와개발3은 양군 유지. 시간변화 목표33/33도 실제 정답6종뿐이며 전체18단어/새사람 웹 검증으로 확대하지 않는다. Claude 실제종료0·새구현28검사·학습/평가/독립58검산종료0, 두 모델 동결. 약58.47초 학습, 재시도0.
- 사용자/보류 열람·새추출/다운로드/문의·카메라/화면/Unity/삭제/백업/예약0. 앞선 제한적 특징학습과 이번 입력추가 모두 이점 미입증이므로 추가 구조탐색을 바로 이어가지 않는다. 다음은 공개 결과로 고정한 기존분류층 후보83212868의 기존 웹 평가 캐시 대입이다. `research/front_keyphase_v1/ops/motion_position_pair_20260922/RESULT.md`.

'''
    next_text=f'''### {human} 다음 단일 작업 — 기존 분류층 후보의 고정 후 웹 입력 회귀 확인

- `ops/motion_position_pair_20260922/NEXT_SINGLE_TASK.json`을 따른다. 앞선 분류층 후보83212868을 기존 공개 결과만으로 고정하고, 동일 encoder가 입증되면 기존 사용자36기록/의미확인 공개2개의 특징과 b8/d1 저장 결과를 재사용해 새head 점수만 계산한다. 아직 이 후속 평가 미실행.
- 기존 학습 미사용 사용자라도 개발 노출1명이며 공개2개 촬영자미상이다. 전체93/목표18 개선·회귀를 분리하고 최종 보류시험/새사람 일반화로 부르지 않는다. 사용자자료 학습·결과기반튜닝/후보재선택·자동기본승격·새촬영/추출/화면·자료문의/다운로드·보류개봉 없음.
- 이번두후보와 이전검증은완료·재실행하지 않는다. 추가 자료의 사람/의미 조건은 별도이며 전체 프로젝트의 진행 차단으로 확대하지 않는다. 기존역할/원본보존/전체목표 미완료 유지.

'''
    text=text.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+current,1)
    text=text.replace('## 9. 다음 단일 작업\n\n','## 9. 다음 단일 작업\n\n'+next_text,1)
    root.write_text(text,encoding='utf-8')
    files=[q for q in H.iterdir() if q.is_file()]+[q for q in R.iterdir() if q.is_file()]
    write(H/'CLOSEOUT.json',dict(status='COMPLETE_FROZEN',at_kst=stamp,record=record,
       hashes={str(q):sha(q) for q in files},worker_exit_codes=s['autonomous_workers_current_unit'],
       root_and_state_updated=True,next_task_executed=False,project_complete=False))
    assert read(state_path)['motion_position_pair_20260922']['status']==record['status']
    assert root.read_text(encoding='utf-8').count(current)==1
    print(json.dumps(dict(status='CLOSED',at_kst=stamp,result=str(H/'RESULT.md'),next_task_started=False,project_complete=False)),flush=True)

if __name__=='__main__':main()
