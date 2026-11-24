from common import *

def main():
 assert not (U/'AUTHORIZATION.json').exists()
 p=O/'word100_feature_expansion_plan_20261113/PLAN.json'
 write(U/'AUTHORIZATION.json',dict(user_approved='2026-11-13 all six selective expansion stages, retain original graph work, parallel public multi-view avatar review',parent_plan=str(p),parent_plan_sha256=sha(p),prior_waiting_status_superseded=True,extra_word_cap=20,production_tcn_classes=100,all_3000_training=False,private_training=False,production_replacement=False,avatar_scope='source/rig/method feasibility review, public FIT only; no private recordings/new downloads/external transfer',delegation=dict(labels='expansion_labels_review_1113',old_graph_C7='c7_graph_recovery_1113',avatar='avatar_multiview_review_1113',root='input/extraction/network/training/evaluation and integration verification')))
 write(U/'GRAPH.json',dict(nodes={'G1_select':'RUNNING','G2_labels':'RUNNING','G3_inputs':'RUNNING','G4_train':'WAIT_LABEL_VALIDATION','G5_regression':'WAIT_TRAIN','G6_feedback':'WAIT_VALIDATION','C7_motion_hold':'PARALLEL_CAUSE_AUDIT','AV_avatar':'PARALLEL_SOURCE_AND_RIG_REVIEW'},links=[['G1_select','G2_labels'],['G1_select','G3_inputs'],['G2_labels','G4_train'],['G3_inputs','G4_train'],['G4_train','G5_regression'],['C7_motion_hold','G5_regression'],['G5_regression','G6_feedback']],loops=dict(label_conflict='G2_labels',missing_observation='G3_inputs',training_regression='G4_train',action_link_failure='C7_motion_hold'),production_change=False))
 write(U/'CURRENT_EXECUTION.json',dict(status='PREPARATION_AND_PARALLEL_REVIEWS',running=None))
 # Reflect explicit approval, without rewriting historical frozen plan.
 state=read(O/'STATE.json');state['active_task']=U.name;state['next_single_action']='승인된6단계 진행: 공통특징 누락대비 선별·다인정답검수·RTMW추출재사용,이후고정9학습비교. 기존C7와공개5뷰아바타검토 병렬. C·T3-F/현재웹/개인자료금지유지.'
 state[U.name]=dict(status='AUTHORIZED_IN_PROGRESS',authorization=str(U/'AUTHORIZATION.json'),graph=str(U/'GRAPH.json'),training_runs=0,production_changed=False,private_used=False)
 write(O/'STATE.json',state)
 ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=ctx.read_text('utf-8-sig');title='### 2026-11-13 사용자6단계 실행 승인·기존그래프와아바타 병렬 검토'
 for h,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','사용자가선별확장1~6전체실행을승인. 종전승인대기해제. Codex root는입력/학습비교,별도에이전트는RGB정답/C7기존그래프/공개5뷰아바타검토를담당. 모델·성능완료아직아님.'),('## 9. 다음 단일 작업',state['next_single_action'])]:
  t=t.replace(h,h+'\n\n'+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/AUTHORIZATION.json`, `GRAPH.json`.\n',1)
 ctx.write_text(t,'utf-8')
 h=ROOT/'NEW_CHAT_HANDOFF.md';old=h.read_text('utf-8');h.write_text('# 최신 승인 — 2026-11-13 실행 재개\n사용자가선별확장1~6단계와기존C7그래프진행,아바타공개5뷰병렬검토를명시승인했다. 아래이전승인대기상태는해제되었다. 현재단위='+str(U)+'\nAUTHORIZATION/GRAPH/CURRENT_EXECUTION와에이전트실행상태를확인하여중복하지않는다. root입력/학습,labels/C7/avatar분리위임. 보호규칙유지.\n\n'+old,'utf-8')
 event('authorized')
if __name__=='__main__':main()
