"""Create the authorized isolated T3 experiment. Earlier units are read-only."""
from pathlib import Path
import json, shutil, datetime, difflib
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
BAL=O/'word100_balanced_batch_20261019'; PLAN=O/'word100_temporal_phase_plan_20261019'
def put(n,s):
 p=U/n;assert not p.exists(),str(p);p.write_text(s,encoding='utf8')
def js(n,x):put(n,json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def one(s,a,b):
 assert s.count(a)==1,a;return s.replace(a,b,1)
at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
 (U/'before').mkdir(exist_ok=True);assert not (U/'before'/p.name).exists();shutil.copyfile(p,U/'before'/p.name)
put('common.py',(BAL/'common.py').read_text(encoding='utf8').replace('Balanced training-batch experiment','Soft temporal summary experiment').replace('BAL31','T3'))
old=(BAL/'train.py').read_text(encoding='utf8');s=old.replace('BAL31','T3')
s=one(s,'Balanced minibatch paired runs.','Soft temporal summary paired runs.')
s=one(s,'from balanced import balanced_batches','from temporal import TemporalPhaseTCN')
s=one(s,'m=MaskedSignTCN(**MODEL);assert m.parameter_count()==37636',"m=TemporalPhaseTCN(**MODEL);assert m.parameter_count()==47236")
s=one(s,"statehash(m.state_dict())==read(BASE/f'training_C_{seed}_RESULT.json')['initial_state_sha256']","statehash({k:v for k,v in m.state_dict().items() if k!='phase_weight'})==read(BASE/f'training_C_{seed}_RESULT.json')['initial_state_sha256']")
s=one(s,"for step,indices in enumerate(balanced_batches(order,TRAIN['batch_size']),1):\n   rr=[train[i] for i in indices];","for step,j in enumerate(range(0,899,32),1):\n   rr=[train[i] for i in order[j:j+32]];")
s=one(s,"'feature_schema':'original_C_280_balanced_batches'","'feature_schema':'original_C_280_soft_temporal3','model_type':'wis_t3_masked_tcn','parameters':47236")
s=one(s,"'balanced.py'","'temporal.py'")
put('train.py',s)
put('TRAINER_CHANGE.diff',''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='prior_BAL_trainer_read_only',tofile='T3_trainer')))
s=(BAL/'verify_results.py').read_text(encoding='utf8').replace('BAL31','T3')
s=one(s,"assert result['initial_sha256']==read(BASE/f\"training_C_{result['seed']}_RESULT.json\")['initial_state_sha256']", "assert result['initial_sha256']==read(U/'READY.json')['initial_states'][str(result['seed'])]['full_state_sha256']")
s=one(s,"assert b['step']==step and len(b['clips'])==31 and len(b['choices'])==31","expected_size=32 if step<29 else 3\n   assert b['step']==step and len(b['clips'])==expected_size and len(b['choices'])==expected_size")
s=one(s,"for v in VIEWS:\n  for s in ['11','12','13']", "for v in VIEWS:\n  checks[v+'_failures_preserved']=sum(x['failures'] for x in new[v])<=sum(x['failures'] for x in old[v])\n  for s in ['11','12','13']")
s=one(s,'Batch partition intervention including gradient composition/padding/dropout/BN; not BN-only causal isolation.','Soft temporal summaries plus extra head intervention; not order-only causal isolation.')
put('verify_results.py',s)
s=(BAL/'matched_window_audit.py').read_text(encoding='utf8').replace('BAL31','T3')
s=one(s,'Secondary descriptive overlap comparison, added solely because candidate seed15 stops at epoch77 by frozen patience12.','Prespecified secondary descriptive overlap comparison if either paired history ends early by frozen patience12.')
# An early stop before epoch62 is reported as insufficient late-window coverage, never extended.
s=one(s,"x=[r[view+'_top1']/3 for r in h];", "if len(h)<2:return {'epochs':len(h),'first':h[0]['epoch'] if h else None,'last':h[-1]['epoch'] if h else None,**{k:None for k in ['accuracy_mean','sd_pp','adjacent_mean_pp','adjacent_max_pp','accuracy_min','accuracy_max','mean_ce']}}\n x=[r[view+'_top1']/3 for r in h];")
s=one(s,"sum(row[a][v][k] for row in out['paired'])/3", "sum(row[a][v][k] for row in out['paired'])/3 if all(row[a][v][k] is not None for row in out['paired']) else None")
put('matched_window_audit.py',s)
js('AUTHORIZATION.json',{'at':at,'operator':'Codex local','user_request':'너가 말한 주의할 점에 유의하면서 진행 계획대로 전부 진행해줘.','approved_plan':str(PLAN/'PLAN.json'),'allowed':['isolated T3 implementation','unchanged public input and model validation','conditional three fresh paired runs','saved score/trace audit and report'],'forbidden':['user/mentor recording access','production C/web/DTW replacement','extra arms or parameter search','new extraction/download','external data transfer','Git']})
p=json.loads((PLAN/'PLAN.json').read_text(encoding='utf8'));p.update(status='FROZEN_APPROVED_EXECUTION',training_authorized_now=True,current_authorization='User approved full proposed implementation, validation and conditional three-run comparison on this turn.',at=at,arm='T3',prior_plan_read_only=str(PLAN/'PLAN.json'))
js('FROZEN_PROTOCOL.json',p)
state=json.loads((O/'STATE.json').read_text(encoding='utf-8-sig'));nxt='T3隔離実装の入力・初期同等性・時間重み・勾配検証後、同一C条件で3seed学習と保存C比較。'
nxt='T3 별도 구현의 입력·초기 동등성·시간 가중·기울기 검증 후 동일 C 조건으로 3seed 학습과 저장 C 비교. 사용자/멘토 제외, 운영 C/웹/DTW 유지.'
state.update(current_stage='WORD100_TEMPORAL_PHASE_TRAINING_PREPARING',activity_state='IN_PROGRESS',next_single_action=nxt)
state[U.name]={'status':'PREPARING','at':at,'unit':U.name,'training_authorized':True,'production_changed':False,'user_mentor_data_used':False,'next_single_action':nxt}
(O/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=ctx.read_text(encoding='utf8')
s=s.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n### 2026-10-19 T3 계획 전체 수행 승인 — 검증 준비\nops/'+U.name+'. 사용자 최신 승인에 따라 별도 T3 구현·검증 후 3seed 학습/저장C비교 진행. 이전 계획-only 상태는 해당 시점 이력. 기존 C/웹/DTW 유지, 사용자/멘토 자료 제외.\n',1)
s=s.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n### 2026-10-19 승인 T3 검증→3회 학습→비교\n'+nxt+'\n',1);ctx.write_text(s,encoding='utf8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n갱신: '+at+'\n\n현재 '+U.name+' 준비 중. 사용자가 T3 계획 전체 실행 승인. 새 학습 시작 전 입력/초기 출력/RNG/패딩/미검출/기울기 검증. 3seed 신규 학습 vs 원래 C 저장3회 비교, 원래32×28+3배치. C/웹/DTW 유지, 사용자/멘토미사용. 이전 계획-only 완료는 이력이며 현재 승인범위를 제한하지 않음.\n\n다음: '+nxt+'\n',encoding='utf8')
print(U)
