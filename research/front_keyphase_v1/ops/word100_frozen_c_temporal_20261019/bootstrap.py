from pathlib import Path
import json,shutil,datetime,difflib
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];T=O/'word100_temporal_phase_training_20261019';D=O/'word100_t3_regression_diagnosis_20261019'
def put(n,s):
 p=U/n;assert not p.exists(),str(p);p.write_text(s,encoding='utf8')
def js(n,d):put(n,json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def one(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b,1)
for n in ['TRAIN_RUNNING.json'] :assert not (T/n).exists()
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
 (U/'before').mkdir(exist_ok=True);assert not (U/'before'/p.name).exists();shutil.copyfile(p,U/'before'/p.name)
put('common.py',(T/'common.py').read_text(encoding='utf8').replace('Soft temporal summary experiment','Frozen trained C temporal auxiliary experiment').replace("ARMS=('T3',)","ARMS=('T3F',)"))
put('temporal.py',(T/'temporal.py').read_text(encoding='utf8'))
old=(T/'train.py').read_text(encoding='utf8');s=old.replace('T3','T3F')
s=one(s,'from temporal import TemporalPhaseTCN','from frozen_temporal import FrozenCTemporal')
start=s.index('def initialize(seed):');end=s.index('\ndef batch(',start)
s=s[:start]+'''def initialize(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 path=BASE/f'training_C_{seed}_best.pt';r=read(BASE/f'training_C_{seed}_RESULT.json')
 assert sha(path)==r['checkpoint_sha256']
 ck=torch.load(path,map_location='cpu',weights_only=True)
 m=FrozenCTemporal(**MODEL)
 missing,unexpected=m.load_state_dict(ck['state_dict'],strict=False)
 assert missing==['phase_weight'] and not unexpected
 assert m.parameter_count()==47236 and sum(p.numel() for p in m.parameters() if p.requires_grad)==9600
 assert statehash(m.frozen_state())==statehash(ck['state_dict'])
 m.assert_frozen();return m

'''+s[end:]
s=one(s,"m=initialize(seed);init=statehash(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=TRAIN['lr'],weight_decay=TRAIN['weight_decay'])", "m=initialize(seed);init=statehash(m.state_dict());frozen_hash=statehash(m.frozen_state());opt=torch.optim.AdamW([m.phase_weight],lr=TRAIN['lr'],weight_decay=TRAIN['weight_decay'])\n probe_x,probe_p,_,_=batch(train[:8],arm,epoch=1)\n with torch.no_grad():probe_base=m.base_and_contrast(probe_x,probe_p)[0].clone()")
s=one(s,"torch.nn.utils.clip_grad_norm_(m.parameters(),1.)","torch.nn.utils.clip_grad_norm_([m.phase_weight],1.)")
s=one(s,"tr.append({'step':step,'clips':[r['clip_id'] for r in rr],'choices':choices})", "m.assert_frozen();assert statehash(m.frozen_state())==frozen_hash\n   assert all(p.grad is None for name,p in m.named_parameters() if name!='phase_weight')\n   tr.append({'step':step,'clips':[r['clip_id'] for r in rr],'choices':choices,'frozen_state_sha256':frozen_hash,'frozen_modules_eval':True})")
s=one(s,"pred=scores(m,dev,arm);metrics=", "with torch.no_grad():assert torch.equal(m.base_and_contrast(probe_x,probe_p)[0],probe_base)\n  pred=scores(m,dev,arm);metrics=")
s=one(s,"'trace_sha256':sha(tp)}","'trace_sha256':sha(tp),'frozen_state_sha256':frozen_hash,'base_probe_max_error':0.0}")
s=one(s,"'model_type':'wis_t3_masked_tcn'","'model_type':'wis_t3f_frozen_c_temporal'")
s=one(s,"'feature_schema':'original_C_280_soft_temporal3'","'feature_schema':'original_C_280_frozen_C_soft_temporal3','base_checkpoint_sha256':sha(BASE/f'training_C_{seed}_best.pt'),'frozen_state_sha256':frozen_hash,'trainable_parameters':9600")
s=one(s,"'actual_batches_logged':True,","'actual_batches_logged':True,'frozen_state_sha256':frozen_hash,'base_checkpoint_sha256':sha(BASE/f'training_C_{seed}_best.pt'),'trainable_parameters':9600,'all_steps_frozen_checked':True,'base_probe_max_error':0.0,")
s=one(s,"'temporal.py','FROZEN_PROTOCOL.json'","'temporal.py','frozen_temporal.py','FROZEN_PROTOCOL.json'")
put('train.py',s);put('TRAINER_CHANGE.diff',''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='previous_T3_read_only',tofile='T3F')))
at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
js('AUTHORIZATION.json',{'at':at,'user_request':'기존 C가 배운 내용을 고정하고 시간요약 보조 부분만 학습하는 후보. 동결 검증 통과 시3회 학습하여C/T3비교. 기존C유지,사용자촬영학습·튜닝금지.','operator':'Codex local','approved_plan':str(D/'NEXT_PLAN.json'),'allowed':['isolated T3F implementation','freeze and numerical tests','conditional3runs','saved C/T3 comparison and report'],'forbidden':['user/mentor recordings','heldout people','production/web/DTW change','new extraction/download','external data transfer','Git','new candidate search']})
protocol=json.loads((D/'NEXT_PLAN.json').read_text(encoding='utf8'));protocol.update(at=at,status='FROZEN_APPROVED_THREE_RUNS',training_authorized_now=True,arm='T3F',initialization='Three paired pretrained C checkpoints from original selected epochs77/76/75. Not fresh T3 training. Zero extra head. Only extra head optimized. Epoch0 not a reselection option; unchangedbestCEoverepochs1..80.',input_condition='Exact original899TRAIN/300DEV masked/sensor whole/context, no new features or landmark processing.',execution='CPU4 deterministic. Same batches/schedule/AdamW/max80/patience12. Check frozen parameter+buffer hash every update and fixed publicTRAIN base-output probe everyepoch. FrozenC eval/no_grad, phase head trainable. No caching approximation.',comparison_interpretation='Total effort includes pretraining of C; head-only timing is not from-scratch training cost. Three fixed seeds/base checkpoints, no best seed selection.')
js('FROZEN_PROTOCOL.json',protocol)
nxt='T3-F 동결/동등성/기울기 검증 통과 후 동일899TRAIN/300DEV에서보조head만3회학습,C/T3저장점수비교. 기존C/웹/DTW와사용자자료보존.'
state=json.loads((O/'STATE.json').read_text(encoding='utf8'));state.update(updated_date_kst=at,current_stage='WORD100_FROZEN_C_TEMPORAL_PREPARING',activity_state='IN_PROGRESS',next_single_action=nxt);state[U.name]={'status':'PREPARING','at':at,'training_authorized':True,'operator':'Codex local','production_changed':False,'user_mentor_data_used':False,'next_single_action':nxt};(O/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf8');s=s.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n### 2026-10-19 T3-F 사용자 승인 — 동결 검증 준비\nops/'+U.name+'. 각seed 학습된C를고정하고기존시간요약head9600만학습하는후보. 사용자승인에따라동결검증후3회학습/C·T3저장비교. 기존C/웹/DTW유지,사용자멘토자료제외.\n',1);s=s.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n### 2026-10-19 승인 T3-F 검증→3회 비교\n'+nxt+'\n',1);p.write_text(s,encoding='utf8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n갱신: '+at+'\n\n현재 '+U.name+' 준비. 사용자 T3-F 계획 실행 승인. 학습된C3개 각각고정+기존T3시간head만학습. 초기C출력동등성·파라미터/BN/dropout동결·optimizer/gradient검증전학습금지. 통과시3seed C/T3저장결과비교까지완료. 기존운영C/웹/DTW보존·사용자멘토촬영미사용.\n\n다음: '+nxt+'\n',encoding='utf8')
print('Bootstrapped authorized T3F')
