"""Read existing public experiment evidence; write a proposed plan, never train."""
from pathlib import Path
import json, hashlib, shutil
from datetime import datetime, timezone, timedelta
U = Path(__file__).resolve().parent
O = U.parent; ROOT = O.parents[2]
F = O/'word100_frozen_order_aux_train_20261019'
P = O/'word100_frozen_order_aux_preflight_20261019'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name, data):
    p = U/name; assert not p.exists(), p
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
stamp = datetime.now(timezone(timedelta(hours=9))).isoformat()
assert read(F/'CURRENT_EXECUTION.json')['running'] is False
assert read(F/'EXIT.json')['exit_code'] == 0
assert not (F/'TRAIN_RUNNING.json').exists()
audit = read(F/'FINAL_AUDIT.json')
for name,h in audit['files'].items(): assert sha(F/name)==h,name
analysis = read(F/'ANALYSIS.json'); conclusion = read(F/'CONCLUSION.json')
assert conclusion['sensor_delta_vs_T3F_pp']==0
math_review = read(U/'MATH_REVIEW.json'); assert math_review['optimizer_steps']==0
pins = {str(F/name):h for name,h in audit['files'].items()}
pins[str(F/'FINAL_AUDIT.json')] = sha(F/'FINAL_AUDIT.json')
pins[str(P/'frozen_order_aux.py')] = sha(P/'frozen_order_aux.py')
# Preserve model checkpoints and web integration sources without importing them.
for p,h in read(F/'TRAIN_CONTEXT.json')['pins'].items():
    if p.endswith('_best.pt') or '/service/' in p.replace('\\','/'):
        assert sha(p)==h,p; pins[p]=h
write('SCOPE.json',{'at':stamp,'request':'보조 부분 자체에도 단어를 구별하도록 별도의 학습 신호를 주는 방안 검토; 추가 학습 전에 구체적인 계획 보고',
    'scope':'Review saved public evidence, loss mathematics, proposed paired experiment and documentation only',
    'training_authorized_this_turn':False,'implementation':'Codex local review scripts only; no model/trainer changes',
    'verification':'Same Codex separate mathematical verification and evidence hash checks; no independent agent',
    'source_pins':pins,'no_user_mentor_recordings':True,'preserve_C_T3F_web_DTW':True})
plan = {
 'status':'PROPOSED_WAIT_USER_REVIEW_NO_TRAINING','candidate':'F-G16-AUX01','at':stamp,
 'hypothesis':'Combined CE is already tiny on TRAIN; directly supervising the same auxiliary logits may make the new memory branch discriminative. Supported hypothesis, not established cause or promised gain.',
 'objective':{'formula':'mean CE(stop_gradient(b)+a,y) + 0.1 * mean CE(a,y)',
     'b':'Fully frozen matching-seed T3-F logits','a':'Existing GRU16 last-real-frame state through the SAME zero-initialized bias-free 100-class residual head',
     'target':'Same public word label y for both terms; no new order/completion labels',
     'lambda_aux':0.1,'final_loss_coefficient':1.0,'reduction':'mean per batch in each term',
     'inference':'b+a, unchanged coefficient1; no additional inference head or loss at inference',
     'rationale':'Keep original final loss at1 and add one fixed auxiliary term at0.1. A simple prespecified experimental weight, not data-derived optimum or a 10% influence guarantee. It will dominate the initially tiny final loss.',
     'no_sweep':True,'no_temperature_label_smoothing_class_weights_or_gates':True},
 'alternatives':[
     {'option':'Combined CE only','decision':'Stored F-G16 control; no fresh run needed unless provenance mismatch.'},
     {'option':'Auxiliary CE only','decision':'Not chosen: lacks direct pressure to protect combined predictions.'},
     {'option':'Separate classifier head for auxiliary task','decision':'Not chosen now: adds architecture/parameters, would obscure effect of loss alone.'},
     {'option':'Auxiliary pretraining then final-loss finetuning','decision':'Not chosen now: changes phases and budget; combined objective is the smaller controlled change.'}],
 'fixed':{'base_parameters_and_BN':47236,'trainable_parameters':4000,'memory':'one-layer unidirectional GRU16 on frozen contextual C features; entire model not causal',
     'input_channels':280,'input':'Exact prior C input tensors, masking, normalization, dt, whole/context schedule and view schedule',
     'internal_missing_frames':'Retain chronology; only batch trailing padding ignored',
     'initialization':'Fresh each seed from matching original T3-F and exact prior preflight initial-state hash; do not resume trained F-G16',
     'zero_head':'Initial final output equals T3-F. First step can update head but memory gradient zero; subsequent nonzero head can pass gradients.'},
 'data':{'train_videos':899,'train_signer_ids':list(range(1,10)),'dev_videos':300,'dev_signer_ids':[11,12,13],'word_classes':100,
     'do_not_read':['User recordings','Mentor recordings','ID14-18 raw data'],
     'reserved_final_holdout':'ID14-16 stays untouched; existing ID17/18 evaluation is not a new blind test.',
     'limitation':'DEV used repeatedly for model development and selection; not unbiased new-user accuracy. Three seeds are not900 independent videos.'},
 'runs':{'count':3,'seeds':[20261013,20261014,20261015],'optimizer':'AdamW','lr':0.001,'weight_decay':0.001,'clip_norm':1.0,
     'batch_size':32,'max_epochs':80,'early_stop_patience':12,'cpu_threads':4,
     'checkpoint_selection':'Strict >1e-6 improvement in mean FINAL combined DEV CE across masked/sensor views; auxiliary DEV CE is diagnostic only. No selection by top1, auxiliary loss, personal data or order perturbation.',
     'early_stop_risk':'May stop before auxiliary becomes useful; do not extend epochs or change selection after seeing results.'},
 'comparison':{'primary':'Frozen T3-F and F-G16(lambda0) stored per-seed predictions','secondary':'C stored predictions',
     'metrics':['Final top1/top5 mean and range across3seeds, %p versusTF and FG16','Per-seed paired newly correct/newly incorrect clips','Per-signer and frozen missingness groups','Target rank and confusion across all100 classes','Auxiliary-only top1/CE for TRAIN and DEV from the same forward outputs, diagnostic only','Final CE, auxiliary CE, gradient norms before clipping, update norms, auxiliary logit magnitude and relative class margins'],
     'report_all_seeds':True,'no_final_rescoring':True,
     'late_stability':'61-80 window only if reached; otherwise unavailable; summarize all actually observed epochs',
     'success_gate':'Final sensor mean top1 > T3-F and FG16, with improvement in>=2/3seeds; mean masked top1, both views top5, per-signer and frozen missingness-group mean counts do not regress againstT3-F; no increased input failures. Report every per-seed regression even if mean passes.',
     'mechanism_interpretation':'Auxiliary accuracy alone is not project improvement. Paired lambda0/0.1 isolates this objective change within the setup; does not establish semantic order understanding or causality of every remaining error.',
     'if_no_gain':'Keep C/T3-F. Report failure/no gain and do not automatically retune or run more seeds.'},
 'preflight_after_approval':[
     'Freeze source/checkpoint/input hashes and exact3seed initial states; loss0 mode must match old computation.',
     'Check final loss and weighted aux loss against independent arithmetic; finite losses and gradients.',
     'Check original47236 tensors/BN fixed and optimizer includes only five new tensors totaling4000.',
     'Check zerohead initial prediction equality; first head gradient and later memory gradient on an isolated synthetic check, without changing experiment initial states or touching DEV.',
     'Reuse unchanged input/padding/chronology validation evidence; only rerun checks affected by loss/logging changes.',
     'Record actual batch source IDs/views/segments, each loss, optimizer steps and before/after frozen hashes. Missingness groups fixed beforehand.'
 ],
 'risks':[
     'Auxiliary scores can grow and overturn a correct base answer even when base weights are frozen; joint loss and regression checks reduce but cannot eliminate this risk.',
     'Same word labels can be solved using partial/static cues; no explicit order/completion teaching. Do not label all reversed/cropped sequences as invalid signs.',
     'Frozen TCN features may already lose discriminative motion; supervision cannot restore undetected hand detail.',
     'Overfitting and near-ceiling repeatedly-used DEV can hide problems. Small gradients alone do not quantify AdamW updates.',
     'No guaranteed improvement percentage; no generalization-to-all-signs claim.'
 ],
 'rollout':'After reviewed plan: isolated implementation/preflight ->3runs -> saved-score/trace/tensor verification -> report. No automatic web/model/DTW replacement or personal-data use.',
 'current_execution':{'training_runs':0,'optimizer_steps':0,'project_model_inferences':0,'raw_recordings_read':0},
 'references':[{'title':'PyTorch2.12 CrossEntropyLoss','url':'https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html','scope':'CE definition and reduction, not evidence of expected accuracy'},
     {'title':'Deeply-Supervised Nets (2014)','url':'https://arxiv.org/abs/1409.5185','scope':'General precedent for companion classification objectives; not this architecture, coefficient or KSL performance validation'}]
}
write('PROPOSED_PROTOCOL.json',plan)
write('REVIEW.json',{'status':'COMPLETE_PLAN_ONLY','at':stamp,'prior_sensor_mean_percent':{'C':98.66666666666667,'T3F':99.22222222222223,'FG16':99.22222222222223},
    'prior_new_correct_vs_T3F':0,'prior_new_wrong_vs_T3F':0,'prior_first_train_ce':[r['train_ce_first'] for r in analysis['rows']],
    'interpretation':plan['hypothesis'],'math_review':math_review['status'],'training_ready':False,'candidate_implemented':False,
    'training_runs':0,'user_recordings_read':0,'plan':'PROPOSED_PROTOCOL.json','implementation':'Codex','verification':'Same Codex, separate NumPy/finite difference versus torch calculations; no external reviewer'})
before=U/'before';before.mkdir(exist_ok=False)
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
    shutil.copy2(p,before/p.name)
print('Plan written; all old evidence/checkpoints verified, no training or project inference.')
