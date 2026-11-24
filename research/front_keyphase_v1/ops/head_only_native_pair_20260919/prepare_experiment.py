"""Freeze one bounded follow-up comparison after independent new-code checks."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy
import hashlib
import json

OPS=Path(__file__).resolve().parent
WEB=OPS.parents[1]
OLD=WEB/'ops/visibility_native_pair_20260919'
RUN=WEB/'runs/head_only_native_pair_v1'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):
    with p.open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

assert not RUN.exists() and not (OPS/'plan.json').exists()
verification=read(OPS/'synthetic_verification.json')
assert verification['status']=='PASS_SYNTHETIC_HEAD_ONLY'
assert verification['passed']==verification['total']
helper=WEB/'tools_head_only_native_pair.py'
assert sha(helper)==verification['helper_sha256']
old_ready=read(OLD/'execution_ready.json')
for path,digest in old_ready['frozen_execution_inputs'].items():
    assert sha(Path(path))==digest, path
plan=copy.deepcopy(read(OLD/'plan.json'))
plan['head_only']={'encoder_lr':0.0,'freeze_encoder_buffers':True,'training_scope':'head_only_frozen_encoder_and_buffers'}
policy=copy.deepcopy(read(OLD/'comparison_policy.json'))
policy.update({
    'prepared_at_kst':datetime.now(timezone(timedelta(hours=9))).isoformat(),
    'question':'Can classifier-head-only updates incorporate the approved native data addition while retaining the existing frozen-encoder model behavior?',
    'reason':'The prior full fine-tuning control and treatment both lost reviewed missing-hand recognition. A statistics-only counterfactual was insufficient. This run freezes encoder parameters AND all buffers throughout optimization.',
    'changed_from_prior_native_pair':['Training scope: head only; encoder learning rate is zero, encoder eval mode and buffers frozen.'],
    'legacy_plan_encoder_lr':'The unchanged generic preflight requires a positive inherited field. head_only.encoder_lr explicitly overrides it to zero before training; saved effective settings MUST have encoder lr 0.',
    'effective_learning_rates':{'encoder':0.0,'head':0.001},
    'unchanged':['Initial visibility checkpoint','all data IDs','signer split','approved semantics','seed20260919','660 steps','batch32','head learning rate and weight decay','93 classes','no new augmentation','fixed evaluation cohorts'],
    'acceptance_note':'Report all changes versus paired control and historical frozen parent. Head-only freeze success alone is not recognition success; no default promotion.',
})
write(OPS/'plan.json',plan)
write(OPS/'comparison_policy.json',policy)
pins=[helper,WEB/'tools_native_data_pair.py',WEB/'tools_prep_union93_candidate.py',WEB/'tools_visibility_experiment.py',
      OPS/'plan.json',OPS/'comparison_policy.json',OPS/'synthetic_verification.json',OPS/'verify_head_only.py',
      OPS/'claude_generic_contract.txt',OPS/'claude_response.json',Path(plan['checkpoint']),Path(plan['base_inputs']),
      Path(plan['candidate_report']),Path(plan['semantic_review']['evidence']),
      WEB/'ops/native_data_pair_20260919/audit_native_pair_result.py',Path(policy['evaluations_fixed_before_training']['web']['plan'])]
write(OPS/'execution_ready.json',{'status':'NEW_HEAD_ONLY_PATH_VERIFIED_AND_FIXED_PLAN_READY',
    'at_kst':policy['prepared_at_kst'],'authorship':'Claude Code implemented the isolated training helper; Codex applied and independently verified it.',
    'completed_code_checks_reused':'Original generic runner provenance/split/metadata checks reused; only the new head-only behavior tested synthetically.',
    'frozen_execution_inputs':{str(p):sha(p) for p in pins},'out':str(RUN)})
print(json.dumps({'status':'READY','new_helper_checks':verification['passed'],'training_started':False,'effective_encoder_lr':0}))
