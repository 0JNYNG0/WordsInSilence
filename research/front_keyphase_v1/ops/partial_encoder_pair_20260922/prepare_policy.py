"""Local operational policy assembly; no observation arrays or training."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json

H=Path(__file__).resolve().parent;L=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')

parent=L/'runs/public_anchor_0p01_pair_v1/checkpoint_treatment_final.pt'
assert sha(parent)=='b8d6045c0c154697a5d6d3440927a48cf1e9db3146421db0a920da9fa684870d'
paths={
 'metadata':L/'ops/context_training_pair_20260921/feature_metadata.json',
 'old_features':L/'ops/context_training_pair_20260921/public_features.npz',
 'old_feature_audit':L/'ops/context_training_pair_20260921/FEATURE_AUDIT.json',
 'schedule':L/'runs/context_head_pair_v1/training_arrays.npz',
 'teacher_scores':L/'ops/public_prediction_retention_20260922/training_scores_and_gram.npz',
 'nikl_manifest':L/'runs/union93_candidate_v1/inputs.json',
 'visibility_provenance':L/'runs/visibility_candidate_v1/provenance.json',
 'mp_plan':L/'ops/public_window_sensitivity_20260921/plan.json',
 'old_eval':L/'runs/output_retention_candidate_v1/evaluation.json',
 'reference_candidate':L/'runs/output_retention_candidate_v1/checkpoint_final.pt',
 'adapter':L/'tools_prep_union93_candidate.py',
 'channels':L.parent/'sign_model_pilot_v1/src/pilot/data.py',
 'model':L.parent/'sign_model_pilot_v1/src/pilot/model_masked_v1.py'}
p={
 'schema':'partial_encoder_matched_pair_v1','at_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
 'final_review':'Existing full finetuning regressed; encoder freezing is not proven sole cause. One matched limited-unfreezing comparison can test this without new source acquisition or input changes.',
 'parent':{'path':str(parent),'sha256':sha(parent)},
 'paths':{k:str(v) for k,v in paths.items()},'pins':{str(v):sha(v) for v in paths.values()},
 'arms':{'control':'head.* only','treatment':'head.* and all parameters of blocks.3.* only'},
 'fixed':'Both arms same initial checkpoint,280 input channels,93 classes,base2122,context224,original schedule. All modules eval; dropout off; all buffers and prefix parameters fixed. No new augmentation.',
 'settings':{'steps':660,'half_batch':16,'learning_rate':0.001,'weight_decay':0.001,'anchor_coefficient':0.01,'output_coefficient':1.0,'cpu_threads':2,'adamw_betas':[0.9,0.999],'adamw_eps':1e-8},
 'loss':'CE over32 + .5*.01*sum squared trainable parameter change + .5*mean squared class-centered student-teacher logit delta on16 base draws. Teacher is fixed parent. Both arms identical formula; no development labels in losses.',
 'objective_change':'Sampled base output preservation replaces exact full-base fixed-feature Gram penalty. This change affects BOTH arms; historical d1 candidate is reference, not matched control. New control training is necessary.',
 'caching':'Cache per-frame frozen-prefix output before blocks.3, derived from original temporal inputs. Old pooled64D vectors cannot train changed block. New cache must reproduce old parent pooled outputs within atol1e-4/rtol1e-5 and same top1 before training.',
 'boundaries':{'train_signers':list(range(1,10)),'development_signers':[11,12,13],'reserved_signers':[14,15,16],'official_excluded':[17,18],'unassigned':[10],'quarantined':[25],'user_records':'never read/train/select in this unit','ksl':'not used','public_mp7':'evaluation only,train identities01/02; not new-person web validation'},
 'evaluation_groups':{'annotation_dev':699,'whole_dev':74,'public':32,'mp':14},
 'gates':{
  'annotation_count':'treatment full93 count >= max(690,control count)',
  'annotation_macro_recall_improvement':'treatment mean recall over the81 observed ground-truth classes > control by more than1e-12; report macroF1 too',
  'annotation_each_signer':'treatment full93 count for each of11/12/13 >= matched control',
  'annotation_target':'project18 supported33 count >= max(32,control)',
  'apology_dev':'all3 correct in full93 and project18',
  'whole':'all74 full93 and all30 supported target18 correct',
  'public':'all18 intact,5 approved masked,9 source training correct both scopes',
  'mp':'all7 whole and7 annotation correct both scopes'},
 'decision':'Eligible only if all gates pass and independent audit passes; no automatic default promotion. No robustness claim for12 reference-only target labels.',
 'stopping':'Exactly one pair of660 updates. No result-driven coefficient/seed/step/threshold search; preserve failure. Fix actual implementation contract bugs before real training; no retraining for poor scores.',
 'roles':'Claude writes generic learner/cache functions. Codex local orchestration and independent verification. No observations/arrays/checkpoints sent externally.',
 'not_done':'New-person MediaPipe generalization,execution correctness,feedback validity,Unity integration remain unverified.'}
write(H/'comparison_policy.json',p)
print(json.dumps({'status':'POLICY_FROZEN','sha256':sha(H/'comparison_policy.json'),'train_runs_planned':2,'new_data':0,'next':'validate generic implementation and prepare train prefix cache'}))
