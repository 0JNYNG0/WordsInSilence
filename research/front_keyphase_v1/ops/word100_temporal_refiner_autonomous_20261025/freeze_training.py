from pathlib import Path
from datetime import datetime
import json,hashlib
U=Path(__file__).resolve().parent;B=U.parent/'word100_temporal_public_bundle_20261025'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
auth=read(U/'AUTHORIZATION.json');policy=read(U/'TEACHER_POLICY.json');review=read(U/'REVIEW.json')
assert auth['training_authorized'] and policy['train_signer_ids']==[1,2,3,4,5,6]
assert review['status']=='PASS_FOR_BOUNDED_ROBUST_PROXY_EXPERIMENT_NOT_ANATOMICAL_GT'
assert not review['blocking_findings_remaining']
assert all(review['static_contract_review'].values())
for name,pin in review['files_reviewed'].items():assert sha(U/name)==pin,(name,'changed after independent review')
for path,value in read(B/'PROTECTED_BEFORE.json').items():assert sha(path)==value,path
required=[U/n for n in ('train_compare.py','refiner_model.py','teacher_reliability.py','TEACHER_POLICY.json','RGB_INDEX.json','RGB_FEATURE_RESULT.json','RESEARCH_CONTRACT.json','AUTHORIZATION.json','REVIEW.json')]
required += [B/n for n in ('MANIFEST.json','SCENARIOS.json','load_inputs.py')]
pins={str(p.resolve()):sha(p) for p in required}
p=dict(status='FROZEN',training_authorized=True,at=datetime.now().astimezone().isoformat(),
    bundle_dir=str(B),features_manifest=str(U/'RGB_INDEX.json'),source_pins=pins,
    splits={'train':[1,2,3,4,5,6],'dev':[7],'test':[8,9]},
    training={'seeds':[17,29,43],'epochs':80,'batch_size':16,'lr':.001,'weight_decay':.0001},
    model={'window':11,'hidden':48,'pose_projection':64,'rgb_projection':32,'rgb_dim':576},
    loss={'name':'quality_weighted_smooth_l1','beta':.05,'selection':'DEV07 weighted proxy loss, earliest minimum'},
    cpu_threads=4,
    eligibility='Controlled robust-pseudo-label experiment; independent anatomical-truth adequacy remains unproven.',
    limitations=['B2/B3 2D only; no hidden-finger truth','synthetic pixel-visible corruption is distinct from physical occlusion','original observed errors are preserved','people08/09 held out from refiner, previously seen by classifiers'],
    adoption='No production promotion; all seeds and negative outcomes reported. Test results cannot change criteria.')
write(U/'FROZEN_PROTOCOL.json',p)
write(U/'VISIBLE_READINESS.json',dict(status='PASS_ROBUST_PROXY_EXPERIMENT',training_allowed=True,
    protocol_sha256=sha(U/'FROZEN_PROTOCOL.json'),original_pilot_hold_preserved=True,anatomical_truth_ready=False,
    reason='New user-authorized bounded claim: weighted noisy pseudo-target reconstruction, independent image diagnostics, and fixed-model compatibility separately evaluated.',
    failed_old_assumption='Uniformly correct MediaPipe21-point targets; no longer asserted.'))
print(json.dumps({'frozen':sha(U/'FROZEN_PROTOCOL.json'),'training_ready':'robust proxy experiment only'}))
