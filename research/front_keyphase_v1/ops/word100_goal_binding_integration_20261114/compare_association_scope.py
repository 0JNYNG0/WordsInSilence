from pathlib import Path
import json,hashlib,collections,copy
from association_scope import scope_associations
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
f=O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json'
families=read(f)['words'];source=U/'FULL_GRAPH_COMPARISON.json'
write('ASSOCIATION_SCOPE_FROZEN.json',dict(code_sha256=sha(U/'association_scope.py'),source_sha256=sha(source),guide_sha256=sha(f),
 contract='A single observed posture cannot establish temporal progress. Keep candidates and every error measurement. Never filter by expected direction/order.',
 no_new_model=True,no_threshold_tuning=True,no_final8=True))
rows=[];checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
for r in read(source)['rows']:
 temporal={f['id']:f['axis']is not None for f in families[r['clip_id'].split('_')[2]]}
 snapshot=copy.deepcopy(r['cards']);scoped=scope_associations(r['cards'],temporal)
 ck('full_candidates_retained',r['cards']==snapshot and len(scoped['cards'])==len(snapshot))
 ck('no_completion_or_error_inferred',all(not c['permits_whole_completion']and not c['permits_educational_error']for c in scoped['cards']))
 for c in scoped['cards']:
  old=next(x for x in r['cards']if x['card']==c['card'])
  ck('all_witnesses_retained',c['role_witnesses']==old['role_confirmation']['matching'])
 if r['kind']in ['full_normal','speed075','speed125']:
  ck('normal_roles_preserved',all(c['state']=='role_observations_available'for c in scoped['cards']))
 rows.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],association=scoped))
# Target-specific test truth is used only here for auditing the edited final
# event, never as an input or branch to the observation policy above.
last=[]
for r in rows:
 if r['kind']not in ['omit_final_event','mask_final_event']:continue
 f=[s for s in families[r['clip_id'].split('_')[2]]if s['required']][-1]
 c=next(c for c in r['association']['cards']if c['card']==f['id'])
 last.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],state=c['state']))
ck('all_36_final_edits_accounted',len(last)==36)
ck('no_final_edits_promoted_to_observed_interval',all(r['state']!='role_observations_available'for r in last))
# A two-sample short observation is retained; a one-sample static handshape is
# allowed to be a posture observation. Neither is declared a complete action.
sample=dict(card='generic',family=999,chosen=dict(span=[0,1]),role_confirmation=dict(matching=[dict(family=999,span=[0,1])]))
ck('short_temporal_evidence_kept',scope_associations([sample],{'generic':True})['cards'][0]['state']=='role_observations_available')
sample['role_confirmation']['matching'][0]['span']=[0,0]
ck('one_point_temporal_not_progress',scope_associations([sample],{'generic':True})['cards'][0]['state']=='posture_snapshot_only')
ck('static_handshape_not_rejected_by_duration',scope_associations([sample],{'generic':False})['cards'][0]['state']=='role_observations_available')
write('ASSOCIATION_SCOPE_COMPARISON.json',dict(rows=rows,edited_final_actions=last,
 summary=dict(collections.Counter(r['state']for r in last)),model_inference=False,semantic_completion_accuracy=None,
 limitation='36 = six public videos x two artificial final edits x three seeds. Unknown/ambiguous is not proof of user omission.'))
write('ASSOCIATION_SCOPE_VERIFIED.json',dict(passed=all(c['passed']for c in checks),total=len(checks),checks=checks,
 failures=[c for c in checks if not c['passed']],same_Codex_separate_contract_checks=True,web_changed=False))
print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),last_status=collections.Counter(r['state']for r in last),failures=[c for c in checks if not c['passed']]),ensure_ascii=False))
