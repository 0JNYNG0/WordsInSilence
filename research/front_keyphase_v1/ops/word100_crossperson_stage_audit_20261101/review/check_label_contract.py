"""Independent label/denominator check, without reading pair predictions."""
from pathlib import Path
from collections import Counter
import json,hashlib,ast
U=Path(__file__).resolve().parents[1];O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0
def check(v,n):
 global checks
 checks+=1
 if not v:raise AssertionError(n)
def canon(k,s):
 if k=='bilateral_open_hands_lift_and_approach_center':return 'bilateral_open_hands_converge_near_face:move'
 if k=='right_closed_hand_reorients_toward_shoulder':return k+(':hold' if s in ('hold','post_reorientation_hold') else ':move')
 if k=='bilateral_open_hands_converge_near_face':return k+(':move' if s=='converging' else ':hold')
 return k

rows=read(U/'AUTHORIZATION.json')['rows'];labels=read(U/'PHYSICAL_CORRESPONDENCE.json')
fit=read(U/'review/FIT_PHYSICAL_STAGES.json');dev=read(U/'review/DEV_PHYSICAL_STAGES.json')
ph={c['alias']:c for c in read(O/'word100_core_stage_evidence_20261031/PHASE_DATA.json')['clips']}
merged={c['alias']:c for c in labels['clips']}
check(set(merged)=={r['alias'] for r in rows} and len(rows)==16,'16 aliases')
check(labels['neural_training_target'] is False and labels['linguistic_truth'] is False,'physical draft only')
check(labels['source_hashes']=={n:sha(U/'review'/n) for n in ['FIT_PHYSICAL_STAGES.json','DEV_PHYSICAL_STAGES.json']},'source labels unchanged')
clipstats=[]
for origin,src in [('fit',fit),('development',dev)]:
 for c in src['clips']:
  alias=c['alias'];known={};outer=set();parts=[];m=merged[alias];n=c['native_frame_count']
  check(sha(c['source_webm'])==c['source_webm_sha256'],alias+' media hash')
  if origin=='fit':
   original={s['segment_id']:s['core'] for s in ph[alias]['segments'] if s['role']=='core_candidate'}
   check({s['segment_id']:s['core'] for s in c['segments']}==original,alias+' old cores unchanged')
  for s in c['segments']:
   for p in s['parts'] if origin=='fit' else [s]:
    core=p['core'] if origin=='fit' else p['certain_interior'];envelope=p['outer'] if origin=='fit' else [p['start_bracket'][0],p['end_bracket'][1]]
    rid=p['part_id'] if origin=='fit' else p['stage_id'];kind=canon(p['kind_key'] if origin=='fit' else p['proposed_kind_key'],p['motion_subphase'])
    occ=s['occurrence_index_candidates'] if origin=='fit' else s['cycle_or_unknown'];occ=occ[0] if isinstance(occ,list) and len(occ)==1 else None if isinstance(occ,list) else occ
    for i in range(core[0],core[1]+1):
     check(i not in known,alias+' unique interior');check(i in c['reviewed_native_frame_ids'],alias+' actual reviewed frame')
     known[i]=(rid,kind,occ)
    outer.update(range(envelope[0],envelope[1]+1))
  labelsby={}
  for r in m['regions']:
   check(r['kind'] is None or r['kind'] in labels['kinds'],alias+' kind catalog')
   for i in range(r['outer'][0],r['outer'][1]+1):
    check(i not in labelsby,alias+' merged disjoint');labelsby[i]=r
  check(set(labelsby)==outer,alias+' envelope retained')
  for i in range(n):
   r=labelsby.get(i)
   if i in known:
    check(r is not None and (r['id'],r['kind'],r['occurrence'])==known[i],alias+' interior source exact')
    check(r['core']==r['outer'] and r['fine_finger_3d_or_exact_contact_validated'] is False,alias+' no boundary/precision promotion')
   elif i in outer:check(r['kind'] is None and r['core'] is None and r['region_type']=='boundary',alias+' uncertainty retained')
   else:check(r is None,alias+' outside not invented')
  check(m['reviewed_known_frames']==len(known) and m['uncertain_transition_frames']==len(outer-set(known)) and m['outside_reviewed_scope_frames']==n-len(outer),alias+' denominator partition')
  clipstats.append(dict(alias=alias,known=len(known),uncertain=len(outer-set(known)),outside=n-len(outer)))

pairs=[(q,r) for q in rows for r in rows if q['alias']!=r['alias'] and r['role']=='fit' and q['word_id']==r['word_id']]
opportunities=[(q['alias'],r['alias'],s['segment_id'],q['role']) for q,r in pairs for s in ph[r['alias']]['segments'] if s['role']=='core_candidate']
check(len(pairs)==36 and len(opportunities)==len(set(opportunities))==117,'fixed pairs/opportunities')
roles=Counter(x[3] for x in opportunities);check(roles['fit']==78 and sum(roles.values())-roles['fit']==39,'fixed fit/dev')
protected=read(O/'word100_semantic_episode_pilot_20261031/FROZEN.json')['protected']
check(len(protected)==32,'32 protected files')
for p,h in protected.items():check(sha(p)==h,'protected sha '+p)
for r in rows:check(sha(r['input'])==r['input_sha256'],'public input sha '+r['alias'])
code=(U/'evaluate.py').read_text(encoding='utf-8');tree=ast.parse(code)
imports=[x.module for x in ast.walk(tree) if isinstance(x,ast.ImportFrom)]+[a.name for x in ast.walk(tree) if isinstance(x,ast.Import) for a in x.names]
check(not any(any(z in (i or '').lower() for z in ('torch','numba','match','mediapipe')) for i in imports),'no model/matcher imported')
check("m['saved_window_center']=m['center']" in code and "if p[1]==refcenter" in code,'saved vs new representative retained')
check("m['sign_correct']=None;m['sign_error']=None" in code,'educational verdicts null')
check(sum(c['native_frame_count'] for c in dev['clips'])==624 and len(dev['viewed_evidence']['actual_viewed_images'])==30,'DEV review inventory')
out=dict(status='PASS_LABEL_CONTRACT',reviewer='same Codex A checking root merge independently; A also authored DEV physical draft, not independent human annotation',checks=checks,
 pair_prediction_files_read=0,new_searches=0,new_model_inference=0,denominators=dict(pairs=36,total=117,fit=78,dev=39),source_core_anchors_unchanged=39,
 protected_files_sha_verified=32,public_input_files_sha_verified=16,clips=clipstats,
 representative_contract='Path methods use lower median of unique saved query rows linked to the fixed original reference index center. Existing saved window/core center is retained separately. Mean posture has no path and keeps its original stored center.',
 caveats=['observed_physical_core is a schema name for certain physical interiors; DEV includes rest/preparation/return interiors too. It does not mean linguistic core.',
 '624 DEV native RGB frames were reviewed in contact sheets; this is not dense finger/3D/contact or linguistic ground truth.',
 'All117 old reference-anchor opportunities are retained even if the new point label is reference-uncertain, boundary, unreviewed, or absent.',
 'Historical development exposure and same-Codex annotation remain limitations; no independent human or strict blind claim.',
 'This pre-execution check has not inspected or validated actual saved pair results; that is a separate post-freeze audit.'],
 hashes={n:sha(U/n) for n in ['PHYSICAL_CORRESPONDENCE.json','merge_labels.py','evaluate.py','review/FIT_PHYSICAL_STAGES.json','review/DEV_PHYSICAL_STAGES.json']})
(U/'review/LABEL_CONTRACT_CHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(out['status'],checks,'checks; no pair predictions read')
