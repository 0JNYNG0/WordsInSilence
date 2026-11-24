"""Independent metadata and temporal support audit; no inference or training."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,sys
sys.stdout.reconfigure(encoding='utf8')
U=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=[U/'research/DEEPHAND_SIGNER_JOIN.json',U/'data/DEEPHAND_INVENTORY.json',U/'research/phoenix_metadata/phoenix2014-groundtruth-dev.stm']
pins={str(p):sha(p) for p in paths}
j=read(paths[0]);inventory=read(paths[1]);rows=j['rows']
stm=defaultdict(set)
for line in paths[2].read_text(encoding='utf8').splitlines():
    if not line.strip() or line.startswith(';;'):continue
    c=line.split();assert len(c)>=3
    stm[c[0]].add(c[2])
assert all(len(v)==1 for v in stm.values()) and len(stm)==j['official_stm_clips']==540
assert pins[str(paths[2])]==j['stm_sha256'] and pins[str(paths[1])]==j['inventory_sha256']
inv={r['id']:r for r in inventory['rows']};assert len(inv)==len(rows)==3359
def split(s):return 'fit' if int(s[-2:])<=5 else 'val' if int(s[-2:])<=7 else 'test'
by=defaultdict(dict);hashroles=defaultdict(set);broadcastroles=defaultdict(set)
for r in rows:
    i=inv[r['image_id']]
    assert r['recorded_signer_id'] in stm[r['clip']]
    assert all(r[k]==i[k] for k in ['clip','label','original_frame_index'])
    assert sha(Path(i['image']))==i['image_sha256']
    assert r['original_frame_index'] not in by[r['clip']]
    by[r['clip']][r['original_frame_index']]=r
    hashroles[i['pixel_sha256']].add(split(r['recorded_signer_id']))
    broadcastroles[i['broadcast']].add(split(r['recorded_signer_id']))
assert all(len(v)==1 for v in hashroles.values())
stats={s:Counter() for s in ['fit','val','test']};changeclips={s:set() for s in stats};tripclips={s:set() for s in stats}
patterns={s:Counter() for s in stats};classes={s:Counter() for s in stats};signers={s:set() for s in stats}
pairkeys=set();tripkeys=set();transitions=[];changed_triplets=[];pulses=[]
for clip,d in by.items():
    assert len({r['recorded_signer_id'] for r in d.values()})==1
    for i,r in d.items():
        s=split(r['recorded_signer_id']);z=stats[s];z['centers']+=1;classes[s][r['label']]+=1;signers[s].add(r['recorded_signer_id'])
        occ=tuple(int(i+k in d) for k in [-2,-1,0,1,2]);patterns[s][''.join(map(str,occ))]+=1
        z['available_context_slots']+=sum(occ);z['missing_context_slots']+=5-sum(occ)
        z['full_5frame_windows']+=int(all(occ));z['center_only_windows']+=int(sum(occ)==1)
        z['has_past_and_future']+=int(any(occ[:2]) and any(occ[3:]));z['has_adjacent_past_and_future']+=int(occ[1] and occ[3])
        z['window_any_class_difference']+=int(any(d[i+k]['label']!=r['label'] for k in [-2,-1,1,2] if i+k in d))
        if i+1 in d:
            q=d[i+1];ch=r['label']!=q['label'];z['consecutive_pairs']+=1;z['changed_pairs']+=ch
            pairkeys.add((clip,i,i+1,ch,r['label'],q['label']))
            if ch:
                changeclips[s].add(clip)
                transitions.append(dict(split=s,clip=clip,signer=r['recorded_signer_id'],from_frame=i,to_frame=i+1,labels=[r['label'],q['label']]))
        if i-1 in d and i+1 in d:
            labs=(d[i-1]['label'],r['label'],d[i+1]['label']);ch=len(set(labs))>1
            z['consecutive_triplets']+=1;z['changed_triplets']+=ch
            pulse=labs[0]==labs[2] and labs[1]!=labs[0];z['pulse_triplets']+=int(pulse)
            tripkeys.add((clip,i,labs))
            if ch:
                tripclips[s].add(clip)
                changed_triplets.append(dict(split=s,clip=clip,center=i,labels=list(labs)))
            if pulse:pulses.append(dict(split=s,clip=clip,center=i,labels=list(labs)))
for s,z in stats.items():
    z['class_count']=len(classes[s]);z['signer_count']=len(signers[s]);z['changed_pair_clips']=len(changeclips[s]);z['changed_triplet_clips']=len(tripclips[s])
    z['clip_count']=sum(split(next(iter(d.values()))['recorded_signer_id'])==s for d in by.values())
assert pairkeys=={(p['clip'],p['from'],p['to'],p['label_change'],*p['labels']) for p in j['adjacent_labelled_pairs']}
assert tripkeys=={(p['clip'],p['center'],tuple(p['labels'])) for p in j['consecutive_labelled_triplets']}
assert len(pairkeys)==j['adjacent_labelled_pair_count']==2049
assert len(transitions)==j['label_change_pair_count']==38
assert len(tripkeys)==j['triplet_count']==1449
assert len(changed_triplets)==j['triplet_with_different_label_count']==26
assert all(sha(Path(p))==h for p,h in pins.items())
result={
 'status':'PASS_METADATA_AND_COUNT_RECONSTRUCTION_CONDITIONAL_FOURWAY_FEASIBILITY',
 'review_scope':'Independent reconstruction from official STM, saved manual annotation inventory and frame IDs; not model evaluation or hidden-hand ground truth.',
 'source_pins':pins,'source_images_file_hash_checked':3359,'official_clip_signer_joined':3359,
 'source_timing':{'fps':25,'relative_step_us':40000,'offsets_us':[-80000,-40000,0,40000,80000],
  'provenance':'https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/',
  'time_type':'derived from documented fps and original integer frame indices, not newly decoded original video PTS'},
 'fixed_split':{s:sorted(v) for s,v in signers.items()},'stats_by_split':{s:dict(z) for s,z in stats.items()},
 'all':{'centers':3359,'original_classes':45,'clips':443,'consecutive_pairs':2049,'changed_pairs':38,'consecutive_triplets':1449,'changed_triplets':26,'pulse_triplets':3,'full5_windows':846,'center_only_windows':580},
 'class_counts_by_split':{s:dict(v) for s,v in classes.items()},
 'validation_or_test_classes_absent_in_fit':{s:sorted(set(classes[s])-set(classes['fit'])) for s in ['val','test']},
 'context_availability_patterns':{s:dict(v) for s,v in patterns.items()},
 'cross_split_exact_pixel_hash_conflicts':0,
 'broadcast_groups_crossing_splits':[k for k,v in broadcastroles.items() if len(v)>1],
 'changed_pairs':transitions,'changed_triplets':changed_triplets,'pulse_triplets':pulses,
 'fourway_allowed_claim':'Compare manual center-frame 45-class physical handshape classification on signer-separated retained crops, using actually available same-clip frames within +/-2 and an explicit presence mask.',
 'required_protocol_checks':[
  'Keep each original manual class target only on its labelled source frame; no propagated targets on absent slots.',
  'All four conditions use identical3359-center universe/split and report unavailable-pose/input failures in fixed denominators.',
  'Present neighbors are matched by exact same-clip frameID offsets; never nearest retained crop, time compression, still-image copying or crossing clips.',
  'Missing crop availability mask differs from missing joint detector mask and both differ from unknown handshape truth.',
  'The temporal model receives RGB/joints/masks/times, never neighboring ground-truth class labels.',
  'Use original45 source classes; no implicit binary/fist label merge.',
  'Fit/calibrate/select only on fit/val; score test once after protocol and model selection frozen.',
  'Interpret25fps-derived offsets as relative frame time, not verified source video PTS.',
  'Maintain source crop-transform provenance; this crop-only study does not validate original-frame hand/body motion or anatomical left/right identity.',
  'Report accuracy by context availability and class; test supports28 of45 classes, so no all45-class test-generalization claim.',
  'Report unique adjacent pairs/unique source clips, not repeated overlapping-window counts as independent transitions.',
  'Add a prespecified fit-only majority and mask-pattern baseline to diagnose annotation-selection information; do not tune masks using test outcomes.'
 ],
 'transition_metrics':{
  'eligible_test_pairs':226,'test_changed_pairs':6,'test_changed_pair_clips':5,'test_stable_pairs':220,
  'eligible_test_triplets':164,'test_changed_triplets':3,'test_changed_triplet_clips':2,'test_short_pulse_triplets':0,
  'report':['Correct center class on changed-pair endpoints versus stable frames','Both endpoint class labels correct on each changed pair','Predicted-change versus true-label-change with false alarms on220 stable pairs','All three labels correct on164 triplets, changed3 reported separately','Per-clip/person summaries and raw counts; no frame-independent confidence interval'],
  'not_supported':['Held-out A-B-A short-pulse preservation (zero test cases)','Exact phase transition boundaries','Full motion ordering across whole signs','Actual occlusion recovery accuracy','KSL tolerance or pedagogical correction accuracy']},
 'label_selection_caution':'Only handshape-labelled retained crops are available. Neighbor presence may correlate with annotation/class selection. Temporal gains can reflect this restricted sampling; masks-only and context-stratum diagnostics are needed.',
 'supersedes_previous_static_only_limitation':'Official signer linkage and exact original consecutive frame IDs now support a limited temporal crop-context comparison. This does not establish dense complete video or independent KSL temporal truth.',
 'no_training_or_model_inference_by_reviewer':True,'no_user_mentor_access':True,'no_production_or_source_changes':True
}
p=U/'verification/TEMPORAL_AVAILABILITY_REVIEW.json';assert not p.exists()
p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:result[k] for k in ['status','stats_by_split','all','validation_or_test_classes_absent_in_fit','broadcast_groups_crossing_splits']},ensure_ascii=False,indent=2))
