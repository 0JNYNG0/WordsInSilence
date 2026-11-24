"""Separate implementation of manifest, consensus, support and preservation checks."""
import json,hashlib,collections,re
from pathlib import Path
from PIL import Image
U=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
F=read(U/'FROZEN.json'); B=read(U/'RUBRIC.json'); I=read(U/'visual_inputs/INDEX.json')
R=read(U/'RESULT.json'); D=read(U/'DRAFT_LABELS.json'); M=read(U/'LEARNING_MANIFEST.json')
root=read(U/'ROOT_VISUAL_LABELS_RAW.json')['rows_by_alias']
secondary={};secondseq={}
for p in [U/'prior_labels/IMAGE_LABELS.json',U/'review/IMAGE_LABELS_FIRST_PASS.json']:
 x=read(p)
 assert x['rubric_sha256']==sha(U/'RUBRIC.json')
 for q in x['rows']:
  key=(q['alias'],q['ordinal']);assert key not in secondary;secondary[key]=q
 for q in x['sequences']:assert q['alias'] not in secondseq;secondseq[q['alias']]=q
features=list(B['features']); ords=B['static_sample_ordinals']
ir={x['alias']:x for x in I['rows']};fr={x['alias']:x for x in F['rows']}
checks={};errors=[]
def check(name,condition):
 checks[name]=bool(condition)
 if not condition:errors.append(name)
check('all_36_alias_sets',set(fr)==set(ir)==set(root)==set(secondseq) and len(fr)==36)
check('all_108_fixed_sample_keys',set(secondary)=={(a,n)for a in fr for n in ords} and len(secondary)==108)
check('frozen_split_16_8_8_4',dict(collections.Counter(x['role']for x in fr.values()))=={'fit_candidate':16,'held_person':8,'held_expression_seen_person':8,'held_expression_new_person':4})
check('frozen_unique_content36',len({x['source']['sha256']for x in fr.values()})==36)
fit={x['word_id']for x in fr.values()if x['role']=='fit_candidate'}
held={x['word_id']for x in fr.values()if x['role'].startswith('held_expression')}
check('held_expression_disjoint',not fit.intersection(held) and len(fit)==8 and len(held)==4)
check('recorded_ID_split',all(x['recorded_person_id']in([1,2]if x['role']in('fit_candidate','held_expression_seen_person')else[3])for x in fr.values()))
meta=['clip_id','word_id','label','recorded_person_id','role']
check('index_matches_frozen_metadata',all(all(ir[a][k]==fr[a][k]for k in meta)for a in fr))
check('indexed_source_content_matches_frozen',all(ir[a]['source_sha256']==fr[a]['source']['sha256']for a in fr))
check('all16_unique_monotonic_samples',all(len(x['frames'])==16 and len({t['frame_id']for t in x['frames']})==16 and [t['ordinal']for t in x['frames']]==list(range(1,17)) and all(p['pts_us']<q['pts_us']for p,q in zip(x['frames'],x['frames'][1:]))for x in ir.values()))
check('PTS_matches_saved_observation_within1us',all(abs(t['pts_us']-t['observation_timestamp_us'])<=1 for x in ir.values()for t in x['frames']))
check('nearest_target_error_at_most20ms',all(abs(t['pts_us']-t['target_us'])<=20000 for x in ir.values()for t in x['frames']))
imagebad=[];imagecount=0
for x in ir.values():
 for t in x['frames']:
  imagecount+=1
  with Image.open(t['file']) as im:
   if hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()!=t['rgb_sha256']:imagebad.append(t['file'])
check('576_saved_RGB_pixel_hashes',imagecount==576 and not imagebad)

def consensus(a,b):
 if a is None and b is None:return None,'both_unknown'
 if a is None or b is None:return None,'one_unknown'
 if a==b:return a,'both_visible_agree'
 return None,'visible_disagreement'
staticstatus=collections.Counter();seqstatus=collections.Counter(); rebuilt=[];rebuiltseq=[]
saved={(x['alias'],x['ordinal']):x for x in D['rows']}
seqsaved={x['alias']:x for x in D['sequences']}
mm={(x['alias'],x['ordinal']):x for x in M['static']}
staticbad=[];seqbad=[];manifestbad=[]
for a in sorted(ir):
 frames={t['ordinal']:t for t in ir[a]['frames']}
 for n in ords:
  out={k:fr[a][k]for k in meta};out.update(alias=a,ordinal=n,labels={})
  z=saved[a,n]
  for k in meta:
   if z[k]!=out[k]:staticbad.append([a,n,'metadata',k])
  for key in ['frame_id','pts_us']:
   if z[key]!=frames[n][key]:staticbad.append([a,n,'frame',key])
  if z['image']!=frames[n]['file']:staticbad.append([a,n,'image'])
  for j,k in enumerate(features):
   pv=root[a]['values'][ords.index(n)][j];sv=secondary[a,n][k]
   value,status=consensus(pv,sv);out['labels'][k]=value;staticstatus[status]+=1
   if (z['labels'][k],z['status'][k],z['primary'][k],z['secondary'][k])!=(value,status,pv,sv):staticbad.append([a,n,k])
  if mm[a,n]['labels']!=out['labels'] or any(mm[a,n][k]!=out[k]for k in meta if k!='label') or mm[a,n]['pts_us']!=frames[n]['pts_us']:manifestbad.append([a,n])
  rebuilt.append(out)
 out={k:fr[a][k]for k in meta};out.update(alias=a,labels={})
 for j,side in enumerate(['R','L']):
  value,status=consensus(root[a]['seq'][j],secondseq[a][side]);seqstatus[status]+=1;out['labels'][side]=value
  if seqsaved[a]['labels'][side]!=value or seqsaved[a]['status'][side]!=status:seqbad.append([a,side])
 rebuiltseq.append(out)
check('static_consensus_independently_rebuilt',not staticbad and len(saved)==108)
check('sequence_consensus_independently_rebuilt',not seqbad and len(seqsaved)==36)
check('learning_manifest_exact_consensus_and_times',not manifestbad and len(mm)==108)
check('review_status_totals',dict(staticstatus)==R['static_review_status'] and dict(seqstatus)==R['sequence_review_status'])
check('known_unknown_denominators',R['counts']['known_static_consensus']==staticstatus['both_visible_agree']==653 and R['counts']['unknown_static_consensus']==103 and sum(staticstatus.values())==756 and R['counts']['known_sequence_consensus']==seqstatus['both_visible_agree']==57 and R['counts']['unknown_sequence_consensus']==15 and sum(seqstatus.values())==72)
def support(rows,feature):
 out={}
 for value,name in[(1,'present'),(0,'absent'),(None,'unknown')]:
  selected=[q for q in rows if q['labels'][feature]==value]
  out[name]={'cells':len(selected),'clips':len({q['clip_id']for q in selected}),'expressions':sorted({q['word_id']for q in selected}),'recorded_people':sorted({q['recorded_person_id']for q in selected})}
 return out
roles=['fit_candidate','held_person','held_expression_seen_person','held_expression_new_person']
coveragebad=[];passes=[]
for cov in R['coverage']:
 k=cov['feature'];allc=support(rebuilt,k);by={r:support([q for q in rebuilt if q['role']==r],k)for r in roles}
 test=support([q for q in rebuilt if q['role'].startswith('held_expression')],k)
 gate=[len(by['fit_candidate']['present']['expressions'])>=2,len(by['fit_candidate']['absent']['expressions'])>=2,test['present']['cells']>0,test['absent']['cells']>0]
 if cov['all']!=allc or cov['by_role']!=by or list(cov['composition_checks'].values())!=gate or cov['composition_pass']!=all(gate) or cov['truth_validated'] is not False:coveragebad.append(k)
 if all(gate):passes.append(k)
seqcovbad=[]
for side in ['R','L']:
 val={'all':support(rebuiltseq,side),'by_role':{r:support([q for q in rebuiltseq if q['role']==r],side)for r in roles}}
 if val!=R['sequence_coverage'][side]:seqcovbad.append(side)
check('all_static_feature_support_counts',not coveragebad and len(passes)==5==R['counts']['composition_pass_features'])
check('sequence_support_counts',not seqcovbad)
check('no_training_or_truth_claim',R['counts']['training_runs']==0 and R['counts']['human_verified_labels']==0 and M['training_authorized_by_this_file'] is False and M['status']=='PREPARED_NOT_TRAINING_READY')
check('no_unknown_filled_by_source',all(z['labels'][k]is None for z in D['rows']for k in features if z['primary'][k]is None or z['secondary'][k]is None or z['primary'][k]!=z['secondary'][k]))
pins={**F['source_pins'],**R['pins']}
for x in F['rows']:pins[x['record']]=x['record_sha256']
protection=read(U.parent/'word100_face_relation_alignment_20261027/FROZEN.json')['protected']
pinbad=[p for p,h in pins.items()if sha(p)!=h]
protectedbad=[p for p,h in protection.items()if sha(p)!=h]
check('all_saved_source_record_result_pins_unchanged',not pinbad)
check('32_operating_model_DTW_web_pins_unchanged',len(protection)==32 and not protectedbad)
out={'status':'PASS_WITH_SCOPE_LIMITS'if not errors else'ISSUE','checks':checks,'errors':errors,'separate_implementation':True,'independent_human_truth':False,
 'checked':{'clips':36,'RGB_decoded_pixel_hashes':imagecount,'static_cells':756,'sequence_hand_cells':72,'source_record_result_pins':len(pins),'operating_pins':len(protection)},
 'recomputed_static_status':dict(staticstatus),'recomputed_sequence_status':dict(seqstatus),'composition_pass_features':passes,
 'not_composition_ready':['L_index_only has no fit positive expression','L_face_overlap has fewer than two fit positive expressions','L open/grouped sequence also has zero fit positives; it has no readiness claim'],
 'observations':['Independent X-only image draft was frozen before reading root labels. Alias word IDs were previously visible; not fully blind.','384 image panels and nine full fixed-sample PNGs were directly inspected by this reviewer. Old12 are delegated to the separate prior-label reviewer; no new independent visual review of those12 is claimed here.','Only one X static case had opposite known values: X21 ordinal5 L_face_overlap. Consensus properly leaves it unknown. Other X disagreements involve at least one unknown.','Lowered hands, edge-on finger states and wrist-versus-forearm boundaries cause most uncertainty. Unknown retention is appropriate; it does not demonstrate improved recognition.','32 model/web/DTW hashes match the prior protection manifest. No model inference, training or user-record reading was performed by this review.'],
 'limitations':['No actual new model accuracy or educational correctness has been established.','Recorded ID3 is held out for this new feature split only; old C/T3-F had prior exposure.','Five composition checks passing does not mean five validated feature predictors.','Sequence zeros concern sampled observations only, and must never label unseen frames or semantic absence.'],
 'reviewer_repair':'First checker mistakenly compared RGB-pixel hashes to compressed PNG-file hashes. Preserved that attempt and corrected to decoded RGB bytes; no source or project artifact changed.',
 'result_sha256':sha(U/'RESULT.json'),'draft_labels_sha256':sha(U/'DRAFT_LABELS.json'),'manifest_sha256':sha(U/'LEARNING_MANIFEST.json')}
(U/'review/FINAL_REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'checks':checks,'checked':out['checked'],'passfeatures':passes},ensure_ascii=True,indent=2))
