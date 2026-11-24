"""Record the current RGB review; old annotations are never overwritten."""
from pathlib import Path
import json, hashlib, collections
U=Path(__file__).resolve().parent; OPS=U.parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x): (U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
F=read(U/'FROZEN.json')['features']
draft=read(OPS/'word100_shared_feature_bundle_20261027/DRAFT_LABELS.json')
seqs=read(OPS/'word100_shape_sequence_bundle_20261027/BUNDLE.json')['sequences']
# All 24 contact sheets were visually inspected in this turn. Ambiguities are
# masked, not relabeled from temporal context or word meaning.
uncertain={
 ('V11',11,'R_open_four'):'Intermediate curved fingers; four extended fingers not securely visible.',
 ('X05',5,'R_face_overlap'):'Projected face/hand boundary is borderline at this view.',
 ('X18',11,'hands_overlap'):'Hand-versus-wrist boundary is ambiguous; forearm overlap is not necessarily hand overlap.'}
for a in ['X17','X18']:
 for n in [5,8,11]: uncertain[a,n,'R_open_four']='Edge-on four-finger separation not securely visible; retain position evidence.'
for a,n in [('X22',5),('X23',5),('X24',5),('X24',8)]:
 for k in ['R_open_four','L_open_four']: uncertain[a,n,k]='Oblique/foreshortened triangular presentation; individual four-finger extension uncertain.'
rows=[];changes=[];known=0
for r in draft['rows']:
 y={k:r['labels'][k] for k in F}; known+=sum(v is not None for v in y.values())
 for k in F:
  reason=uncertain.get((r['alias'],r['ordinal'],k))
  if reason and y[k] is not None:
   changes.append(dict(alias=r['alias'],frame_id=r['frame_id'],feature=k,old=y[k],new=None,reason=reason));y[k]=None
 rows.append({**{k:r[k] for k in ['alias','clip_id','word_id','label','recorded_person_id','role','ordinal','frame_id','pts_us']},'labels':y})
notes={
 'N01-N03':'Index extension maintained during alternating hand motion; overlap does not force identity change.',
 'N04-N06':'Index/open-hand changes inspected chronologically; N06 final original focus image additionally inspected.',
 'N07-N09':'Index shape maintained while hand moves from face toward other hand.',
 'N10-N12':'Flat-hand forearm motion; edge-on uncertain open-four fields remain unknown.',
 'N13-N15':'Thumb-up/folded non-thumb fingers during alternating vertical movement; overlap unknowns retained.',
 'N16-N18':'Oblique flat/triangular hand presentation; existing unknowns remain unknown.',
 'T01-T03':'Right index occluded/foreshortened against left hand; no inferred index label added.',
 'T04-T06':'Edge-on hand, later finger configuration changes; T06 final full focus image rechecked.',
 'T07-T09':'Resting left hand; unknown open-four stays unknown, not an error.',
 'T10-T12':'Open right hand maintained during vertical movement near left palm.',
 'T13-T15':'Opening/closing changes preserved; ambiguous intermediate curved fingers stay unknown.',
 'T16-T18':'Open hand to folded fingers: intermediate transition not filled as a stable shape.'}
# No clear opposite binary label found after review; uncertain labels above are
# downgraded. This is same-Codex reinspection, NOT independent expert validation.
support={}
for k in F:
 support[k]={str(v):sorted({r['word_id'] for r in rows if r['role']=='fit_candidate' and r['labels'][k]==v}) for v in [0,1]}
print('roles',collections.Counter(r['role'] for r in rows))
class_pass=all(len(support[k][str(v)])>=2 for k in F for v in [0,1])
save('LABEL_REVIEW.json',dict(status='PASS_PROVISIONAL' if class_pass else 'FAIL_CLASS_SUPPORT',
 reviewed_static_images=108,reviewed_short_windows=36,reviewed_chronological_frames=468,
 reviewed_static_known_cells=known,clear_binary_contradictions=0,uncertain_downgrades=changes,
 support=support,class_gate=class_pass,review_notes=notes,
 source='RGB-visible physical attributes; prior Codex labels rechecked by Codex, no independent human/expert truth',
 exclusions='Unknown never filled; 2D overlap not contact; no required repetition/semantic stage/learner-error labels',
 pins=read(U/'REVIEW_INDEX.json')['pins']))
save('LABELS.json',dict(static=rows,sequences=seqs,review_sha256=sha(U/'LABEL_REVIEW.json')))
print('class_pass',class_pass,'known',known,'downgrades',len(changes),'support',support)
