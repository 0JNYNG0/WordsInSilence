"""Public-only inventory and guide supervision audit; never reads user recordings."""
from pathlib import Path
import hashlib, json, re, sys
from datetime import datetime, timezone, timedelta
from collections import Counter
import numpy as np
sys.dont_write_bytecode = True
U=Path(__file__).resolve().parent; O=U.parent; R=O.parents[2]
OLD=O/'word100_multipeople_live_feedback_20261101'
WORDS=('WORD0739','WORD1133','WORD1567','WORD2523')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
 p=U/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 assert not (U/'PREPARED.json').exists()
 public=read(OLD/'PUBLIC_INPUTS.json'); bank=read(OLD/'REFERENCE_BANK.json')
 labels=read(OLD/'TRAIN_DATASET.json'); desc=read(O/'word100_description_evidence_feedback_20261101/REPORT_DATA.json')
 rows=public['rows']; assert len(rows)==48
 pins={str(OLD/n):sha(OLD/n) for n in ['PUBLIC_INPUTS.json','REFERENCE_BANK.json','TRAIN_DATASET.json']}
 pins[str(O/'word100_description_evidence_feedback_20261101/REPORT_DATA.json')]=sha(O/'word100_description_evidence_feedback_20261101/REPORT_DATA.json')
 prepared=[]; sums={}; hashes=set(); gids=[]
 for row in rows:
  cid=row['clip_id']; assert re.fullmatch(r'NIA_SL_WORD\d{4}_REAL\d{2}_F',cid)
  assert row['word_id'] in WORDS and row['person'] in list(range(1,10))+[11,12,13]
  assert row['role']==('fit' if row['person']<=9 else 'development')
  assert row['source_sha256'] not in hashes, 'Duplicate public video across split';hashes.add(row['source_sha256'])
  p=Path(row['input']); assert sha(p)==row['input_sha256'];pins[str(p)]=sha(p)
  a=np.load(p,allow_pickle=False); n=len(a['frame_id'])
  assert a['x'].shape==(n,69) and a['rgb'].shape==(n,768) and a['mask'].dtype==bool
  assert np.all(np.diff(a['frame_id'])>0) and np.all(np.diff(a['timestamp_us'])>0)
  assert np.isfinite(a['rgb']).all() and np.isfinite(a['x'][a['mask']]).all()
  lr=[l for l in labels['rows'] if l['clip_id']==cid]
  assert len(lr)==3, (cid,len(lr))
  for l in lr:
   assert l['role']==row['role']
   ix=np.flatnonzero(a['frame_id']==l['center_frame']);assert len(ix)==1
   assert a['timestamp_us'][ix[0]]==l['center_timestamp_us']
  br=next((b for b in bank['rows'] if b['clip_id']==cid),None)
  stages=[] if br is None else [{k:s.get(k) for k in ['canonical_stage_id','description','frames','canonical_core_frame_ids','source_cards','boundary_provenance','description_scope']} for s in br['stages']]
  if row['person']==1:
   for st in stages:
    gids.append(dict(word_id=row['word_id'],label=row['label'],**st,
       supervision='saved canonical visual span, not independent educational correctness'))
  nr=dict(row,labels=lr,stages=stages,
          stage_supervision='canonical_visual' if row['person']==1 else ('dtw_weak_estimate' if br else 'not_annotated'))
  prepared.append(nr)
  for l in lr:
   for j,t in enumerate(labels['targets']):
    k=(row['role'],t);v=sums.setdefault('|'.join(k),{'positive':0,'negative':0,'unknown':0})
    v['positive' if l['target_mask'][j] and l['y'][j]>0.5 else 'negative' if l['target_mask'][j] else 'unknown']+=1
 source_clauses=[]
 for c in desc['source_clause_catalog']:
  if c['word_id'] in WORDS:
   source_clauses.append({k:c.get(k) for k in ['id','word_id','source_span','source_url','source_entry_id','source','expected_feature_value','observable_components','unsupported_components','requiredness','tolerance']})
 save('INPUT_MANIFEST.json',dict(rows=prepared,targets=labels['targets'],feature_names=public['feature_names'],source_pins=pins,
   historical_exposure='All are existing research materials. IDs11-13 are held out from this fit, previously used DEV, not untouched test or independent human gold. Existing canonical and DTW-projected spans are separated.'))
 save('GUIDE_SUPPORT.json',dict(guides=gids,source_clauses=source_clauses,label_counts=sums,
  weak_spans_not_gold=True,new_expert_labels=0,full_independent_stage_test_available=False,
  note='Physical feature labels are prior visual review, not independent KSL educational error labels; preserve provenance.'))
 save('PREPARED.json',dict(status='PUBLIC_INPUTS_AND_SUPERVISION_INVENTORIED',clips=len(rows),fit_clips=36,development_clips=12,
  words=list(WORDS),known_center_labels=144,guide_stage_count=len(gids),source_clause_count=len(source_clauses),
  public_input_hashes_verified=True,duplicate_source_hashes=0,new_training=0,new_inference=0,
  next='Validate trainable pretrained projection and cached-feature identity; fix first comparison contract before training.'))
 print(json.dumps(read(U/'PREPARED.json'),ensure_ascii=False))
if __name__=='__main__':main()
