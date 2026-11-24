"""Record six actually re-viewed public FIT sheets; no unseen frame or semantic-completion labels."""
from pathlib import Path
import sys,hashlib,itertools,collections
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 assert not (W/'REPEAT_EVIDENCE_EXPANSION.json').exists()
 p=O/'word100_autonomous_localization_20261111/FIT_LABELS.json';old=read(p);rows=[];pairs=[]
 ranges={('WORD0003',1):(21,71,2),('WORD1290',1):(37,89,2),('WORD0003',3):(0,120,3),('WORD1290',3):(54,81,1),('WORD0003',5):(29,54,1),('WORD1290',5):(43,64,1)}
 for r in old['rows']:
  word=r['clip_id'].split('_')[2];key=(word,r['person'])
  if key not in ranges:continue
  assert r['role'] in ['fit','fit_additional'] and r['moving']=='R' and sha(r['sheet'])==r['sheet_sha256'];lo,hi,step=ranges[key];frames=list(range(lo,hi+1,step));points=[]
  for i,e in enumerate(r['events']):
   for f in frames:
    if e['onset'][1]<f<e['offset'][0]:points.append((f,i))
  for a,b in itertools.combinations(points,2):
   pairs.append(dict(clip_id=r['clip_id'],word_id=word,person=r['person'],frames=[a[0],b[0]],events=[a[1],b[1]],label=0 if a[1]==b[1] else 1,split='train',source_sha256=r['source_sha256']))
  rows.append(dict(clip_id=r['clip_id'],word_id=word,person=r['person'],source_sha256=r['source_sha256'],sheet=r['sheet'],sheet_sha256=r['sheet_sha256'],viewed_frames=frames,rechecked=True,events=r['events'],group_relation='Listed repeated submovements belong to the same observed alternating-hand-height sequence; this does not establish complete sign extent or required count.',onset_offset_changed=False,whole_sign_complete=None,required_repetitions=None,limits='Other small movements outside listed events remain unknown. No forced contact/handshape/parent-end labels. Coarse bands with no strict-inner viewed point produce no pair.' ))
 assert len(rows)==6
 before=read(W/'COVERAGE_AUDIT.json');allpairs=before['pairs']+pairs;counts={}
 for split in ['train','probe']:
  counts[split]={}
  for label in range(3):
   rr=[r for r in allpairs if r['split']==split and r['label']==label];counts[split][str(label)]=dict(pairs=len(rr),words=sorted({r['word_id'] for r in rr}),clips=len({r['clip_id'] for r in rr}))
 write(W/'REPEAT_EVIDENCE_EXPANSION.json',dict(at=stamp(),rows=rows,new_pairs=pairs,counts_after=counts,old_source=str(p),old_source_sha256=sha(p),status='REPEAT_DIVERSITY_EXPANDED_DISTINCT_MAIN_ACTIONS_STILL_LIMITED',reviewer='root Codex actually re-viewed6 RGB contact sheets this turn; no independent human/KSL gold',training_allowed=False,next='Do not train the confounded3-class model yet. Distinct main-action pairs still come only from WORD0739. Select additional already-owned FIT multi-action guides for source review; keep prep/return distinction separately labeled rather than disguising it as main-action diversity.'))
 print('Re-viewed6 FIT sources; added',len(pairs),'sample pairs.',counts)

if __name__=='__main__':main()
