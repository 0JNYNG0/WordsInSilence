"""Source-level distinct-action witnesses from six RGB overviews actually inspected.

These are sampled group-membership labels, not exact motion boundaries, finger
truth, semantic completeness or a required repetition count.
"""
from pathlib import Path
import sys,hashlib,itertools,collections
W=Path(__file__).resolve().parent;U=W.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

S={
 ('WORD0185',1):([45,48,51],[66,69,72]),
 ('WORD0185',3):([69,72,75],[90,93,96]),
 ('WORD0185',5):([33,36,39],[51,54,57]),
 ('WORD0332',1):([33,36,39],[54,57,60]),
 ('WORD0332',3):([63,66,69],[84,87,90]),
 ('WORD0332',5):([57,60,63],[84,87,90])}

def main():
 assert not (W/'DISTINCT_ACTION_WITNESSES.json').exists();meta=read(W/'DISTINCT_SOURCE_INPUTS.json');rows=[];pairs=[]
 for r in meta['rows']:
  assert r['role']=='fit' and sha(r['source'])==r['source_sha256'] and sha(r['sheet'])==r['sheet_sha256'];a,b=S[r['word_id'],r['person']];assert set(a+b)<=set(r['sampled_frames']) and max(a)<min(b)
  descriptions=['오른손을 어깨 뒤쪽으로 향하게 하는 장면','오른손과 세운 왼손 사이의 접근·반복 장면'] if r['word_id']=='WORD0185' else ['얼굴 양옆에서 두 손을 흔드는 장면','손모양을 바꾼 뒤 두 손을 위로 올리는 장면']
  rows.append(dict(r,viewed=True,groups=[dict(id=r['clip_id']+':group'+str(i),sample_frames=f,description_ko=t) for i,(f,t) in enumerate(zip([a,b],descriptions))],exact_onset_offset=None,semantic_completion=None,required_repetitions=None,guide_match='Gross sequential movement relationship consistent with source description; full variant/finger/contact correctness NOT verified. In WORD0185 initial finger configuration cannot be equated to the dictionary open-hand wording.',reviewer='root Codex actual RGB overview review; no expert gold'))
  for f,h in itertools.product(a,b):pairs.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],source_sha256=r['source_sha256'],frames=[f,h],label=2,meaning='distinct observed main-action groups',split='unassigned_until_pair_protocol_frozen'))
 write(W/'DISTINCT_ACTION_WITNESSES.json',dict(at=stamp(),rows=rows,pairs=pairs,status='SAMPLED_DISTINCT_ACTION_RELATIONS_REVIEWED',training_allowed=False,limits=['Source first/second order is annotation only; selection must not force expected order.','Within-group atomic-motion identity remains unassigned; no same-atomic label invented.','Only listed viewed frames are targets. Exact action endpoints need dense source review.','Not proof of educational correctness or universal tolerance.']))
 previous=read(W/'REPEAT_EVIDENCE_EXPANSION.json');original=read(W/'COVERAGE_AUDIT.json');allpairs=original['pairs']+previous['new_pairs']+pairs
 # New pair head has never been trained. Split before any such scores; upstream
 # models have seen these public FIT people and must not be called fully unseen.
 for r in allpairs:r['split']='probe' if r['person'] in [5,8,9] else 'train'
 counts={}
 for split in ['train','probe']:
  counts[split]={}
  for c in range(3):
   rr=[r for r in allpairs if r['split']==split and r['label']==c];counts[split][str(c)]=dict(pairs=len(rr),words=sorted({r['word_id'] for r in rr}),people=sorted({r['person'] for r in rr}),clips=len({r['clip_id'] for r in rr}))
 train={r['source_sha256'] for r in allpairs if r['split']=='train'};probe={r['source_sha256'] for r in allpairs if r['split']=='probe'};assert not train&probe
 diversity=all(len(counts[s][str(c)]['words'])>=2 and len(counts[s][str(c)]['people'])>=2 for s in counts for c in range(3))
 write(W/'RELATION_LABEL_BUNDLE.json',dict(at=stamp(),rows=allpairs,counts=counts,source_files={str(p):sha(p) for p in [W/'COVERAGE_AUDIT.json',W/'REPEAT_EVIDENCE_EXPANSION.json',W/'DISTINCT_ACTION_WITNESSES.json']},pair_head_train_people=sorted({r['person'] for r in allpairs if r['split']=='train'}),pair_head_probe_people=[5,8,9],source_sha_disjoint=True,basic_class_word_diversity=diversity,training_allowed=False,next_gate='Independent label reconstruction; input/observation-mask mapping; within-source/group balanced sampling and time-gap-only baseline; freeze comparison before new pair-head training. No direct complete-span or semantic-completion target yet.',scope='Only new relation-head split. Upstream stage/shared100 previously saw these people. Pair counts correlated; no fresh-user or total feedback accuracy claim. Previous stage split is unchanged.'))
 print('Added54 distinct-group pairs from6 source clips. New pair-head split coverage:',counts,'basic diversity',diversity)

if __name__=='__main__':main()
