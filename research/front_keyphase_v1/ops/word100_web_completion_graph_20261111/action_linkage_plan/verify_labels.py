"""Reconstruct relation labels independently; do not force atomic labels from parent membership."""
from pathlib import Path
import sys,itertools,collections,hashlib
W=Path(__file__).resolve().parent;U=W.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r):return (r['clip_id'],tuple(r['frames']),r['label'])

def main():
 assert not (W/'LABEL_VERIFICATION.json').exists()
 bundle=read(W/'RELATION_LABEL_BUNDLE.json');original=read(W/'COVERAGE_AUDIT.json');extra=read(W/'REPEAT_EVIDENCE_EXPANSION.json');distinct=read(W/'DISTINCT_ACTION_WITNESSES.json');dense=read(original['source']);expected=[]
 for p,h in bundle['source_files'].items():assert sha(p)==h
 for r in dense['rows']:
  seen={f for s in r['sheets'] for f in s['frames']};event_to_parent={e:g['group_id'] for g in r['groups'] for e in g['event_ids']};points={}
  for e in r['events']:
   for f in seen:
    if e['onset'][1]<f<e['offset'][0]:assert f not in points;points[f]=(e['event_id'],event_to_parent[e['event_id']])
  for a,b in itertools.combinations(sorted(points),2):
   aa,bb=points[a],points[b];expected.append((r['clip_id'],(a,b),0 if aa[0]==bb[0] else 1 if aa[1]==bb[1] else 2))
 assert collections.Counter(expected)==collections.Counter(key(r) for r in original['pairs'])
 expected=[]
 for r in extra['rows']:
  assert r['rechecked'] and sha(r['sheet'])==r['sheet_sha256'];points={}
  for i,e in enumerate(r['events']):
   for f in r['viewed_frames']:
    if e['onset'][1]<f<e['offset'][0]:assert f not in points;points[f]=i
  for a,b in itertools.combinations(sorted(points),2):expected.append((r['clip_id'],(a,b),0 if points[a]==points[b] else 1))
 assert collections.Counter(expected)==collections.Counter(key(r) for r in extra['new_pairs'])
 expected=[]
 for r in distinct['rows']:
  assert r['viewed'] and sha(r['source'])==r['source_sha256'] and sha(r['sheet'])==r['sheet_sha256'];a,b=[g['sample_frames'] for g in r['groups']];assert set(a+b)<=set(r['sampled_frames']);assert not set(a)&set(b)
  expected.extend((r['clip_id'],(f,h),2) for f in a for h in b)
 assert collections.Counter(expected)==collections.Counter(key(r) for r in distinct['pairs'])
 assert collections.Counter(key(r) for r in bundle['rows'])==collections.Counter(key(r) for r in original['pairs']+extra['new_pairs']+distinct['pairs'])
 rows=[]
 for r in bundle['rows']:
  rows.append(dict(r,parent_same=int(r['label'] in [0,1]),atomic_same=int(r['label']==0),basis='reviewed_atomic_event_and_parent_hierarchy' if r['label']!=2 else 'reviewed_distinct_parent_groups'))
 # Same parent does not imply same atomic movement. Keep atomic target absent.
 for r in distinct['rows']:
  for g in r['groups']:
   for a,b in itertools.combinations(g['sample_frames'],2):rows.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],source_sha256=r['source_sha256'],frames=[a,b],label=None,parent_same=1,atomic_same=None,split='probe' if r['person'] in [5,8,9] else 'train',basis='reviewed_same_parent_only_atomic_unknown'))
 assert len(rows)==len(bundle['rows'])+36
 for r in rows:assert r['split']==('probe' if r['person'] in [5,8,9] else 'train')
 tr={r['source_sha256'] for r in rows if r['split']=='train'};pr={r['source_sha256'] for r in rows if r['split']=='probe'};assert not tr&pr
 counts={}
 for split in ['train','probe']:
  rr=[r for r in rows if r['split']==split];counts[split]=dict(pairs=len(rr),parent=dict(collections.Counter(r['parent_same'] for r in rr)),atomic={str(k):v for k,v in collections.Counter(r['atomic_same'] for r in rr).items()},source_clips=len({r['clip_id'] for r in rr}))
 write(W/'MULTILEVEL_RELATION_TARGETS.json',dict(at=stamp(),rows=rows,counts=counts,parent_target='same observed action group / different group',atomic_target='same atomic movement / different movement / unknown',unknown_atomic_masked=True,word_id_metadata_only=True,training_allowed=False,next_gate='Prepare matching body-normalized feature pairs with actual observation masks. Balance source/group contributions, compare word-only and time-gap-only diagnostics, freeze small pair-head protocol before training. Exact boundary/completion targets still require separate source evidence.',upstream_exposure='IDs05/08/09 excluded only from future pair-head fit; upstream shared/stage models have seen some/all of them. Not a new-person benchmark.'))
 protected=read(U/'PROTECTIONS.json');assert all(sha(p)==h for p,h in protected.items())
 report=dict(at=stamp(),status='PASS_SOURCE_RECONSTRUCTION_PARTIAL_TRAINING_READINESS',initial_three_class_pairs=len(bundle['rows']),multilevel_pairs=len(rows),unknown_atomic_targets=36,counts=counts,protected_hashes=len(protected),source_sha_disjoint=True,model_training=False,web_changed=False,reviewer='Same root Codex second reconstruction, not independent human/KSL ground truth',limits=['Basic class/word diversity is improved, not complete dataset adequacy. Pair observations within videos are correlated.','No automatic assignment to an exact full interval, no required repetition count, no educational completion target.','Same-parent examples must not be silently converted to same atomic event.','Left/right feature correspondence and simultaneous-feature labels still need explicit observation-based coverage before integration.'])
 write(W/'LABEL_VERIFICATION.json',report);print('Reconstructed',len(bundle['rows']),'existing pairs; multilevel targets',len(rows),'including36 masked atomic labels.',counts)

if __name__=='__main__':main()
