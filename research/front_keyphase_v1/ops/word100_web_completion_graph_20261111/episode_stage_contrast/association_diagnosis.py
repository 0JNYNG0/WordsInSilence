"""Diagnostic oracle only: do stored proposals contain separable witnessed actions?

Source labels are used only to inspect proposal availability. Never deploy this
oracle or count it as learned model accuracy. No new selection rule is applied.
"""
from pathlib import Path
import sys,itertools
W=Path(__file__).resolve().parent;U=W.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
def main():
 out=W/'ASSOCIATION_DIAGNOSIS.json';assert not out.exists();saved=read(W/'BINDING_V2_SCORES.json')['rows'];pairs={};records=[]
 for r in saved:
  if r['seed']==42 and 'truth' in r:pairs.setdefault((r['clip_id'],r['kind']),{})[r['family']]=r
 for (clip,kind),groups in pairs.items():
  good={}
  for fam,r in groups.items():
   w=set(r['truth']['witness']);other=set(r['truth']['wrong_witness'])
   good[fam]=[c for c in r['candidates'] if w<=set(c['body_observed']) and not (other&set(c['body_observed']))]
  choices=[]
  for a,b in itertools.product(good[2],good[3]):
   if not set(a['body_observed'])&set(b['body_observed']):choices.append(dict(B=a['span'],C=b['span'],B_source=a['source'],C_source=b['source']))
  chosen={k:set(r['chosen']['body_observed']) if r['chosen'] else set() for k,r in groups.items()}
  records.append(dict(clip_id=clip,kind=kind,pure_witness_proposals={str(k):len(v) for k,v in good.items()},separable_candidate_pair_exists=bool(choices),examples=choices[:3],independent_selected_overlap=len(chosen[2]&chosen[3]),semantic_correctness=None))
 write(out,dict(at=stamp(),rows=records,clips=3,conditions=2,separable_pairs=sum(r['separable_candidate_pair_exists'] for r in records),comparison_pairs=len(records),scope='Source-truth oracle proposal availability only, not model recognition or feedback accuracy. Uses known DEV diagnostic labels and is forbidden in runtime selection. No rule forcing all action intervals disjoint is introduced: concurrent body/hand features and parent-child actions remain valid.',conclusion='Existing candidates already contain separate witnessed B/C spans in the inspected cases. The remaining selector can reuse overlapping evidence. Next review needs descriptor-level same/distinct/unknown occurrence supervision and a null option, without desired-order filtering or a universal non-overlap rule.',training=False,inference=False,web_changed=False))
 print('Oracle proposal availability',sum(r['separable_candidate_pair_exists'] for r in records),'/',len(records),'pairs; NOT learned accuracy.')
if __name__=='__main__':main()
