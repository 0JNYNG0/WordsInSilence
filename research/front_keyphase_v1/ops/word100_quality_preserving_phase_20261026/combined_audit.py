from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
from collections import Counter,defaultdict
sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;O=U.parent;E=O/'word100_phase_evidence_preserving_20261026';P=O/'word100_point_sequence_split_20261022'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 v=read(E/'VERIFIED.json');rows=v['rows'];normal=Counter();omissions=defaultdict(list);pins={};out=[]
 plan=dict(at=datetime.datetime.now().astimezone().isoformat(),rule='Diagnostic fixed OR: unchanged original instantaneous bad channel OR complete new sequence bad channel. No thresholds, reference minima, or scores changed. Assess normal added warning over own identity and short omission relative own identity by original >=5of8 references. EP primary failed gate remains unchanged. This audit cannot replace it retrospectively.',inputs=[str(E/'VERIFIED.json'),str(E/'REGRESSION_AUDIT.json')],no_training=True)
 with (U/'COMBINED_AUDIT_FROZEN.json').open('x',encoding='utf-8') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
 cache={}
 def get(folder,name):
  p=folder/(name+'.npz')
  if str(p) not in cache:cache[str(p)]=load(p);pins[str(p)]=sha(p)
  return cache[str(p)]
 for m in rows:
  if m['folder']!='cross' or m['kind']=='identity':continue
  name=m['name'];base='identity__'+m['query']+'__'+m['reference'];old=get(P/'cross',name);oi=get(P/'cross',base);new=get(E/'cases','cross__'+name);ni=get(E/'cases','cross__'+base)
  def flags(z):return z['point_bad']|z['component_bad'].any(-1)
  oldb=flags(old);oldib=flags(oi);newb=flags(new);newib=flags(ni)
  common_point=old['point_comparable']&oi['point_comparable'];common_seq=old['sequence_comparable']&oi['sequence_comparable'];common=old['expected']&(common_point|common_seq)
  if m['kind'] in ['fast075','slow15','pace','pause']:
   normal['pairs']+=1;normal['original_combined_bad']+=int((common&oldb).sum());normal['new_combined_bad']+=int((common&newb).sum());normal['original_incremental_bad']+=int((common&oldb&~oldib).sum());normal['new_incremental_bad']+=int((common&newb&~newib).sum())
  elif m['kind'].startswith('skip'):
   removed=np.setdiff1d(np.arange(len(oi['source_index'])),old['source_index'].astype(int));loc=np.zeros_like(common);refs=np.unique(oi['path'][np.isin(oi['path'][:,0],removed),1]);loc[refs]=old['expected'][refs]
   # Preserve original novelty separately per information type: point + sequence.
   op=loc&common_point&old['point_bad']&~oi['point_bad'];os=loc&common_seq&old['component_bad'].any(-1)&~oi['component_bad'].any(-1);ns=loc&common_seq&new['component_bad'].any(-1)&~ni['component_bad'].any(-1)
   omissions[m['query']+'__'+m['kind']].append((bool((op|os).any()),bool((op|ns).any())))
 counts=defaultdict(Counter)
 for key,pairs in omissions.items():
  assert len(pairs)==8;kind='short3' if '__skip3_' in key else 'long20pct';a=sum(p[0] for p in pairs);b=sum(p[1] for p in pairs);c=counts[kind];c['variants']+=1;c['old_detected']+=a>=5;c['new_detected']+=b>=5;c['gained']+=a<5 and b>=5;c['lost']+=a>=5 and b<5;out.append(dict(key=key,old_references=a,new_references=b))
 for p,h in pins.items():assert sha(Path(p))==h
 result=dict(at=datetime.datetime.now().astimezone().isoformat(),status='SAVED_COMPOSITE_DIAGNOSTIC_NOT_RETROACTIVE_ADOPTION',normal=dict(normal),omissions={k:dict(v) for k,v in counts.items()},variants=out,pins=pins,primary_ep_failure_preserved=True,semantic_ground_truth=False)
 with (U/'COMBINED_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 print('COMBINED_AUDIT',dict(normal),dict(counts),flush=True)
if __name__=='__main__':main()
