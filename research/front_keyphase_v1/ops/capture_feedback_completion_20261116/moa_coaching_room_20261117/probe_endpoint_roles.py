"""Public-only attribution: can hand-form change hide an observed arrival?"""
from pathlib import Path
import sys,json
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent
sys.path.insert(0,str(G/'rest_stability_endpoint_20261117'))
from endpoint_candidate import EndpointLesson, EndpointRelation, ShortArrivalLesson, np, torch, region_distance, approaches
from probe_variants import variants
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))

class RoleRelation(EndpointRelation):
 def __call__(self,f,quality,ts):
  original,_=self.inner(f,quality,ts);g=self.inspection(f);d=region_distance(self.feature.channel)
  close=quality&np.isfinite(d)&(g==2)&(d<=self.limit)
  paired=close&(np.r_[False,close[:-1]]|np.r_[close[1:],False])
  s=original.copy()
  # Candidate: geometry provides the interaction proposal; visible near evidence
  # and unchanged independent movement remain necessary for an arrival.
  s[(original==0)&quality&(g==1)]=1
  s[paired]=2;self.added=np.flatnonzero(s!=original).tolist()
  return s,approaches(s,ts)

class RoleLesson(EndpointLesson):
 def __init__(self):
  super().__init__();r=self.relation
  self.relation=RoleRelation(r.inner,r.inspection,r.feature,r.limit)

def main():
 torch.set_num_threads(2);a=ShortArrivalLesson();b=RoleLesson();rows=[]
 donor=next(r for r in read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows'] if r['clip_id']=='NIA_SL_WORD2633_REAL01_F')
 dr=dict(np.load(donor['raw']));i=donor['anchors'][len(donor['anchors'])//2];hand=dr['xy_px'][i,112:133];hand=(hand-hand[0])/np.linalg.norm(hand[9]-hand[0])
 for r in read(G/'frontal_scope_recovery_20261117/INPUTS.json')['rows']:
  if r['word']!='WORD1290' or r['person'] not in [1,2,3,11,12,13]:continue
  assert r['id'].endswith('_F') and 'user_' not in r['raw']
  raw=dict(np.load(r['raw']));q=a.analyze(raw)
  for kind,z in variants(raw,hand,q['events']):
   qa=a.analyze(z);qb=b.analyze(z);row=dict(id=r['id'],kind=kind,before=len(qa['events']),after=len(qb['events']),before_status={c['id']:c['status']for c in qa['criteria']},after_status={c['id']:c['status']for c in qb['criteria']},changed_relation=qb['endpoint_candidate_support']);rows.append(row)
   print(r['id'],kind,row['before'],row['after'],flush=True)
 (D/'ENDPOINT_ROLE_PROBE.json').write_text(json.dumps(dict(rows=rows,private_data=False,training=0,scope='Public frontal normal transforms and an explicitly synthetic hand-form substitution; not actual human errors or final accuracy'),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
