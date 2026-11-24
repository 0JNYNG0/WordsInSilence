from common import *
from criteria import variant,CONDITIONS
from evidence_geometry import compare_hand
import time,traceback

def main():
 assert not (U/'GUARDED_GEOMETRY.json').exists()
 scope=read(U/'SCOPE.json');check(scope['pins'])
 old=read(U/'GEOMETRY.json');problem=[r for r in old['rows'] if r['condition']=='other_signer_same_word' and r['view']=='masked' and r['metrics']['local']['ordered']==0]
 write(U/'FIRST_PROBE_FINDING.json',{'status':'MEASUREMENT_SPECIFICATION_LIMIT_PRESERVED','at':now(),'initial_probe_sha256':sha(U/'GEOMETRY.json'),
  'zero_masked_local_cross_signer_pairs':len(problem),'denominator':799,'example':problem[0],
  'cause':'Unrestricted nearest/monotone matches can use a single common local wrist anchor, always(0,0), for all frames. This is a flaw of the new exploratory distance, not evidence of a bug in C/T3F or production DTW.',
  'action':'Preserve first result. Add conservative complete-landmark same-side diagnostic; local wrists excluded by definition. Do not tune a distance cutoff.'})
 write(U/'EVIDENCE_GUARD_PROTOCOL.json',{'at':now(),'code_sha256':sha(__file__),'criteria_sha256':sha(U/'criteria.py'),'guard_sha256':sha(U/'evidence_geometry.py'),
  'scope_sha256':sha(U/'SCOPE.json'),'first_finding_sha256':sha(U/'FIRST_PROBE_FINDING.json'),
  'fixed_support':'Separate L/R; global all21same-side landmarks; local all20nonwristsame-side landmarks. No threshold search; any incomplete frame has unknown shape evidence, never filled0.',
  'retained':'Same3398pair/view conditions and all inputs. Coordinates, time, model, hand assignment unchanged.',
  'limitation':'Strict evidence requirement may abstain on usable partial hands. Both-fully-observed side still may rest; role and semantic phase labels remain unreviewed. Geometry scores are not pass/fail.'})
 started=time.monotonic();cache={};out=[]
 def load(cid,v):
  if (cid,v) not in cache:
   r=read(B/'data/records'/f'{cid}.json')
   with np.load(r['views'][v]['channel_file'],allow_pickle=False) as z:cache[cid,v]=z['x_whole']
  return cache[cid,v]
 for i,r in enumerate(old['rows']):
  x=load(r['reference'],r['view'])
  y=load(r['query'],r['view']) if r['condition']=='other_signer_same_word' else variant(x,r['condition'])[0]
  q={k:r[k] for k in ['reference','query','word_code','label','view','condition','semantic_label']}
  q['metrics']={f'{axis}_{side}':compare_hand(x,y,axis,si) for axis in ('global','local') for si,side in enumerate(('L','R'))}
  out.append(q)
  if (i+1)%600==0:print(f'Guarded geometry{i+1}/3398, {time.monotonic()-started:.1f}s',flush=True)
 check(scope['pins'])
 write(U/'GUARDED_GEOMETRY.json',{'status':'COMPLETE_DIAGNOSTIC_ONLY','protocol_sha256':sha(U/'EVIDENCE_GUARD_PROTOCOL.json'),'rows':out,'seconds':time.monotonic()-started,
  'new_model_inference':0,'training':0,'user_data_read':0,'production_changed':False})
 print('COMPLETE3398pair/views with4separatehand/axis metrics; first finding preserved.')
if __name__=='__main__':
 try:main()
 except BaseException:
  if not (U/'FIRST_GUARD_ERROR.json').exists():write(U/'FIRST_GUARD_ERROR.json',{'at':now(),'traceback':traceback.format_exc()})
  raise
