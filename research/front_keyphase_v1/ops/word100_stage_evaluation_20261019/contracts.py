from common import *
from criteria import assess_tokens,compare

def main():
 assert not (U/'CONTRACT_TESTS.json').exists()
 simple={'expert_reviewed':True,'blocks':[{'sequence':[s],'minimum_cycles':1,'maximum_cycles':1} for s in 'ABC']}
 repeat={'expert_reviewed':True,'blocks':[{'sequence':['A','B'],'minimum_cycles':1,'maximum_cycles':None},{'sequence':['C'],'minimum_cycles':1,'maximum_cycles':1}]}
 optional={'expert_reviewed':True,'blocks':[{'sequence':['A'],'minimum_cycles':1,'maximum_cycles':1},{'sequence':['B'],'minimum_cycles':0,'maximum_cycles':1},{'sequence':['C'],'minimum_cycles':1,'maximum_cycles':1}]}
 cases=[('normal','ABC',simple,True,'MATCHES_REVIEWED_STRUCTURE'),('slow_dwell','AAAABBBCCCCC',simple,True,'MATCHES_REVIEWED_STRUCTURE'),
 ('missing_required','AC',simple,True,'STRUCTURE_REVIEW_NEEDED'),('order_swap','BAC',simple,True,'STRUCTURE_REVIEW_NEEDED'),
 ('allowed_repeat','ABABABC',repeat,True,'MATCHES_REVIEWED_STRUCTURE'),('extra_phase','ABDC',repeat,True,'STRUCTURE_REVIEW_NEEDED'),
 ('optional_omitted','AC',optional,True,'MATCHES_REVIEWED_STRUCTURE'),('optional_present','ABC',optional,True,'MATCHES_REVIEWED_STRUCTURE'),
 ('unknown_gap','A?C',simple,True,'INSUFFICIENT_OBSERVATION'),('partially_seen','AC',simple,False,'INSUFFICIENT_OBSERVATION'),
 ('unknown_repeat_policy','ABABC',{'expert_reviewed':False,'blocks':[]},True,'REVIEW_REQUIRED'),
 ('missing_not_reviewed','AC',{'expert_reviewed':False,'blocks':[]},True,'REVIEW_REQUIRED')]
 output=[]
 for name,tokens,c,visible,expected in cases:
  r=assess_tokens(list(tokens),c,visible);assert r['status']==expected and r['error'] is None
  output.append({'case':name,'expected':expected,'actual':r})
 # Permuting anonymous phase names cannot change a result.
 relabel=0
 for offset in range(25):
  mp={t:f'phase_{offset}_{i}' for i,t in enumerate('ABCD')};mp['?']='?'
  for _,tokens,c,visible,expected in cases:
   cc=read_json_copy(c)
   for block in cc['blocks']:block['sequence']=[mp[t] for t in block['sequence']]
   assert assess_tokens([mp[t] for t in tokens],cc,visible)['status']==expected;relabel+=1
 # Mathematical requirements for independent phase signatures. No linguistics.
 def encoded(tokens):
  x=np.zeros((len(tokens),280),np.float32);x[:,195:279]=1;x[1:,279]=1/30
  for i,t in enumerate(tokens):
   x[i,18:102]=ord(t)-ord('A');x[i,102:186]=ord(t)-ord('A')
  return x
 base=encoded('ABC');geometry=[]
 for name,tokens in [('dwell','AABBCCC'),('omission','AC'),('order','BAC'),('repeat','ABABC')]:
  r=compare(base,encoded(tokens),'global');geometry.append({'name':name,'value':r})
  if name=='dwell':assert r['ordered']==r['bag']==0
  elif name=='omission':assert r['bag']>0
  elif name=='order':assert r['bag']==0 and r['order_gap']>0
 # A geometry-only score can penalize an allowed repeated cycle: a real
 # counterexample to treating distance/order cost as a universal sign verdict.
 assert geometry[-1]['value']['ordered']>0
 write(U/'CONTRACT_TESTS.json',{'status':'PASS','reviewed_symbolic_cases':output,'relabel_invariance_cases':relabel,'geometric_counterexamples':geometry,
  'language_validity':'Synthetic token contracts only; not reviewed Korean signs. Public catalog remains unreviewed.','code_sha256':sha(__file__),'criteria_sha256':sha(U/'criteria.py')})
 print('PASS12criteria cases+300phase-name invariances; geometry repeat counterexample verified.')

def read_json_copy(x):return json.loads(json.dumps(x))
if __name__=='__main__':main()
