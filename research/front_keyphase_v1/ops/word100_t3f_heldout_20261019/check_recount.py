"""Synthetic integrity checks of the new 200-item score verifier, no real predictions."""
from common import *
from verify import recount
import math,copy,ast

def main():
 rows=[]
 for sid in [17,18]:
  for label in range(100):
   rows.append({'clip_id':f'synthetic_{sid}_{label}','signer_id':sid,'label_index':label,'status':'OK','logits':[0.]*100,'prediction':0,'top5':list(range(5)),'correct':label==0,'top5_correct':label<5,'cross_entropy':math.log(100)})
 r=recount(rows);assert r['correct']==2 and r['top5']==10 and r['failures']==0
 cases=['equal_logits_tie_ranks_and_denominator']
 fail=copy.deepcopy(rows);fail[0].update(status='INPUT_UNAVAILABLE',logits=None,prediction=None,top5=[],correct=False,top5_correct=False)
 q=recount(fail);assert q['correct']==1 and q['failures']==1;cases.append('failed_input_kept_in_denominator')
 for name,mutation in [('wrong_top5',lambda x:x[0].update(top5=[1,0,2,3,4])),('wrong_cross_entropy',lambda x:x[0].update(cross_entropy=0)),('missing_video',lambda x:x.pop()),('duplicate_video',lambda x:x[0].update(clip_id=x[1]['clip_id']))]:
  bad=copy.deepcopy(rows);mutation(bad)
  try:recount(bad)
  except AssertionError:cases.append(name+'_rejected')
  else:raise AssertionError(name+' was accepted')
 for p in U.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
 write(U/'VERIFIER_SELFCHECK.json',{'at':now(),'status':'PASS','checks':cases,'real_prediction_count':0,'training':0,'verifier_sha256':sha(U/'verify.py')},new=True)
 print('PASS',len(cases),'synthetic score-verifier checks; Python syntax valid')

if __name__=='__main__':main()
