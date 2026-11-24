"""One predeclared ablation: is the old hand-model veto still useful? No fit."""
from diagnose import *
from finger_runtime import classify,merge_observers
def candidate(f,a,b):
 s,base=combine(f,a,b);allknown=(f>=0).all(-1)
 conflict=((b==0)&(f==0).any(-1))|((b==1)&(f==1).any(-1))
 return np.where(allknown,base,np.where(conflict,-1,b))
def stats(y,b,c):
 return {k:dict(total=len(y),correct=int((v==y).sum()),wrong=int(((v>=0)&(v!=y)).sum()),unknown=int((v<0).sum()))for k,v in [('before',b),('candidate',c)]}|dict(new_wrong=int(((c>=0)&(c!=y)&~((b>=0)&(b!=y))).sum()),lost_correct=int(((b==y)&(c!=y)).sum()))
def main():
 torch.set_num_threads(2);net=ShapeConsensus()
 assert not(V/'CANDIDATE_FROZEN.json').exists()
 write(V/'CANDIDATE_FROZEN.json',dict(candidate='Remove only original whole-hand veto; keep all individual-finger contradictions and fixed .85 / 3-seed confidence. No additional weights or threshold fitting.',
  reason='Public REAL16/17 open-hand cases: old coarse confidently closed, expanded confidently open. Establish whether old veto prevents errors elsewhere before considering removal.',
  gate='No additional definite errors or loss of existing correct on historical reviewed hand labels AND separate previously reviewed quality-clean hand labels; no new completion under negative controls.',
  no_training=True,no_private_data=True,note='REAL16/17 now diagnosis, not independent final test. All other prior used sources explicitly regression, not globally untouched.'))
 d=np.load(O/'word100_finger_repair_20261115/INPUTS.npz');rows=json.loads((O/'word100_finger_repair_20261115/INPUTS.json').read_text(encoding='utf-8'))['rows'];ps={}
 for kind,models in net.finger.models.items():
  pp=[]
  for m,mu,sd in models:
   x=(d['x']-mu)/sd if kind=='shared_finger_chain'else(d['x_old']-mu)/sd
   if kind!='shared_finger_chain':x[:,:,63:]=0
   with torch.inference_mode():p=m(torch.tensor(x)).sigmoid().numpy()
   if kind!='shared_finger_chain':p=p[:,1:]
   pp.append(classify(p,d['availability']))
  ps[kind]=np.array(pp)
 f,_=merge_observers(ps['expanded_whole_hand'],ps['shared_finger_chain'],ps['old_v2']);a,_=net.before.windows(d['x'],d['availability']);b,_=net.expanded.windows(d['x'],d['availability']);old,_=combine(f,a,b);new=candidate(f,a,b)
 y=np.where((d['y']==1).all(1),0,np.where((d['y']==0).all(1),1,2));known=(d['y']>=0).all(1)&d['availability'].all(1);ids=known&np.array([r['role']=='development'for r in rows]);result={'historic':stats(y[ids],old[ids],new[ids])}
 changes=[dict(row=rows[i],truth=int(y[i]),before=int(old[i]),candidate=int(new[i]))for i in np.flatnonzero(ids&(old!=new))];write(V/'CANDIDATE_HISTORIC_CHANGES.json',changes)
 # Separately reviewed public fingers/coordinate quality, people 10/16.
 lab=json.loads((O/'word100_finger_error_split_20261115/LABELS.json').read_text(encoding='utf-8'))['rows'];pairs=[];cache={}
 for r in lab:
  if r['role']=='fit' or min(r['shape'])<0 or min(r['quality'])<1:continue
  # Corrected source truth is applied equally to both candidates.
  target=r['shape'].copy()
  if r['clip_id']=='NIA_SL_WORD2748_REAL10_F'and r['frame']==61 and r['hand']=='R':target[0]=0
  key=(r['raw_path'],r['hand'])
  if key not in cache:
   raw=load(r['raw_path']);xx=channel(raw);f=net.finger.infer(xx,r['hand'])['states'];a,_=net.before.sequence(xx,r['hand']);b,_=net.expanded.sequence(xx,r['hand']);old,_=combine(f,a,b);new=candidate(f,a,b);cache[key]=(old,new)
  old,new=cache[key];i=r['frame'];truth=0 if all(v==1 for v in target)else 1 if all(v==0 for v in target)else 2
  pairs.append(dict(clip=r['clip_id'],frame=i,hand=r['hand'],truth=truth,before=int(old[i]),candidate=int(new[i])))
 for person in [10,16]:
  rr=[r for r in pairs if f'REAL{person:02d}_'in r['clip']];result['source_reviewed_'+str(person)]=stats(np.array([r['truth']for r in rr]),np.array([r['before']for r in rr]),np.array([r['candidate']for r in rr]))
 result['admission_pass']=all(r['new_wrong']==0 and r['lost_correct']==0 for r in result.values());write(V/'CANDIDATE_RESULTS.json',result);write(V/'CANDIDATE_SEPARATE_ROWS.json',pairs);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
