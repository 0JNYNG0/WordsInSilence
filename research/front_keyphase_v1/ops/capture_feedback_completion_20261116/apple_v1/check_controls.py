from candidate import *
import copy
def selection(raw,ids):
 n=len(raw['frame_id']);ids=np.asarray(ids);z={k:v[ids].copy()if isinstance(v,np.ndarray)and v.ndim and len(v)==n else copy.deepcopy(v)for k,v in raw.items()}
 z['timestamp_us']=np.arange(len(ids),dtype=np.int64)*33333;z['frame_id']=np.arange(len(ids));return z
def replace_hand(raw,donor,sourcehand,frame,side,a,b):
 z=copy.deepcopy(raw);sa=91 if sourcehand=='L'else 112;ta=91 if side=='L'else 112
 p=donor['xy_px'][frame,sa:sa+21].copy();p-=p[0];scale=np.linalg.norm(p[9])
 if sourcehand!=side:p[:,0]*=-1
 for i in range(a,b+1):
  q=z['xy_px'][i,ta:ta+21];ss=np.linalg.norm(q[9]-q[0]);z['xy_px'][i,ta:ta+21]=p*(ss/max(scale,1e-5))+q[0];z['score'][i,ta:ta+21]=donor['score'][frame,sa:sa+21]
 return z
def main(factory=Candidate,destination=None,corrected=False):
 dest=destination or U;dest.mkdir(exist_ok=True);torch.set_num_threads(2);net=factory();inputs=read(U/'INPUTS.json')['rows'];tt={r['person']:r for r in read(U/'SOURCE_TRUTH.json')['rows']};fit=next(r for r in inputs if r['person']==6);donor=dict(np.load(fit['raw']));results=[]
 for r in inputs:
  if r['role']=='fit':continue
  z=dict(np.load(r['raw']));n=len(z['frame_id']);t=tt[r['person']];m1,m2=t['mouth_sweep'];w1,w2=t['wipe_sweeps'][0];last=t['pair_anchors'][-1];cases=[('normal',z,None,True)]
  for name,end in [('only_first',m2),('before_wipe',w1),('half_wipe',w1+(w2-w1)//2)]:cases.append((name,selection(z,np.arange(end+1)),None,'diagnostic'if corrected and name=='half_wipe'else False))
  ix=np.arange(n);a,b=(max(0,m1-10),w1-5)if corrected else(m1,m2);ix[a:b+1]=ix[a:b+1][::-1];cases.append(('reverse_mouth',selection(z,ix),'mouth_motion',False))
  ids=np.r_[np.arange(m2+1),np.arange(w1,w2+1)[::-1]];cases.append(('single_reverse_wipe',selection(z,ids),'wipe_motion',False))
  ids=np.r_[np.arange(m1),np.arange(w1,last+1),np.arange(m1,m2+1),np.arange(last+1,n)];cases.append(('swapped_order',selection(z,ids),'sequence',False))
  cases.append(('constant',selection(z,np.full(n,m2)),None,False))
  missing=copy.deepcopy(z);missing['score'][:,91:133]=0
  if corrected:missing['xy_px'][:,91:133]=np.nan
  cases.append(('missing',missing,None,False))
  shape=replace_hand(z,donor,'L',48,'R',m1,m2);cases.append(('index_to_fist',shape,'index_shape',False))
  shape=replace_hand(z,donor,'L',48,'R',w1,last);cases.append(('wipe_to_fist',shape,'wipe_shape',False))
  shape=replace_hand(z,donor,'L',48,'R',w1+max(1,(w2-w1)//2),last);cases.append(('change_during_wipe',shape,'wipe_shape',False))
  for sc in[.65,1.6]:
   q=copy.deepcopy(z);q['timestamp_us']=(q['timestamp_us']*sc).astype(np.int64);cases.append((f'time_{sc}',q,None,None))
  q=copy.deepcopy(z);center=np.array([int(z['source_width'])/2,int(z['source_height'])/2]);q['xy_px']=(q['xy_px']-center)*.75+center;cases.append(('spatial_scale',q,None,None))
  baseline=None
  for name,raw,required,expected in cases:
   a=net.analyze(raw);cs={c['id']:c['status']for c in a['criteria']};full=all(v=='supported'for v in cs.values())
   if name=='normal':baseline=cs;ok='difference'not in cs.values()
   elif expected=='diagnostic':ok=None
   elif expected is None:ok=cs==baseline
   else:ok=not full and (not required or cs[required]!='supported')
   result=dict(person=r['person'],case=name,pass_check=ok,all_supported=full,criteria=cs,events=a['events']);results.append(result)
   write(dest/'controls'/f'p{r["person"]}_{name}.json',a)
 write(dest/'CONTROLS.json',dict(rows=results,passed=sum(r['pass_check']is True for r in results),failed=[r for r in results if r['pass_check']is False],scope='Five previously seen public videos plus controlled coordinate/time edits. Not actual learner-error accuracy.',
  corrected_contract='Missing coordinates = NaN; zero score alone is not absence under original adapter contract. Reverse whole first-action source extent; temporal half is diagnostic, never an assumed semantic error.'if corrected else None))
 print(json.dumps(dict(total=len(results),passed=sum(r['pass_check']is True for r in results),diagnostic=sum(r['pass_check']is None for r in results),failures=[r for r in results if r['pass_check']is False]),ensure_ascii=False))
if __name__=='__main__':main()
