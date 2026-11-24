"""Ordered visible evidence, independent shape checks and native-time spans."""
from observer import *
from train_progress_axis import axis_input
from geometry_guard import hand_occlusion
def extrema_span(values,a,b,axis):
 v=values[a:b+1,axis];first=int(np.argmin(v))+a;second=int(np.argmax(v))+a
 return sorted([first,second])
def criterion(cid,status='unknown',source=None,difference=None,**kw):
 return dict(id=cid,status=status,source_spans=source or[],difference_spans=difference or[],**kw)
def shape_check(cid,states,expected,a,b,visible,ts):
 s=np.array(states);support=[];wrong=[]
 for value in[True,False]:
  mask=(s==expected)if value else((s>=0)&(s!=expected));mask&=visible;ids=np.flatnonzero(mask[a:b+1])+a
  for run in contiguous(ids):
   if len(run)>=3 and(value or ts[run[-1]]-ts[run[0]]>=100000):(support if value else wrong).append([int(run[0]),int(run[-1])])
 return criterion(cid,'difference'if wrong else'supported'if support else'unknown',support,wrong,expected=expected)
class Lesson(AppleObserver):
 def __init__(self):super().__init__(local=True);self.axis=Ensemble('axis')
 def analyze(self,raw):
  z=super().analyze(raw);ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));ts=ch['timestamp_us'];rel=np.array(z['rel']);av=np.array(z['available']);criteria=[];events=[]
  for k in[1,2]:
   candidates=z['regions'][str(k)]
   if not candidates:criteria.append(criterion('mouth_motion'if k==1 else'wipe_motion'));continue
   a,b=max(candidates,key=lambda x:x[1]-x[0]);axis=k-1;values=rel[:,axis*2:axis*2+2];ea,eb=extrema_span(values,a,b,axis)
   q,_=self.axis.predict(axis_input(values[ea:eb+1],axis)[None],.75);actual=int(q[0]);expected=k;wrong=actual==3-k
   status='unknown'if not av[ea:eb+1,axis].all()or eb-ea<2 else'difference'if wrong else'supported'if actual==expected else'unknown'
   criteria.append(criterion('mouth_motion'if k==1 else'wipe_motion',status,[[ea,eb]]if status=='supported'else[],[[ea,eb]]if wrong else[],observed_state=actual,observed_span=[a,b],extent=[ea,eb]))
   events.append(dict(region=k,start=ea,end=eb,context=[a,b]))
  mouth=next((e for e in events if e['region']==1),None);pair=next((e for e in events if e['region']==2),None)
  if mouth:
   a,b=mouth['start'],mouth['end'];f=np.array(z['shape']['R']['fingers']);match=np.where((f==[1,0,0,0]).all(1),1,np.where((f>=0).all(1),0,-1))
   criteria.append(shape_check('index_shape',match,1,a,b,np.array(z['shape']['R']['available']).all(1),ts))
  else:criteria.append(criterion('index_shape'))
  for h,cid,expected in[('L','support_shape',1),('R','wipe_shape',0)]:
   if pair:
    a,b=pair['context'];a=max(0,a-6)if h=='L'else a
    oc=hand_occlusion(ch,h);s=np.array(z['shape'][h]['states']);criteria.append(shape_check(cid,s,expected,a,b,~oc,ts))
   else:criteria.append(criterion(cid))
  if mouth and pair:
   correct=mouth['end']<pair['start'];criteria.append(criterion('sequence','supported'if correct else'difference',[[mouth['start'],mouth['end']],[pair['start'],pair['end']]]if correct else[],[[pair['start'],mouth['end']]]if not correct else[]))
  else:criteria.append(criterion('sequence'))
  return dict(target='WORD0099',criteria=criteria,required_ids=[c['id']for c in criteria],events=events,observations=z,qualified=False,overall_pass=False,whole_word_correct=None,
   seconds=z['seconds'],timestamp_us=ts.tolist(),limitations=['Exact 3D contact and hidden fingers not inferred','Projected region/axis and shape evidence; original source expressions only'])
def main():
 torch.set_num_threads(2);net=Lesson();out=[]
 for r in read(U/'INPUTS.json')['rows']:
  z=net.analyze(dict(np.load(r['raw'])));write(U/'composed'/f'p{r["person"]:02d}.json',z);out.append(dict(person=r['person'],criteria={c['id']:c['status']for c in z['criteria']},events=z['events']))
 write(U/'COMPOSED.json',dict(rows=out));print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
