"""Visible evidence before completion gates; no handshape gate hides direction."""
from sparse_observer import *
def runs(mask):
 ids=np.flatnonzero(mask)
 return [a.tolist()for a in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)if len(a)]
def criterion(cid,status='unknown',spans=None,diff=None):return dict(id=cid,status=status,source_spans=spans or[],difference_spans=diff or[])
def shape_requirement(cid,shape,h,expected,mask,ts):
 s=shape[h];desired=1 if expected==0 else 0
 # Same criterion as the admitted coach: known whole-hand state, no known
 # contrary finger. Require every long finger to have actual positive evidence
 # within the run, rather than forcing four decisions on every single frame.
 good=(s['states']==expected)&~((s['fingers']>=0)&(s['fingers']!=desired)).any(1);wrong=(s['states']>=0)&(s['states']!=expected)
 wrong|=((s['fingers']>=0)&(s['fingers']!=(1 if expected==0 else 0))).any(1)
 support=[[r[0],r[-1]]for r in runs(good&mask)if len(r)>=3 and(s['fingers'][r]==desired).any(0).all()]
 diff=[[r[0],r[-1]]for r in runs(wrong&mask)if len(r)>=3 and ts[r[-1]]-ts[r[0]]>=100000]
 return criterion(cid,'difference'if diff else'supported'if support else'unknown',support,diff)
class FeedbackLesson(SparseLesson):
 def analyze(self,raw):
  z=super().analyze(raw);n=len(raw['timestamp_us']);ts=raw['timestamp_us'];criteria=[];events=[]
  ids=['sweep_shape','sweep_direction','sweep_endpoint','left_fist','right_fist','repeat_motion','order']
  if not z['trace']:return dict(**z,target='SEN0354',criteria=[criterion(i)for i in ids],required_ids=ids,qualified=False,overall_pass=False,timestamp_us=ts.tolist())
  xy=raw['xy_px'];sc=np.linalg.norm(xy[:,5]-xy[:,6],axis=1);el=xy[:,7];wr=xy[:,9];tip=xy[:,112];axis=wr-el;den=np.square(axis).sum(1)
  q=np.sum((tip-el)*axis,1)/np.maximum(den,1);gap=np.linalg.norm(tip-(el+q[:,None]*axis),axis=1)/np.maximum(sc,1);level=(tip[:,1]-(xy[:,5,1]+xy[:,6,1])/2)/np.maximum(sc,1)
  on=(gap<=.30)&(q>=-.15)&(q<=1.4)&(level>-.1)&(level<1.35)&(den>(.15*sc)**2)&np.isfinite(q);on[:3]=False;on[-3:]=False
  sweeps=[e for e in z['events']if e['kind']=='sweep_observed'];span=lambda e:[int(np.searchsorted(ts,e['start_us'])),int(np.searchsorted(ts,e['end_us']))]
  candidates=[r for r in runs(on)if len(r)>=3];sweep=max(candidates,key=len)if candidates else[];mask=np.zeros(n,bool)
  if sweeps:
   a,b=span(sweeps[0]);sweep=list(range(a,b+1))
  if sweep:mask[sweep]=True
  criteria.append(shape_requirement('sweep_shape',self.last_shape,'R',0,mask,ts))
  c=criterion('sweep_direction')
  if sweep:
   a,b=sorted([sweep[int(np.argmin(q[sweep]))],sweep[int(np.argmax(q[sweep]))]])
   # Preserve the exact learned evidence that produced the observed event.
   # Re-querying a wider interval can mix the transition into the action.
   direction=sweeps[0]['action_direction']if sweeps else self.action.movement(int(ts[a]),int(ts[b]),2,'R','L')
   if len(sweep)<3 or np.any(np.diff(ts[sweep])>self.coach.gap_pause_us):direction=dict(eligible=False,state=-1,reason='observation_gap')
   status='supported'if direction['eligible']and direction['state']==2 else'difference'if direction['eligible']and direction['state']==0 else'unknown'
   c=criterion('sweep_direction',status,[[a,b]]if status=='supported'else[],[[a,b]]if status=='difference'else[]);c['model_evidence']=direction;events.append(dict(region=1,start=a,end=b,context=[sweep[0],sweep[-1]]))
  criteria.append(c)
  criteria.append(criterion('sweep_endpoint','supported',[span(e)for e in sweeps])if sweeps else criterion('sweep_endpoint'))
  # Preserve the existing sequential repeat gate. Before the first action is
  # found, an ambiguous preparation pose does not become a false fist error.
  paired=np.zeros(n,bool);repeat=[e for e in z['events']if e['kind']=='downstroke_observed'];complete=z['completed'][1]
  if complete:
   a,b=span(repeat[0])[0],span(repeat[-1])[1];paired[a:b+1]=True
  elif sweeps:
   # Locate an actual downward attempt independently of its handshape. A hand
   # being lifted into the next pose is not yet the requested repeated action.
   start=span(sweeps[0])[1]+1;top=None;top_i=None;begin=None;last=None
   for i in range(start,n-3):
    vv=(xy[i,[91,112]]-(xy[i,5]+xy[i,6])/2)/max(sc[i],1)
    valid=bool(np.isfinite(vv).all()and abs(vv[0,1]-vv[1,1])<=.22 and .30<=np.linalg.norm(vv[0]-vv[1])<=1.50 and(vv[:,1]>-.15).all()and(vv[:,1]<.90).all())
    if not valid:
     if begin is not None:paired[begin:(last or begin)+1]=True
     top=None;top_i=None;begin=None;last=None;continue
    y=vv[:,1];mean=float(y.mean())
    if top is None or (begin is None and mean<float(top.mean())):top=y.copy();top_i=i
    if begin is None and mean-float(top.mean())>=.060 and (y-top>=.045).all():begin=top_i
    if begin is not None:last=i
   if begin is not None:paired[begin:(last or begin)+1]=True
  for h,cid in [('L','left_fist'),('R','right_fist')]:criteria.append(shape_requirement(cid,self.last_shape,h,1,paired,ts))
  criteria.append(criterion('repeat_motion','supported',[span(e)for e in repeat])if complete else criterion('repeat_motion'))
  if repeat:events.append(dict(region=2,start=span(repeat[0])[0],end=span(repeat[-1])[1],context=[span(repeat[0])[0],span(repeat[-1])[1]]))
  criteria.append(criterion('order','supported',[span(sweeps[0]),span(repeat[-1])])if sweeps and complete else criterion('order'))
  z.update(target='SEN0354',criteria=criteria,required_ids=ids,qualified=False,overall_pass=False,timestamp_us=ts.tolist(),display_events=events,repeat_count=len(repeat),scope='Single public source guided prototype; no hello word-TCN or multi-person accuracy. Required stages combine learned shape/direction with unchanged explicit geometric progression.')
  return z
