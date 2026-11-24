from endpoint_v2 import *

class TemporalVisibilityLesson(ShapeLesson):
 def form_states(self,ch,hand):
  s,recovered,form=PreservedArrivalLesson.form_states(self,ch,hand);ob=self.shape.observations[hand];projected,_=self.shape.projection.sequence(ch['tcn280'],hand)
  # A new coarse positive needs some independent visible extended-finger
  # evidence. A palm classifier alone cannot resolve all-unknown digits, or
  # override a separately observed distinct-finger pattern.
  use=(s!=2)&(form==0)&(projected!=2)&np.any(ob['fingers']==1,axis=1)&ob['available'].all(1)&~ob['projection_conflict']
  recovered|=use&(s!=0);s[use]=0
  return s,recovered,form
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();q=super().analyze(raw,lesson);ch=channels(raw);m=self.motion.sequence(ch)
  prior={c['id']:dict(c)for c in q['criteria']};approved=qualified_endpoints(ch,m,self.target_region.limit,np.array(q['body_relation_states']),q['events']);q=compose(self,q,raw,started,approved);ts=ch['timestamp_us']
  # Adding an event changes inspection coverage, not the reliability of an
  # already ambiguous observation. Retain its exact uncertainty provenance.
  for c in q['criteria']:
   p=prior[c['id']];uncertain=p.get('uncertain_spans',[])
   if p['status']=='difference'or not uncertain:continue
   blocked=np.zeros(len(ts),bool)
   for a,b in uncertain:blocked[a:b+1]=True
   remaining=[]
   for a,b in c.get('difference_spans',[]):
    ids=np.arange(a,b+1)
    remaining.extend([[int(r[0]),int(r[-1])]for r in contiguous(ids[~blocked[a:b+1]])if len(r)>=3 and ts[r[-1]]-ts[r[0]]>=100000])
   c.update(status='difference'if remaining else'unknown',difference_spans=remaining,uncertain_spans=uncertain,uncertainty_reason=p['uncertainty_reason'])
  q['seconds']=time.perf_counter()-started;return q

def main():
 torch.set_num_threads(2);net=TemporalVisibilityLesson();out=L/'qualified_checks';out.mkdir(exist_ok=True);rows=[]
 for r in current_sources():
  a=read(L/'audit'/(r['id']+'.json'))['output'];b=net.analyze(dict(np.load(r['raw'])));aa={c['id']:c['status']for c in a['criteria']};bb={c['id']:c['status']for c in b['criteria']};row=dict(id=r['id'],before=aa,after=bb,events_before=len(a['events']),events_after=len(b['events']),lost=[k for k,v in aa.items()if v=='supported'and bb[k]!='supported'],new_warning=[k for k,v in bb.items()if v=='difference'and aa[k]!='difference'],bounds=all(e in b['events']for e in a['events']));rows.append(row);write(out/(r['id']+'.json'),dict(before=a,after=b))
 write(out/'RESULT.json',dict(rows=rows,regressions=sum(bool(r['lost']or r['new_warning']or not r['bounds'])for r in rows),full_before=sum(all(v=='supported'for v in r['before'].values())for r in rows),full_after=sum(all(v=='supported'for v in r['after'].values())for r in rows)));print('ADMISSION',read(out/'RESULT.json')['regressions'],flush=True)
if __name__=='__main__':main()
