import candidate as base
from candidate import *

def qualified_endpoints(ch,motion,limit,body,old):
 out=[]
 for e in observed_endpoint_events(ch,motion,limit):
  a,b=e['start'],e['end'];observed=body[a:b+1]
  # Explicit non-action body posture rejects preparatory approach. Unknown
  # frames are not relabeled, and at least two observed action states remain.
  if np.any(observed==0)or np.count_nonzero(observed>0)<2:continue
  # The terminal hold belongs to its cycle; don't count it a second time.
  if any(max(a,p['start'])<=min(b,p['end'])for p in old):continue
  out.append(e)
 return out

class FinalLesson(ShapeLesson):
 def analyze(self,raw,lesson=None):
  started=time.perf_counter();q=super().analyze(raw,lesson);ch=channels(raw);m=self.motion.sequence(ch)
  approved=qualified_endpoints(ch,m,self.target_region.limit,np.array(q['body_relation_states']),q['events'])
  # Reuse the shared criterion composer with the already qualified proposals.
  return compose(self,q,raw,started,approved)

def compose(net,q,raw,started,approved):
 # Pass proposals explicitly; no global patching or identity-specific gate.
 return base.append_endpoints(net,q,raw,started,approved)

if __name__=='__main__':
 torch.set_num_threads(2);net=FinalLesson();out=L/'final_checks';out.mkdir(exist_ok=True);rows=[]
 for r in current_sources():
  a=read(L/'audit'/(r['id']+'.json'))['output'];b=net.analyze(dict(np.load(r['raw'])));aa={c['id']:c['status']for c in a['criteria']};bb={c['id']:c['status']for c in b['criteria']};row=dict(id=r['id'],before=aa,after=bb,events_before=len(a['events']),events_after=len(b['events']),lost=[k for k,v in aa.items()if v=='supported'and bb[k]!='supported'],new_warning=[k for k,v in bb.items()if v=='difference'and aa[k]!='difference'],bounds=all(e in b['events']for e in a['events']));rows.append(row);write(out/(r['id']+'.json'),dict(before=a,after=b))
  if aa!=bb or len(a['events'])!=len(b['events']):print(row,flush=True)
 write(out/'RESULT.json',dict(rows=rows,regressions=sum(bool(r['lost']or r['new_warning']or not r['bounds'])for r in rows),full_before=sum(all(v=='supported'for v in r['before'].values())for r in rows),full_after=sum(all(v=='supported'for v in r['after'].values())for r in rows)))
