from cycle_candidate import *

def coverage_change(a,b,criterion_id):
 before=next(c for c in a['criteria']if c['id']==criterion_id);after=next(c for c in b['criteria']if c['id']==criterion_id)
 added=[e for e in b['events']if e not in a['events']];spans=[[e['start'],e['end']]for e in added]
 return not all(c['status']=='supported'for c in a['criteria'])and after['status']=='unknown'and not after.get('difference_spans')and bool(added)and bool(after.get('unconfirmed_events'))and all(x in after.get('source_spans',[])for x in before.get('source_spans',[]))and all(x in spans for x in after['unconfirmed_events'])

def verify(a,b):
 aa={c['id']:c['status']for c in a['criteria']};bb={c['id']:c['status']for c in b['criteria']};coverage=[];lost=[]
 for k,v in aa.items():
  if v!='supported'or bb[k]=='supported':continue
  (coverage if coverage_change(a,b,k)else lost).append(k)
 warnings=[k for k,v in bb.items()if v=='difference'and aa[k]!='difference'];preserved=all(e in b['events']for e in a['events'])
 return dict(passed=not lost and not warnings and preserved,lost=lost,coverage_extended_unknown=coverage,new_warning=warnings,old_events_preserved=preserved)

def main():
 torch.set_num_threads(2);net=CycleLesson();out=M/'qualified';out.mkdir(exist_ok=True);rows=[]
 write(M/'GATE_CLARIFICATION.json',dict(initial='No lost supported status',clarification='Distinguish losing already observed support from adding a real previously missed action whose shape is unobservable. Previously overall-passed clips must remain passed. A previously incomplete clip may gain a genuine event and retain old per-event support while the new event explicitly remains unknown.',required=['Prior whole-clip was already incomplete','No new wrong warning','Every old event exact bounds preserved','Every previously supported source span preserved','Every new unconfirmed span belongs solely to an added event'],reason='16L first cycle handshape remains observed, second actual cycle is newly found but its support fingers are foreshortened. Forcing global green would invent evidence. This is reported separately, not hidden as zero status change.',author='Codex evidence audit; no thresholds or labels tuned'))
 for r in all_sources():
  a=baseline(r['id']);b=net.analyze(dict(np.load(r['raw'])));row=dict(id=r['id'],before={c['id']:c['status']for c in a['criteria']},after={c['id']:c['status']for c in b['criteria']},events_before=len(a['events']),events_after=len(b['events']),**verify(a,b));rows.append(row);write(out/(r['id']+'.json'),dict(before=a,after=b))
  if row['before']!=row['after']or len(a['events'])!=len(b['events']):print(row,flush=True)
 write(out/'RESULT.json',dict(rows=rows,failed=sum(not r['passed']for r in rows),coverage_extended_unknown=[r['id']for r in rows if r['coverage_extended_unknown']],full_before=sum(all(v=='supported'for v in r['before'].values())for r in rows),full_after=sum(all(v=='supported'for v in r['after'].values())for r in rows)));print('QUALIFIED',read(out/'RESULT.json')['failed'],flush=True)
if __name__=='__main__':main()
