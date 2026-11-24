"""Display-only alignment by observed clause identity, never by action count."""
import numpy as np
def align_clauses(query,reference,events,reference_events,dtw):
 lookup={e['region']:e for e in reference_events};pairs=[];matched=[]
 for e in sorted(events,key=lambda e:e['start']):
  r=lookup.get(e['region'])
  if not r:continue
  a,b=e['start'],e['end'];c,d=r['start'],r['end']
  local=dtw(query[a:b+1],reference[c:d+1])['path']
  pairs.extend((int(i+a),int(j+c))for i,j in local);matched.append(e['region'])
 if pairs:
  ids=sorted(set(i for i,j in pairs));ys=[sum(j for i,j in pairs if i==x)/sum(i==x for i,j in pairs)for x in ids]
  # Outside a confirmed action, hold its nearest endpoint. No missing clause
  # is manufactured by stretching a lone observed action over two references.
  mapping=np.rint(np.interp(np.arange(len(query)),ids,ys)).astype(int)
 else:mapping=np.zeros(len(query),dtype=int)
 return [[i,int(j)]for i,j in enumerate(mapping)],dict(scope='Display only, within the same observed clause; no completion evidence',matched_regions=matched,partial_reference=len(matched)<len(reference_events),order_differs=matched!=sorted(matched),unmatched_regions=sorted(set(lookup)-set(matched)))
