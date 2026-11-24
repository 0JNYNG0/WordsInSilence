from candidate import *
from terminal_learning import terminal_features
class VerifiedCandidate(Candidate):
 def __init__(self):super().__init__();self.axis=Ensemble('direction');self.terminal=Ensemble('terminal')
 def analyze(self,raw):
  z=super().analyze(raw);ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));x,av=terminal_features(ch);q,p=self.terminal.predict(x,.85);q[~av]=-1
  pair=next((e for e in z['events']if e['region']==2),None);c=criterion('wipe_terminal')
  if pair:
   a,b=pair['context'];runs=[rr for rr in contiguous(np.flatnonzero(q[a:b+1]==1)+a)if len(rr)>=2]
   if runs:c=criterion('wipe_terminal','supported',[[int(rr[0]),int(rr[-1])]for rr in runs])
   # Long unresolved active-hand interval cannot be green merely because a
   # usable palm was observed before it. The occluded support hand is separate.
   visible=~hand_occlusion(ch,'R');bad=[rr for rr in contiguous(np.flatnonzero(~visible[a:b+1])+a)if len(rr)>=4]
   if bad:
    for cc in z['criteria']:
     if cc['id']=='wipe_shape'and cc['status']=='supported':cc['status']='unknown';cc['unobserved_spans']=[[int(rr[0]),int(rr[-1])]for rr in bad]
  z['criteria'].append(c);z['required_ids']=[c['id']for c in z['criteria']];z['terminal_states']=q.tolist();return z
def main():
 torch.set_num_threads(2);net=VerifiedCandidate();out=[]
 for r in read(U/'INPUTS.json')['rows']:
  z=net.analyze(dict(np.load(r['raw'])));write(U/'verified_candidate'/f'p{r["person"]:02d}.json',z);out.append(dict(person=r['person'],criteria={c['id']:c['status']for c in z['criteria']}))
 write(U/'VERIFIED_NORMAL.json',dict(rows=out));print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
