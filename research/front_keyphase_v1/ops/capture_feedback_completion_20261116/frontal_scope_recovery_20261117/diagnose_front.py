from probe_front import *
def main():
 torch.set_num_threads(2);net=RepairedVisible();rows=[]
 for r in read(N/'INPUTS.json')['rows']:
  if r['id']not in ['NIA_SL_WORD1290_REAL07_F','NIA_SL_WORD1290_REAL11_F']:continue
  require_front(r);raw=dict(np.load(r['raw']));q=net.analyze(raw);from visible_lesson import adapter
  ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));s=net.shape.sequence(ch,'R');x=ch['tcn280'];a,_=net.shape.legacy.before.sequence(x,'R');b,_=net.shape.legacy.expanded.sequence(x,'R');c,_=net.shape.repaired.sequence(x,'R');p=net.plane.sequence(ch,'R');info=[]
  for e in q['events']:
   for i in range(e['start'],e['end']+1):info.append(dict(frame=i,output=int(s['states'][i]),fingers=s['fingers'][i].tolist(),available=s['available'][i].tolist(),base=int(a[i]),expanded=int(b[i]),repaired=int(c[i]),plane=int(p[i])))
  rows.append(dict(id=r['id'],events=q['events'],states=info));print(r['id'],info,flush=True)
 write(N/'FRONT_CAUSES.json',dict(rows=rows))
if __name__=='__main__':main()
