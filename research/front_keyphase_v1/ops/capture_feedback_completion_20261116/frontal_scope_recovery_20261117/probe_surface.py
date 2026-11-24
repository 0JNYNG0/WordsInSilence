from probe_front import *
sys.path.insert(0,str(G/'apple_v1'))
from train_surface import Surface
from visible_lesson import adapter

def main():
 torch.set_num_threads(2);net=Surface();rows=[]
 for r in read(N/'INPUTS.json')['rows']:
  if r['id']not in ['NIA_SL_WORD1290_REAL07_F','NIA_SL_WORD1290_REAL11_F']:continue
  require_front(r);raw=dict(np.load(r['raw']));ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));s=net.sequence(ch['tcn280'],'R');q=read(N/'front_linked'/(r['id']+'.json'))['after'];a=q['events'][0]['start'];b=q['events'][-1]['end'];rows.append(dict(id=r['id'],frames=list(range(a,b+1)),surface=s[a:b+1].tolist()));print(rows[-1])
 write(N/'SURFACE_PROBE.json',dict(rows=rows))
if __name__=='__main__':main()
