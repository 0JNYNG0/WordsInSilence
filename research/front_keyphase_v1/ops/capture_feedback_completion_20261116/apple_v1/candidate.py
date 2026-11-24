from compose import *
from train_surface import Surface
class Candidate(Lesson):
 def __init__(self):super().__init__();self.surface=Surface()
 def analyze(self,raw):
  z=super().analyze(raw);ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));ts=ch['timestamp_us'];pair=next((e for e in z['events']if e['region']==2),None)
  if pair:
   a,b=pair['context'];new=[]
   for h,cid,expected in[('L','support_shape',1),('R','wipe_shape',0)]:
    s=self.surface.sequence(ch['tcn280'],h);vis=~hand_occlusion(ch,h);first=max(0,a-6)if h=='L'else a
    c=shape_check(cid,s,expected,first,b,vis,ts);c['semantic_target']='observed_palm_surface_form';new.append(c)
   z['criteria']=[c for c in z['criteria']if c['id']not in['support_shape','wipe_shape']]+new
  return z
def main():
 torch.set_num_threads(2);net=Candidate();out=[]
 for r in read(U/'INPUTS.json')['rows']:
  z=net.analyze(dict(np.load(r['raw'])));write(U/'candidate'/f'p{r["person"]:02d}.json',z);out.append(dict(person=r['person'],criteria={c['id']:c['status']for c in z['criteria']}))
 write(U/'CANDIDATE_NORMAL.json',dict(rows=out));print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
