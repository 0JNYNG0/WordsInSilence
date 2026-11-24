"""Add explicitly reviewed preparation/transition negatives, leaving old runs intact."""
from train_observer import *
NEG={1:[38,40,58,60,63,87,90,92,95],2:[41,44,61,63,66,102,105],3:[35,38,60,63,97,100,103,107],4:[57,60,80,83,86,122,125],6:[18,21,44,47,90,93],7:[43,46,49,69,72,104,107,110],8:[47,50,52,55,75,78,81,110,113],9:[66,69,71,102,105,107,110]}
def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not(U/'REGION_REPAIR_FROZEN.json').exists()
 rx=[];ry=[];sources=[];truth={r['person']:r for r in read(U/'SOURCE_TRUTH.json')['rows']}
 for r in read(U/'INPUTS.json')['rows']:
  if r['role']!='fit':continue
  t=truth[r['person']];z=np.load(U/'features'/f'p{r["person"]:02d}.npz');f=z['features'];ch=dict(np.load(r['channels']));rel,av=relation(ch)
  anchors={0:sorted(set(NEG[r['person']]+list(range(0,max(1,t['mouth_anchors'][0]-10),5))+list(range(t['pair_anchors'][-1]+7,len(f),3)))),1:t['mouth_anchors'],2:t['pair_anchors']}
  for y,ids in anchors.items():
   for i in ids:
    if y>0 and not av[i,y-1]:continue
    # A local observed pose decides location. Long-context latent evidence is
    # intentionally reserved for action comparison, not used to fill boundaries.
    rx.append(np.r_[np.nan_to_num(rel[i]),av[i]]);ry.append(y);sources.append(dict(person=r['person'],frame=i,label=y))
 write(U/'REGION_REPAIR_FROZEN.json',dict(cause='Whole-context latent recognized action early during preparation; missing explicit transition negatives.',
   reviewed_negatives=NEG,records=sources,steps=500,seeds=[42,43,44],confidence=.7,
   comparison='First localizes actual body-relative region; 100ConvGRU remains for sequence features and display. Geometry probe cannot manufacture a completed action. No DEV threshold tuning.'))
 runs=train('region_local',np.array(rx),np.array(ry));write(U/'REGION_REPAIR_TRAINED.json',dict(runs=runs));print('REGION_REPAIR',len(rx))
if __name__=='__main__':main()
