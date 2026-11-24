from observer import *
from train_progress_axis import axis_input
from geometry_guard import hand_occlusion
def main():
 torch.set_num_threads(2);prog=Ensemble('axis');out=[]
 for r in read(U/'INPUTS.json')['rows']:
  z=read(U/'local_sequence'/f'p{r["person"]:02d}.json');rel=np.array(z['rel']);ch=dict(np.load(r['channels']));oc={h:hand_occlusion(ch,h)for h in['L','R']};heads=[]
  for k in[1,2]:
   for a,b in z['regions'][str(k)]:
    vv=rel[a:b+1,(k-1)*2:k*2];q,p=prog.predict(axis_input(vv,k-1)[None],.75)
    heads.append(dict(region=k,span=[a,b],full=int(q[0]),p=p[:,0].round(3).tolist()))
  a,b=z['regions']['2'][0]if z['regions']['2']else(0,0);items=[]
  for h in['L','R']:
   s=np.array(z['shape'][h]['states']);vis=~oc[h];items.append(dict(hand=h,span=[max(0,a-6),b],valid=[dict(i=i,s=int(s[i]))for i in range(max(0,a-6),b+1)if vis[i]],masked=int(oc[h][a:b+1].sum())))
  out.append(dict(person=r['person'],progress=heads,shape=items))
 write(U/'COMPOSITION_DIAGNOSTIC.json',dict(rows=out));print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
