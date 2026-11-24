from thanks_candidate import *
from audit_public import read,write,torch,V,B,O,adapter
from train_shape import S
from projection_guard import ProjectionHead

def main():
 torch.set_num_threads(2);dest=D/'thanks';dest.mkdir(exist_ok=True)
 write(dest/'FROZEN.json',dict(candidate='Unknown-only upright recovery from two independently trained coarse heads with no opposing individual finger evidence. Same .85 consensus. Never copy preceding state; no old-known or projection-conflict override.',public_only=True,training_runs=0,gate='No added wrong known labelled anchor; no lost normal/short-motion or shape-error sensitivity. Promote only if meaningful improvement observed.'))
 old,new=CombinedVisible(),ThanksCandidate();rows=[];cache={}
 sources=read(H/'PUBLIC_INPUTS.json')['rows']+read(H/'public_camera/INPUTS.json')['rows']+read(H/'public_views/INPUTS.json')['rows']+read(H/'confirmation_views/INPUTS.json')['rows']
 for r in sources:
  assert 'Temp'not in r['raw']and 'user_'not in r['raw']
  raw=dict(np.load(r['raw']));a=old.analyze(raw);b=new.analyze(raw)
  sa={c['id']:c['status']for c in a['criteria']};sb={c['id']:c['status']for c in b['criteria']}
  row=dict(id=r['id'],before=sa,after=sb,events_before=len(a['events']),events_after=len(b['events']),changed=sa!=sb,
      lost_supported=[k for k in sa if sa[k]=='supported'and sb[k]!='supported'],new_difference=[k for k in sa if sa[k]!='difference'and sb[k]=='difference'])
  rows.append(row);write(dest/(r['id']+'.json'),dict(before=a,after=b))
  if row['changed']:print(row,flush=True)
 # Inspect the actual sequence pipeline at the existing labelled DEV anchors,
 # not a different precomputed classifier aggregation.
 meta=read(S/'INPUTS.json')['rows'];dat=np.load(S/'INPUTS.npz');dev=np.load(B/'local_shape/CONSENSUS_DEVELOPMENT.npz');target=dev['target'].copy()
 correction=next(i for i,r in enumerate(meta)if r['clip_id']=='NIA_SL_WORD0071_REAL13_F'and r['frame']==82 and r['side']=='L');target[correction]=2
 extras={r['clip_id']:r for r in read(O/'word100_feature_expansion_20261113/INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']}
 checks=[]
 for i in np.flatnonzero(dev['dev']):
  r=meta[i];cid=r['clip_id'];key=(cid,r['side'])
  if key not in cache:
   path=extras[cid]['raw']if cid in extras else read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json')['raw_path']
   raw=dict(np.load(path));ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
   cache[key]=(ch['frame_id'],old.shape.sequence(ch,r['side'])['states'],new.shape.sequence(ch,r['side'])['states'])
  ids,a,b=cache[key];j=int(np.flatnonzero(ids==r['frame'])[0]);checks.append(dict(index=int(i),clip=cid,side=r['side'],frame=r['frame'],truth=int(target[i]),before=int(a[j]),after=int(b[j])))
 summary=dict(normal=rows,anchors=checks,lost_correct=sum(x['before']==x['truth']and x['after']!=x['truth']for x in checks),
     new_wrong=sum(x['after']>=0 and x['after']!=x['truth']and(x['before']<0 or x['before']==x['truth'])for x in checks),
     recovered_correct=sum(x['before']<0 and x['after']==x['truth']for x in checks),private_data=False)
 write(dest/'RESULT.json',summary);print('THANKS_RESULT', {k:v for k,v in summary.items()if k not in['normal','anchors']}, 'changed',sum(x['changed']for x in rows),flush=True)
if __name__=='__main__':main()
