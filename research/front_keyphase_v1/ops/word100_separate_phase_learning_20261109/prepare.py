from lib import *
def main():
 assert not (P/'PREPARED.json').exists();torch.set_num_threads(4)
 src={r['clip_id']:r for r in read(D/'PREPARED.json')['rows']};rows=[r for r in common.load_rows() if r['word']>=0];net=common.SharedNet();net.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True));net.eval();out=[];reuse=0;inferences=0
 with torch.inference_mode():
  for r in rows:
   old=np.load(src[r['clip_id']]['path']);assert sha(src[r['clip_id']]['path'])==src[r['clip_id']]['sha256'];n=len(r['frame_id']);data={k:r[k] for k in ['raw_x','y','ym','stage_gold','direct_ym','frame_id','timestamp_us']}
   assert np.array_equal(old['frame_id'],r['frame_id'])
   for view,variant in [('full','normal'),('upper','upper')]:
    x=r['x_'+view]
    if r['role']=='development':
     cached=B/'predictions/shared100_42'/f"{r['clip_id']}_{variant}.npz";z=np.load(cached);data['h_'+view]=z['latent'];data['p_'+view]=z['attributes'];reuse+=1
    else:
     pred=net(torch.from_numpy(x[None]),torch.tensor([n]),torch.tensor([r['word']]));data['h_'+view]=pred['latent'][0].numpy();data['p_'+view]=pred['attributes'][0].sigmoid().numpy();inferences+=1
    # Geometry is in original body-normalized units, not feature-standardized units.
    raw=r['raw_x'].copy()
    if view=='upper':
     ch=np.load(r['channel']);raw=ch['feedback'] if 'feedback' in ch.files else ch['upper_mask']
    data['g_'+view]=geometry(raw)
   yy=old['stage_y'].copy();ww=old['stage_w'].copy()
   if r['role']=='fit':
    direct=(r['sy']>=0)&(r['sw']>=1);yy[direct]=r['sy'][direct]+1;ww[direct]=1
    # DTW-transferred labels keep their weak provenance; unlabelled gaps stay ignored.
    exact=r['stage_gold']>=0;yy[exact]=r['stage_gold'][exact]+1;ww[exact]=1
   else:yy[:]=-1;ww[:]=0
   data.update(stage_y=yy,stage_w=ww);path=P/'inputs'/f"{r['clip_id']}.npz";path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data)
   out.append(dict(clip_id=r['clip_id'],word=r['word'],word_id=r['word_id'],person=r['person'],role=r['role'],source_sha256=r['source_sha256'],path=str(path),sha256=sha(path),pairs=r['pairs'],orders=r['orders']))
 # Normalize geometry from FIT only; binary masks stay unstandardized.
 train=np.concatenate([np.load(r['path'])['g_full'] for r in out if r['role']=='fit']);mean=train.mean(0);std=train.std(0).clip(.05);mean[22:33]=0;std[22:33]=1;mean[55:]=0;std[55:]=1
 write(P/'GEOMETRY_NORMALIZATION.json',dict(mean=mean.tolist(),std=std.tolist(),fit_only=True))
 for r in out:
  z=np.load(r['path']);data={k:z[k].copy() for k in z.files}
  for view in ['full','upper']:
   g=data['g_'+view];g=(g-mean)/std;vm=data['g_'+view][:,22:33]>.5;vv=data['g_'+view][:,55:]>.5;g[:,:22]*=np.repeat(vm,2,1);g[:,33:55]*=np.repeat(vv,2,1);data['g_'+view]=g.astype(np.float32)
  np.savez_compressed(r['path'],**data);r['sha256']=sha(r['path'])
 write(P/'PREPARED.json',dict(rows=out,fit=36,development=12,base_checkpoint=str(CHECKPOINT),base_sha256=sha(CHECKPOINT),frozen_common_backbone=True,fit_feature_inferences=inferences,reused_development_features=reuse,no_DEV_training=True))
 event('prepared',fit=36,dev=12,reused=reuse,new_fit_features=inferences)
if __name__=='__main__':main()
