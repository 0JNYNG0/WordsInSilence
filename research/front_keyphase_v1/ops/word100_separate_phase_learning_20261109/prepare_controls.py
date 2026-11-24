from lib import *

def digest(x):return hashlib.sha256(np.ascontiguousarray(x,dtype=np.float32).tobytes()).hexdigest()
def main():
 assert not (P/'EVAL_INPUTS.json').exists();torch.set_num_threads(4);rows={r['clip_id']:r for r in common.load_rows() if r['word']>=0};cache={}
 transform=load_module('_same_cached_variants',B/'evaluate.py').transform
 for r in rows.values():
  if r['role']!='development':continue
  for v in ['normal','upper','slow','fast','missing_2','missing_long','reversed_main']:
   x,ids=transform(r,v);z=np.load(B/'predictions/shared100_42'/f"{r['clip_id']}_{v}.npz");cache[(r['clip_id'],digest(x))]=(z['latent'],z['attributes'])
 net=common.SharedNet();net.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True));net.eval();out=[];reuse=0;newruns=0;gn=read(P/'GEOMETRY_NORMALIZATION.json')
 def add(cid,kind,raw,ts,fid,phase_old=None,old_decision=None,mapping=None,removed=None):
  nonlocal reuse,newruns
  r=rows[cid];x=common.normalize(raw);key=(cid,digest(x));n=len(x)
  if key in cache:h,a=cache[key];reuse+=1
  else:
   with torch.inference_mode():o=net(torch.from_numpy(x[None]),torch.tensor([n]),torch.tensor([r['word']]))
   h=o['latent'][0].numpy();a=o['attributes'][0].sigmoid().numpy();cache[key]=(h,a);newruns+=1
  g=geometry(raw);vm=g[:,22:33]>.5;vv=g[:,55:]>.5;g=(g-np.array(gn['mean']))/np.array(gn['std']);g[:,:22]*=np.repeat(vm,2,1);g[:,33:55]*=np.repeat(vv,2,1)
  path=P/'controls'/f'{cid}_{kind}.npz';path.parent.mkdir(exist_ok=True);data=dict(raw_x=raw,timestamp_us=ts,frame_id=fid,h=h,p=a,g=g.astype(np.float32))
  if phase_old is not None:data['phase_old']=phase_old
  np.savez_compressed(path,**data);out.append(dict(clip_id=cid,word=r['word'],word_id=r['word_id'],kind=kind,path=str(path),sha256=sha(path),old_decision=old_decision,mapping=mapping,removed=removed,synthetic=kind not in ['full_normal','crop_normal']))
 source=read(O/'word100_positive_direction_20261109/COMPARISON.json')['rows']
 for r in source:
  assert sha(r['prediction'])==r['sha256'];z=np.load(r['prediction']);add(r['clip_id'],r['kind'],z['x'],z['timestamp_us'],z['frame_id'],z['phase'],r['decision'])
 # Reuse exactly the previous deletion mappings. They are engineering controls,
 # not independently labelled whole-event omissions.
 legacy=[r for r in read(D/'EVALUATION.json')['rows'] if r['seed']==42 and r['kind'] in ['omit_last_main_then_rest','remove_middle_event']]
 for r in legacy:
  src=rows[r['clip_id']];ids=np.array(r['mapping']);raw=src['raw_x'][ids].copy();dt=int(np.median(np.diff(src['timestamp_us'])));ts=np.arange(len(raw))*dt;raw[:,-1]=np.r_[0,np.diff(ts)/1e6];old=read(r['output']);add(r['clip_id'],r['kind'],raw,ts,src['frame_id'][ids],np.asarray(old['phase_probability']),None,r['mapping'],r['removed'])
 for r in rows.values():
  if r['role']!='development':continue
  raw=common.mask_hand(common.mask_hand(r['raw_x'],0,len(r['raw_x']),0),0,len(r['raw_x']),1);add(r['clip_id'],'all_hand_missing',raw,r['timestamp_us'],r['frame_id'])
 write(P/'EVAL_INPUTS.json',dict(rows=out,count=len(out),source_existing_controls=93,deletion_controls=len(legacy),whole_hand_missing=12,reused_new_backbone_features=reuse,new_backbone_control_inferences=newruns,synthetic_scope='Deleted time ranges can shorten a repetition or remove anchors, not necessarily a complete required action. Do not report all as real learner errors.',code_sha256=sha(P/'prepare_controls.py')))
 event('controls_prepared',count=len(out),reused=reuse,new_backbone=newruns)
if __name__=='__main__':main()
