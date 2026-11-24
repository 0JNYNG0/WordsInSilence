from core import *
import shutil
def main():
 assert not (N/'INPUTS.json').exists();rows=read(U/'INPUTS.json')['rows'];out=[];original={r['clip_id']:r for r in rows if r['kind']=='full_normal'}
 for r in rows:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);target=np.array([MAP.get(int(y)-1,0) if y>=0 else -1 for y in z['sy']],np.int64);weight=z['sw'].copy();g=geometry(z['raw'],z['timestamp_us'])
  path=N/'inputs'/Path(r['path']).name;write_data=dict(g=g,target=target,weight=weight,frame_id=z['frame_id'],timestamp_us=z['timestamp_us'],raw=z['raw'])
  path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**write_data);out.append(dict(clip_id=r['clip_id'],kind=r['kind'],person=r['person'],word=r['word'],word_id=r['word_id'],role=r['role'],source_sha256=r['source_sha256'],path=str(path),sha256=sha(path),original_input=r['path'],view='full_sensor_or_coordinate_mask',labels='Inherited sparse weight1 and weak0.2; unchanged; not newly verified full boundaries'))
 # Reuse actual cropped-image observations only for current public FIT record IDs.
 # Import adapter/guard code without instantiating Candidate, detectors, camera or TCN.
 sys.path.insert(0,str(D));consumer=load('_motion_plan_consumer',D/'consumer.py');cropped=[]
 for r in read(G/'INPUT_PLAN.json')['rows']:
  cid=r['clip_id']
  if cid not in original or original[cid]['role']!='fit':continue
  p=Path(r['cache']) if r['cache'] else G/'crops'/f'{cid}.npz'
  if not p.exists():continue
  z=np.load(p);guard=consumer.quarantine(z['xy_px'],z['score'],z['timestamp_us'],640,480);converted=consumer.A.convert(guard['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],640,480);raw=converted['tcn280'];base=np.load(original[cid]['path']);assert np.array_equal(base['frame_id'],z['frame_id']);assert original[cid]['source_sha256']==r['source_sha256']
  target=np.array([MAP.get(int(y)-1,0) if y>=0 else -1 for y in base['sy']],np.int64);weight=base['sw'].copy();g=geometry(raw,z['timestamp_us']);weight[~((raw[:,187:189]>.5).all(1))]=0
  path=N/'inputs'/f'{cid}_actual_crop.npz';np.savez_compressed(path,g=g,target=target,weight=weight,frame_id=z['frame_id'],timestamp_us=z['timestamp_us'],raw=raw)
  row=dict(clip_id=cid,kind='actual_crop',person=r['person'],word=original[cid]['word'],word_id=r['word_id'],role='fit',source_sha256=r['source_sha256'],path=str(path),sha256=sha(path),original_input=str(p),original_sha256=sha(p),view='actual_cropped_image_RTM W_cached',labels='Same source frame identity; old labels not independent new ground truth')
  out.append(row);cropped.append(dict(clip_id=cid,person=r['person'],raw_source=str(p),raw_sha256=sha(p)))
 write(N/'INPUTS.json',dict(rows=out,originals=48,fit=36,historical_dev=12,actual_crop_fit=cropped,private=False,raw_detector_runs=0,shape_observation_runs=0))
 # Keep old heads and every previously frozen result immutable.
 protected=read(U/'FROZEN.json')['protected'].copy()
 for p in [U/'EVALUATION_COMPLETE.json',U/'LEGACY_COMPARE.json',D/'app.js',D/'consumer.py',U/'runs/final_attempt_supervised_42/model.pt']:protected[str(p)]=sha(p)
 contract=dict(status='FROZEN_BEFORE_TRAINING',model='random-init2Conv+BiGRU32+11state head, geometry only',epochs=100,batch=6,lr=.001,weight_decay=.01,primary_seed=42,seeds=[42,43,44],folds=[[1,2,3],[4,5,6],[7,8,9]],inner='new geometry branch only; random weights and train-only normalization, no backbone/shape latent input',final='fit01-09; DEV11-13historical regression once, not new independent generalization',training_views=['full_normal','upper_normal','actual_crop','speed075','speed125','gap2','reverse_main','omit_final_event'],semantic_groups=GROUPS,augmentation='No need to duplicate shape changes in geometry training: identical selected coordinates implies identical proposal. Existing transformed phase labels follow source identity. No hidden/absent sample is labelled a normal completed sign.',scope='No new handshape training or learned direction update; proposal+evidence integration experiment',thresholds='Reuse existing0.65 and duration gates; no DEV tuning',protected=protected,code_sha256={str(p):sha(p) for p in [N/'core.py',N/'prepare.py']},input_sha256=sha(N/'INPUTS.json'),adoption_gates=['normal warnings and withheld clips not increased','wrong-shape motion retained','absence not fabricated','important movement not thrown away by DTW','all old cases evaluated, not cherry-picked'])
 write(N/'FROZEN.json',contract);write(N/'CURRENT_EXECUTION.json',dict(status='INPUTS_READY_PENDING_VERIFICATION',training=None));event('prepared',rows=len(out),actual_crop_fit=len(cropped))
if __name__=='__main__':main()
