from common import *
from adapter import *
from collections import Counter
import time, traceback

def process(row,old):
 cid=row['clip_id'];target=U/'inputs'/f'{cid}.json'
 if target.exists():
  saved=read(target);assert saved['prep_context_sha256']==sha(U/'PREP_CONTEXT.json');checkpins(saved['outputs']);return saved
 result={'clip_id':cid,'prep_context_sha256':sha(U/'PREP_CONTEXT.json'),'views':{},'outputs':{},'checks':{}}
 w,h=row['box'][2:];counts=Counter()
 for view,paths in row['views'].items():
  obs,raw,ass=(loadnp(paths[k]) for k in ('observation','raw','assignment'))
  assert np.array_equal(obs['frame_id'],raw['frame_id']) and np.array_equal(obs['timestamp_us'],raw['timestamp_us'])
  ch=loadnp(paths['channels']);arrays={};frame_lookup={int(f):i for i,f in enumerate(obs['frame_id'])}
  for temporal in ('whole','context') if row['role']=='train_candidate' else ('whole',):
   ids=np.array([frame_lookup[int(f)] for f in ch[temporal+'_frame_id']]);x=ch['x_'+temporal]
   assert len(ids)==len(x) and np.array_equal(obs['timestamp_us'][ids],ch[temporal+'_ts'])
   e,st=derive(obs,raw,ass,view,w,h,ids);validate_extra(e)
   arrays['x_'+temporal]=np.concatenate((x,e),axis=1);arrays['states_'+temporal]=st
   arrays[temporal+'_frame_id']=ch[temporal+'_frame_id'];arrays[temporal+'_ts']=ch[temporal+'_ts']
   assert np.array_equal(arrays['x_'+temporal][:,:280],x)
   counts.update(f'{view}:{temporal}:{s}:{k}:{n}' for strow in st for s,hand in enumerate(strow) for k,n in enumerate(hand))
  # Full temporal stream is saved solely for public diagnostics/review.
  e,st=derive(obs,raw,ass,view,w,h,np.arange(len(obs['frame_id'])));validate_extra(e)
  arrays['full_states']=st;arrays['frame_id']=obs['frame_id'];arrays['timestamp_us']=obs['timestamp_us']
  if view=='sensor' and cid in old:
   expected=[]
   for f in old[cid]['frames']:
    hs=[]
    for q in f['hands']:
     pos=POSITION.index(q['position']['state']);ext=q['current_extent']['status'];ov=q['current_overlap']['status'];ev=q['position']['evidence']
     hs.append([pos,1 if ext=='estimated_extent_inside_frame' else 2 if ext=='estimated_extent_crosses_frame' else 0,1 if ov=='no_current_2d_intersection' else 2 if ov=='current_2d_intersection_suspected' else 0,2 if q['overlap']['proximity_hint'] else 1 if 'other_palm_distance_shoulders' in ev and 'history_distance_shoulders' in ev else 0])
    expected.append(hs)
   assert np.array_equal(st,np.array(expected)),(cid,'old diagnostic changed')
   result['checks']['known_sensor_hand_states_preserved']=int(st.shape[0]*2)
  # Hidden-value counterfactual on every masked clip; raw hand evidence is never used.
  if view=='masked':
   mutated={k:v.copy() for k,v in obs.items()};rr={k:v.copy() for k,v in raw.items()}
   hidden=~obs['usable'];mutated['xy_px'][hidden]=123456.0
   rr['pose_conf'][~obs['usable'][:,:33]]=0.999
   rr['hand_xy_px'][:]=42.;rr['raw_count'][:]=0
   ma={k:v.copy() for k,v in ass.items()};ma['sides'][:]=-1
   ee,ss=derive(mutated,rr,ma,view,w,h,np.arange(len(obs['frame_id'])))
   assert np.array_equal(e,ee) and np.array_equal(st,ss),(cid,'hidden leakage')
   result['checks']['hidden_point_mutations']=int(hidden.sum())
  # Future observations cannot alter past feature rows; no global mutable state.
  end=max(1,len(obs['frame_id'])//2)
  _,prefix=derive(obs,raw,ass,view,w,h,np.arange(end))
  assert np.array_equal(st[:end],prefix),(cid,view,'future dependence')
  out=U/'inputs'/f'{cid}_{view}.npz'
  if out.exists():raise RuntimeError(f'Orphan partial output {out}; preserve evidence before recovery')
  np.savez_compressed(out,**arrays);result['outputs'][str(out)]=sha(out);result['views'][view]=str(out)
 result['state_counts']=dict(counts);write(target,result,new=True);return result

def main():
 c=read(U/'PREP_CONTEXT.json');checkpins(c['pins'])
 codepins={str(U/name):sha(U/name) for name in ('common.py','adapter.py','prepare.py')}
 codecontext=U/'PREP_CODE.json'
 if codecontext.exists():checkpins(read(codecontext)['pins'])
 else:write(codecontext,{'at':now(),'pins':codepins,'masked_policy':'No full-frame raw assignment/hidden-point values in supplemental channels; insufficient evidence remains unknown.'},new=True)
 (U/'inputs').mkdir(exist_ok=True)
 old={r['clip_id']:r for r in read(CURRENT/'OUTPUT.json')['records']}
 lock=U/'PREP_RUNNING.json';write(lock,{'pid':os.getpid(),'at':now()},new=True)
 started=time.monotonic();completed=[]
 try:
  for r in c['records']:
   completed.append(process(r,old))
   if len(completed)%25==0:
    status={'at':now(),'stage':'PREPARING','completed':len(completed),'total':1199,'seconds':time.monotonic()-started,'training':0};write(U/'STATUS.json',status);print(json.dumps(status),flush=True)
  checkpins(c['pins']);checkpins(codepins)
  write(U/'PREP_RESULT.json',{'status':'COMPLETE_PENDING_INDEPENDENT_AND_VISUAL_REVIEW','records':1199,'regression_states':sum(r['checks'].get('known_sensor_hand_states_preserved',0) for r in completed),'hidden_point_mutations':sum(r['checks'].get('hidden_point_mutations',0) for r in completed),'seconds':time.monotonic()-started,'record_pins':{str(U/'inputs'/f"{r['clip_id']}.json"):sha(U/'inputs'/f"{r['clip_id']}.json") for r in completed},'training_allowed':False},new=True)
  lock.unlink();print('PREP_COMPLETE',flush=True)
 except BaseException:
  err={'at':now(),'traceback':traceback.format_exc(),'completed':len(completed),'training':0}
  if not (U/'FIRST_PREP_ERROR.json').exists():write(U/'FIRST_PREP_ERROR.json',err,new=True)
  raise
if __name__=='__main__':main()
