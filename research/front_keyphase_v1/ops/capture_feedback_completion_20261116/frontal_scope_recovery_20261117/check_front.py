from frontal_candidate import *
import importlib.util
from visible_lesson import adapter

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 torch.set_num_threads(2);loaded=[];original=torch.load
 def trace(p,*a,**kw):loaded.append(str(Path(p).resolve()));return original(p,*a,**kw)
 torch.load=trace
 net=FrontalLesson();weights=list(dict.fromkeys(loaded));rows=[];out=N/'candidate';out.mkdir(exist_ok=True)
 for r in read(N/'INPUTS.json')['rows']:
  require_front(r)
  if r['word']!='WORD1290':continue
  a=read(r['baseline'])['after'];b=net.analyze(dict(np.load(r['raw'])));aa={c['id']:c['status']for c in a['criteria']};bb={c['id']:c['status']for c in b['criteria']}
  row=dict(id=r['id'],before=aa,after=bb,events_before=len(a['events']),events_after=len(b['events']),lost=[k for k,v in aa.items()if v=='supported'and bb[k]!='supported'],new_warning=[k for k,v in bb.items()if v=='difference'and aa[k]!='difference'],seconds=b['seconds']);rows.append(row);write(out/(r['id']+'.json'),dict(before=a,after=b))
 write(out/'NORMAL.json',dict(rows=rows,loaded_weights=[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())for p in weights],full=sum(all(v=='supported'for v in r['after'].values())for r in rows),failed=sum(bool(r['lost']or r['new_warning'])for r in rows)))
 print('NORMAL',read(out/'NORMAL.json')['full'],read(out/'NORMAL.json')['failed'],flush=True)
 # Independent criteria and temporal controls are frozen before evaluation.
 m=load('frontal_distinction_checks',H/'check_distinctions.py');m.HERE=out;old_sources=m.sources
 def front_sources():
  rr=old_sources()
  for r in rr:require_front(dict(r,clip_id=r.get('clip_id',f"NIA_SL_WORD1290_REAL{r['person']:02d}_F")))
  return rr
 m.sources=front_sources;m.main('distinctions',FrontalLesson)
 previous={(r['person'],r['kind']):r for r in read(M/'final_regression/distinctions/RESULT.json')['rows']};dist=read(out/'distinctions/RESULT.json');lost=[]
 for r in dist['rows']:
  if r['expected']=='diagnostic':continue
  missing=set(previous[r['person'],r['kind']]['differences'])-set(r['differences'])
  if missing:lost.append(dict(person=r['person'],kind=r['kind'],lost=sorted(missing)))
 write(out/'ERROR_PRESERVATION.json',dict(lost=lost));print('ERROR_LOSS',lost,flush=True)
 controls=[]
 for r in read(N/'INPUTS.json')['rows']:
  if r['word']!='WORD1290'or int(r['id'].split('REAL')[1][:2])not in[13,14,15,16,17,18]:continue
  z=dict(np.load(r['raw']));a=read(out/(r['id']+'.json'))['after'];first=a['events'][0]['arrival_index'];second=a['events'][1]['arrival_index'];n=len(z['frame_id'])
  for kind in['first_only','before_second','constant','missing','fast','slow','scaled','long_gap']:
   v={k:x.copy()for k,x in z.items()}
   if kind in['first_only','before_second']:
    end=first+1 if kind=='first_only'else second-1
    for k in['xy_px','score','timestamp_us','frame_id']:v[k]=v[k][:end]
   elif kind=='constant':v['xy_px']=np.repeat(v['xy_px'][first:first+1],n,axis=0)
   elif kind=='missing':v['xy_px'][:,91:]=np.nan
   elif kind in['fast','slow']:v['timestamp_us']=(v['timestamp_us']*(.65 if kind=='fast'else 1.5)).astype(np.int64)
   elif kind=='scaled':v['xy_px']=v['xy_px']*.75+np.array([90,40],np.float32)
   else:v['xy_px'][a['events'][0]['start']:first+1,112:]=np.nan
   b=net.analyze(v);x={c['id']:c['status']for c in a['criteria']};y={c['id']:c['status']for c in b['criteria']};limited=kind in['first_only','before_second','constant','missing','long_gap'];passed=len(b['events'])<2 if limited else len(b['events'])>=len(a['events'])and all(y[k]==val for k,val in x.items()if val=='supported')and not any(val=='difference'and x[k]!='difference'for k,val in y.items());controls.append(dict(id=r['id'],kind=kind,after=y,events=len(b['events']),passed=passed))
 write(out/'TEMPORAL.json',dict(rows=controls,failed=sum(not r['passed']for r in controls)));print('TEMPORAL',len(controls),sum(not r['passed']for r in controls),flush=True)
 # Existing coarse-shape source anchors, separate from exact finger labels.
 shape=read(G/'public_motion_decoupling_20261116/regression/SHAPE.json')['rows'];extra={r['clip_id']:r for r in read(O/'word100_feature_expansion_20261113/INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']};cache={};checks=[]
 for r in shape:
  cid=r['clip'];require_front(dict(id=cid));key=cid,r['side']
  if key not in cache:
   path=extra[cid]['raw']if cid in extra else read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json')['raw_path'];require_front(dict(id=cid,raw=path));z=dict(np.load(path));ch=adapter.convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']));ss=net.shape.sequence(ch,r['side']);cache[key]=(ch['frame_id'],ss['states'])
  ids,ss=cache[key];i=int(np.flatnonzero(ids==r['frame'])[0]);checks.append(dict(clip=cid,frame=r['frame'],side=r['side'],truth=r['truth'],before=r['after'],after=int(ss[i])))
 nw=sum(r['after']>=0 and r['after']!=r['truth']and(r['before']<0 or r['before']==r['truth'])for r in checks);lc=sum(r['before']==r['truth']and r['after']!=r['truth']for r in checks);write(out/'SHAPE.json',dict(rows=checks,new_wrong=nw,lost_correct=lc));print('SHAPE',len(checks),nw,lc,flush=True)
 write(out/'RESULT.json',dict(normal_failed=read(out/'NORMAL.json')['failed'],shape_new_wrong=nw,shape_lost_correct=lc,error_loss=lost,distinction_failed=dist['failed'],temporal_failed=sum(not r['passed']for r in controls),training_runs=3))
if __name__=='__main__':main()
