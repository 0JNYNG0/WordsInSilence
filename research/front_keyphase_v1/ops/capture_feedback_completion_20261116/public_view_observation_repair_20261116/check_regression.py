from view_common import *
from view_candidate import ViewLesson
import importlib.util

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main(factory=ViewLesson,output='regression'):
 torch.set_num_threads(2);net=factory();old=PalmMotionLesson();out=J/output;out.mkdir(exist_ok=True)
 rows=read(E/'regression/SHAPE.json')['rows'];extra={r['clip_id']:r for r in read(O/'word100_feature_expansion_20261113/INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']};cache={};checks=[]
 for r in rows:
  cid=r['clip'];key=cid,r['side']
  if key not in cache:
   path=extra[cid]['raw']if cid in extra else read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json')['raw_path'];ch=channels(dict(np.load(path)));net.shape.sequence(ch,r['side']);ss,_,_=net.form_states(ch,r['side']);cache[key]=(ch['frame_id'],ss)
  ids,ss=cache[key];i=int(np.flatnonzero(ids==r['frame'])[0]);checks.append(dict(clip=cid,frame=r['frame'],side=r['side'],truth=r['truth'],before=r['after'],after=int(ss[i])))
 nw=sum(r['after']>=0 and r['after']!=r['truth']and(r['before']<0 or r['before']==r['truth'])for r in checks);lc=sum(r['before']==r['truth']and r['after']!=r['truth']for r in checks)
 write(out/'SHAPE.json',dict(rows=checks,new_wrong=nw,lost_correct=lc));print('SHAPE',len(checks),nw,lc,flush=True)
 m=load('view_distinctions',H/'check_distinctions.py');m.HERE=out;m.main('distinctions',factory)
 previous={(r['person'],r['kind']):r for r in read(E/'palm_regression/distinctions/RESULT.json')['rows']};dist=read(out/'distinctions/RESULT.json');lost=[]
 for r in dist['rows']:
  p=previous[r['person'],r['kind']]
  if r['expected']=='diagnostic':continue
  missing=set(p['differences'])-set(r['differences'])
  if missing:lost.append(dict(person=r['person'],kind=r['kind'],lost=sorted(missing)))
 write(out/'DISTINCTION_PRESERVATION.json',dict(lost=lost));print('KNOWN_ERROR_LOSS',lost,flush=True)
 m=load('view_patterns',D/'check_confirmation.py');m.D=out;m.FinalFormLesson=factory;m.ThanksCandidate=PalmMotionLesson;write(out/'confirmation/INPUTS.json',dict(rows=[]));m.main()
 controls=[]
 for r in sources():
  if not(r['id'].endswith('_F')and r.get('person')in[13,14,15,16,17,18]):continue
  z=dict(np.load(r['raw']));a=baseline(r['id']);first=a['events'][0]['arrival_index'];second=a['events'][1]['arrival_index'];n=len(z['frame_id'])
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
   oa=old.analyze(v);nb=net.analyze(v);x={c['id']:c['status']for c in oa['criteria']};y={c['id']:c['status']for c in nb['criteria']};limited=kind in['first_only','before_second','constant','missing','long_gap'];passed=len(nb['events'])<2 if limited else len(nb['events'])>=len(oa['events'])and all(y[k]==val for k,val in x.items()if val=='supported')and not any(val=='difference'and x[k]!='difference'for k,val in y.items());controls.append(dict(id=r['id'],kind=kind,before=x,after=y,events_before=len(oa['events']),events_after=len(nb['events']),passed=passed))
 write(out/'TEMPORAL.json',dict(rows=controls,failed=sum(not r['passed']for r in controls)));print('TEMPORAL',len(controls),sum(not r['passed']for r in controls),flush=True)
 write(out/'RESULT.json',dict(shape_new_wrong=nw,shape_lost_correct=lc,known_error_loss=lost,distinction_failed=dist['failed'],temporal_failed=sum(not r['passed']for r in controls)))

if __name__=='__main__':main()
