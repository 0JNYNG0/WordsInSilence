from form_candidate import *
from form_v2 import FinalFormLesson
from train_shape import S
import importlib.util
def main(output_name='final_checks',factory=FinalFormLesson):
 torch.set_num_threads(2);net=factory();out=D/output_name;out.mkdir(exist_ok=True)
 result=read(D/'comparison/RESULT.json');rows=[];checks=[]
 sources={r['id']:r for r in read(H/'PUBLIC_INPUTS.json')['rows']+read(H/'public_camera/INPUTS.json')['rows']+read(H/'public_views/INPUTS.json')['rows']+read(H/'confirmation_views/INPUTS.json')['rows']}
 for r in result['normal']:
  src=sources[r['id']];q=net.analyze(dict(np.load(src['raw'])));after=dict(events=len(q['events']),criteria={c['id']:c['status']for c in q['criteria']});before=r['before']
  rows.append(dict(id=r['id'],before=before,after=after,passed=after['events']==before['events']and all(after['criteria'][k]==v for k,v in before['criteria'].items()if v!='unknown')and not any(v=='difference'and before['criteria'][k]!='difference'for k,v in after['criteria'].items())))
  write(out/(r['id']+'.json'),q)
 extras={r['clip_id']:r for r in read(O/'word100_feature_expansion_20261113/INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']};cache={}
 for r in result['shape_anchors']:
  cid=r['clip'];key=cid,r['side']
  if key not in cache:
   path=extras[cid]['raw']if cid in extras else read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json')['raw_path'];raw=dict(np.load(path));ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));net.shape.sequence(ch,r['side']);s,_,_=net.form_states(ch,r['side']);cache[key]=(ch['frame_id'],s)
  ids,s=cache[key];i=int(np.flatnonzero(ids==r['frame'])[0]);checks.append(dict(r,after=int(s[i])))
 stats=dict(new_wrong=sum(x['after']>=0 and x['after']!=x['truth']and(x['before']<0 or x['before']==x['truth'])for x in checks),lost_correct=sum(x['before']==x['truth']and x['after']!=x['truth']for x in checks))
 write(out/'RESULT.json',dict(normal=rows,anchors=checks,**stats,private_data=False));print('FORM_V2',stats,'normal_failed',sum(not r['passed']for r in rows),'changed',[r['id']for r in rows if r['before']!=r['after']],flush=True)
 spec=importlib.util.spec_from_file_location('specific_distinctions',H/'check_distinctions.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.HERE=out;m.main('distinctions',factory)
if __name__=='__main__':main()
