from common import *
from qualified_motion import QualifiedMotionLesson
import importlib.util

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 torch.set_num_threads(2);net=QualifiedMotionLesson();out=E/'regression';out.mkdir(exist_ok=True)
 rows=read(D/'qualified_checks/RESULT.json')['anchors'];extra={r['clip_id']:r for r in read(O/'word100_feature_expansion_20261113/INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']};cache={};checks=[]
 for r in rows:
  cid=r['clip'];key=cid,r['side']
  if key not in cache:
   path=extra[cid]['raw']if cid in extra else read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json')['raw_path'];ch=channels(dict(np.load(path)));net.shape.sequence(ch,r['side']);ss,_,_=net.form_states(ch,r['side']);cache[key]=(ch['frame_id'],ss)
  ids,ss=cache[key];i=int(np.flatnonzero(ids==r['frame'])[0]);checks.append(dict(clip=cid,frame=r['frame'],side=r['side'],truth=r['truth'],before=r['after'],after=int(ss[i])))
 nw=sum(r['after']>=0 and r['after']!=r['truth']and(r['before']<0 or r['before']==r['truth'])for r in checks);lc=sum(r['before']==r['truth']and r['after']!=r['truth']for r in checks)
 write(out/'SHAPE.json',dict(rows=checks,new_wrong=nw,lost_correct=lc));print('SHAPE',len(checks),nw,lc,flush=True)
 m=load('motion_distinctions',H/'check_distinctions.py');m.HERE=out;m.main('distinctions',QualifiedMotionLesson)
 m=load('motion_patterns',D/'check_confirmation.py');m.D=out;m.FinalFormLesson=QualifiedMotionLesson;m.ThanksCandidate=QualifiedFormLesson
 write(out/'confirmation/INPUTS.json',dict(rows=[]));m.main()
if __name__=='__main__':main()
