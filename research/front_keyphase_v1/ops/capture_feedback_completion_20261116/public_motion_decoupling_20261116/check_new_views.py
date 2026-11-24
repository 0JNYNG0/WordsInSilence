from common import *
from qualified_motion import QualifiedMotionLesson
from check import brief

def main(factory=QualifiedMotionLesson,output='confirmation'):
 torch.set_num_threads(2);old,new=QualifiedFormLesson(),factory();out=E/output;out.mkdir(exist_ok=True);rows=[];controls=[]
 write(out/'SOURCE_REVIEW.json',dict(source='Native RGB contact sheets REAL18 L/R',reviewer='Codex, not sign-language expert',observations='Visible separated hands at53, near at58; separated at63, near again68; lowering78. Both hands appear open/curved; no individual hidden-joint or physical contact truth.',first_only_cut_exclusive=62,role='Final unselected camera-view check; same signer front already known DEV. Not unseen-signer accuracy.',training=False))
 for r in read(E/'confirmation/INPUTS.json')['rows']:
  z=dict(np.load(r['raw']));a=old.analyze(z);b=new.analyze(z);aa,bb=brief(a),brief(b)
  passed=bb['events']==2 and not any(v=='difference'and aa['criteria'][k]!='difference'for k,v in bb['criteria'].items())and all(bb['criteria'][k]=='supported'for k,v in aa['criteria'].items()if v=='supported')
  rows.append(dict(id=r['id'],before=aa,after=bb,passed=passed));write(out/(r['id']+'.json'),dict(before=a,after=b))
  for kind in['first_only','constant']:
   v={k:x.copy()for k,x in z.items()}
   if kind=='first_only':
    for k in['xy_px','score','timestamp_us','frame_id']:v[k]=v[k][:62]
   else:v['xy_px']=np.repeat(v['xy_px'][53:54],len(v['frame_id']),axis=0)
   q=new.analyze(v);passed=len(q['events'])==(1 if kind=='first_only'else 0)and next(c for c in q['criteria']if c['id']=='repeat_order')['status']!='supported';controls.append(dict(id=r['id'],kind=kind,result=brief(q),passed=passed))
 write(out/'RESULT.json',dict(rows=rows,controls=controls,passed=all(r['passed']for r in rows+controls)));print(rows,controls,flush=True)
if __name__=='__main__':main()
