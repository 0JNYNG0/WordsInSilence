from common import *
from palm_geometry import PalmMotionLesson
from check import brief

def main():
 torch.set_num_threads(2);old,new=QualifiedFormLesson(),PalmMotionLesson();out=E/'unselected_12';rows=[];controls=[]
 write(out/'SOURCE_REVIEW.json',dict(reviewer='Codex native RGB review, not expert',observation='Both hands visibly open/curved; separated48, near53, separated58; first near43. Lowering78. First-only cut exclusive48.',role='These side views not previously used for candidate selection. Front signer is known; not unseen-signer accuracy.',gate='No newly attributed normal error or lost support; do not exceed two observed cycles; cut/static must not complete',training=False))
 for r in read(out/'INPUTS.json')['rows']:
  z=dict(np.load(r['raw']));a=old.analyze(z);b=new.analyze(z);aa,bb=brief(a),brief(b)
  passed=aa['events']<=bb['events']<=2 and not any(v=='difference'and aa['criteria'][k]!='difference'for k,v in bb['criteria'].items())and all(bb['criteria'][k]=='supported'for k,v in aa['criteria'].items()if v=='supported')
  rows.append(dict(id=r['id'],before=aa,after=bb,passed=passed));write(out/(r['id']+'.json'),dict(before=a,after=b))
  for kind in['first_only','constant']:
   v={k:x.copy()for k,x in z.items()}
   if kind=='first_only':
    for k in['xy_px','score','timestamp_us','frame_id']:v[k]=v[k][:48]
   else:v['xy_px']=np.repeat(v['xy_px'][48:49],len(v['frame_id']),axis=0)
   q=new.analyze(v);passed=len(q['events'])<2 and next(c for c in q['criteria']if c['id']=='repeat_order')['status']!='supported';controls.append(dict(id=r['id'],kind=kind,result=brief(q),passed=passed))
 write(out/'RESULT.json',dict(rows=rows,controls=controls,passed=all(r['passed']for r in rows+controls)));print(rows,controls,flush=True)
if __name__=='__main__':main()
