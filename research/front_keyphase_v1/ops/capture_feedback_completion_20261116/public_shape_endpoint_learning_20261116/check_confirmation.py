from form_candidate import *
from form_v2 import FinalFormLesson
def main():
 torch.set_num_threads(2);old,new=ThanksCandidate(),FinalFormLesson();rows=[]
 write(D/'confirmation/REVIEW.json',dict(source='Native synchronized camera contact sheets reviewed before final candidate outputs',reviewer='Codex, not sign-language expert',observed='Visible right open palm and left supporting hand; near poses around frames53 and63; no clenched-fist stage claimed',labels='Normal source pose evidence; physical contact and hidden joint state not annotated'))
 for r in read(D/'confirmation/INPUTS.json')['rows']:
  raw=dict(np.load(r['raw']));a=old.analyze(raw);b=new.analyze(raw);aa={c['id']:c['status']for c in a['criteria']};bb={c['id']:c['status']for c in b['criteria']}
  rows.append(dict(id=r['id'],before=aa,after=bb,events_before=len(a['events']),events_after=len(b['events']),passed=all(bb[k]==v for k,v in aa.items()if v!='unknown')and not any(v=='difference'and aa[k]!='difference'for k,v in bb.items())))
 write(D/'confirmation/RESULT.json',dict(rows=rows,passed=all(r['passed']for r in rows),physical_camera=False));print('NEW_VIEWS',rows,flush=True)
 # Distinct public finger patterns inserted in a normal movement. This is a
 # synthetic discrimination check, not actual learner-error accuracy.
 src=next(r for r in read(H/'PUBLIC_INPUTS.json')['rows']if r['id']=='NIA_SL_WORD1290_REAL13_F');raw=dict(np.load(src['raw']));q=old.analyze(raw);lo=q['events'][0]['start'];hi=q['events'][-1]['end'];controls=[]
 for r in read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']:
  if r['role']!='fit'or r['word']not in['WORD2633','WORD2637']:continue
  if sum(x['word']==r['word']for x in controls)>=3:continue
  donor=dict(np.load(r['raw']));i=r['anchors'][len(r['anchors'])//2];h=donor['xy_px'][i,112:133];h=(h-h[0])/np.linalg.norm(h[9]-h[0]);z={k:v.copy()for k,v in raw.items()};xy=z['xy_px'];sz=np.linalg.norm(xy[:,121]-xy[:,112],axis=1);xy[lo:hi+1,112:133]=xy[lo:hi+1,112:113]+h[None]*sz[lo:hi+1,None,None]
  a=old.analyze(z);b=new.analyze(z);aa=next(c for c in a['criteria']if c['id']=='right_shape')['status'];bb=next(c for c in b['criteria']if c['id']=='right_shape')['status'];controls.append(dict(word=r['word'],source=r['clip_id'],before=aa,after=bb,passed=bb!='supported'and not(aa=='difference'and bb!='difference')))
  write(D/'confirmation'/(r['clip_id']+'_pattern.json'),dict(before=a,after=b,source_anchor=i,replace_span=[lo,hi]))
 write(D/'confirmation/PATTERN_CHECK.json',dict(rows=controls,passed=all(r['passed']for r in controls),scope='Synthetic public distinct-pattern control; no user footage'));print('PATTERNS',controls,flush=True)
 assert all(r['passed']for r in rows+controls)
if __name__=='__main__':main()
