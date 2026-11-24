from visible_lesson import *
sys.path.insert(0,str(V.parent));from diagnose_terminal import sources

def main(output_name='distinction_checks',factory=VisibleLesson):
 torch.set_num_threads(2);out=V/output_name;out.mkdir(exist_ok=True);assert not(out/'FROZEN.json').exists()
 candidates=read(S/'INPUTS.json')['rows'];donors=[];seen=set()
 for r in candidates:
  if r['role']!='fit'or r['targets'][1:]!=[0,0,0,0]:continue
  key=r['side'],r['word'];p=O/'word100_rtmw_joint_learning_20261108/records'/f'{r["clip_id"]}.json'
  if key in seen or not p.exists():continue
  if sum(x['side']==r['side']for x in donors)>=3:continue
  donors.append(dict(r,raw_path=read(p)['raw_path']));seen.add(key)
 assert len(donors)>=4
 write(out/'FROZEN.json',dict(donors=donors,
  scope='Synthetic keypoint challenge using public source-labelled flexed hand poses; NOT actual learner-error accuracy.',
  cases=['replace_L/R_flexed_pose','swap_visible_hand_planes','sideways_target_offset','reverse_action','one_frame_shape_blip'],
  gate='Persistent flexed or swapped-plane/off-target/reversed task must not pass; one frame disturbance must not be an attributed error. Report unknown separately.',
  code_sha=sha(V/'visible_lesson.py'),data_identity_input=False))
 net=factory();results=[]
 for r in sources():
  raw=dict(np.load(r['raw_path']));n=len(raw['frame_id']);baseline=net.analyze(raw)
  if not baseline['events']:continue
  lo=baseline['events'][0]['start'];hi=baseline['events'][-1]['end'];rawch=adapter.load(r['raw_path'])
  specs=[('swap_planes',None),('target_away',None),('reverse_main',None),('one_frame_blip',None)]+[('flex_'+d['side']+'_'+d['word'],d)for d in donors]
  for kind,donor in specs:
   z={k:v.copy()for k,v in raw.items()};xy=z['xy_px']
   if donor:
    dr=dict(np.load(donor['raw_path']));di=int(np.flatnonzero(dr['frame_id']==donor['frame'])[0]);st=91 if donor['side']=='L'else 112
    h=dr['xy_px'][di,st:st+21];h=(h-h[0])/np.linalg.norm(h[9]-h[0]);sz=np.linalg.norm(xy[:,st+9]-xy[:,st],axis=1)
    xy[lo:hi+1,st:st+21]=xy[lo:hi+1,st:st+1]+h[None]*sz[lo:hi+1,None,None]
   elif kind=='swap_planes':
    for a,b in[(91,112),(112,91)]:
     h=raw['xy_px'][:,b:b+21]-raw['xy_px'][:,b:b+1];h[:,:,0]*=-1
     scale=np.linalg.norm(raw['xy_px'][:,a+9]-raw['xy_px'][:,a],axis=1)/np.maximum(np.linalg.norm(h[:,9],axis=1),1e-5)
     xy[lo:hi+1,a:a+21]=raw['xy_px'][lo:hi+1,a:a+1]+h[lo:hi+1]*scale[lo:hi+1,None,None]
   elif kind=='target_away':
    width=np.linalg.norm(xy[:,5]-xy[:,6],axis=1);xy[:,112:133,0]+=width[:,None]*1.5
   elif kind=='reverse_main':
    for k in['xy_px','score']:z[k][lo:hi+1]=z[k][lo:hi+1][::-1].copy()
   elif kind=='one_frame_blip':xy[(lo+hi)//2,112:133]=np.nan
   q=net.analyze(z);supported=all(c['status']=='supported'for c in q['criteria']);differences=[c['id']for c in q['criteria']if c['status']=='difference']
   # Time reversing repeated hand-taps can remain a valid repeated gesture;
   # it is diagnostic only, never invented error ground truth.
   expected='diagnostic'if kind=='reverse_main'else'no_attributed_error'if kind=='one_frame_blip'else'not_complete'
   passed=None if expected=='diagnostic'else not differences if expected=='no_attributed_error'else not supported
   results.append(dict(person=r['person'],kind=kind,expected=expected,check=passed,supported=supported,differences=differences,criteria=q['criteria'],events=q['events'],seconds=q['seconds']))
 write(out/'RESULT.json',dict(rows=results,passed=sum(r['check']is True for r in results),failed=sum(r['check']is False for r in results),diagnostic=sum(r['check']is None for r in results),web_admitted=False))
 print('DISTINCTIONS',sum(r['check']is True for r in results),sum(r['check']is False for r in results))
 for r in results:
  if r['check']is False:print('FAIL',r['person'],r['kind'],{c['id']:c['status']for c in r['criteria']})
if __name__=='__main__':main()
