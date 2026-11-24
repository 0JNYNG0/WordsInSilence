from final_temporal import *

def brief(q):return dict(events=len(q['events']),criteria={c['id']:c['status']for c in q['criteria']})
def main():
 torch.set_num_threads(2);old,new=PreservedArrivalLesson(),TemporalVisibilityLesson();rows=[];controls=[];out=L/'verified_extra';out.mkdir(exist_ok=True)
 write(out/'REVIEW.json',dict(reviewer='Codex native public RGB review',evidence='*_main.jpg frames82..126',observation='Visible upright active open/curved palm approaches supporting hand repeatedly, separated holds and ending arms down. No individual hidden digit/physical contact truth inferred.',scope='Known FIT person4 but new unused camera views. No independent-person accuracy claim.'))
 for r in read(L/'confirmation/INPUTS.json')['rows']:
  raw=dict(np.load(r['raw']));a,b=old.analyze(raw),new.analyze(raw);aa,bb=brief(a),brief(b);passed=all(bb['criteria'][k]=='supported'for k,v in aa['criteria'].items()if v=='supported')and not any(v=='difference'and aa['criteria'][k]!='difference'for k,v in bb['criteria'].items())and all(e in b['events']for e in a['events']);rows.append(dict(id=r['id'],before=aa,after=bb,passed=passed));write(out/(r['id']+'.json'),dict(before=a,after=b))
 for r in current_sources():
  if r['id']not in[f'NIA_SL_WORD1290_REAL{p:02d}_{v}'for p in[12,18]for v in['L','R']]:continue
  z=dict(np.load(r['raw']));cut=48 if 'REAL12_'in r['id']else 62;hold=48 if 'REAL12_'in r['id']else 53;n=len(z['frame_id'])
  for kind in ['partial','constant']:
   raw={k:v.copy()for k,v in z.items()}
   if kind=='partial':
    for k in ['xy_px','score','timestamp_us','frame_id']:raw[k]=raw[k][:cut]
   else:raw['xy_px']=np.repeat(raw['xy_px'][hold:hold+1],n,axis=0)
   q=new.analyze(raw);controls.append(dict(id=r['id'],kind=kind,after=brief(q),passed=len(q['events'])<2))
 # Obscure individual ring/little digits only in a public main action. A coarse
 # palm update must not manufacture positive observations for missing digits.
 missing=[]
 for r in current_sources():
  if r['id']not in ['NIA_SL_WORD1290_REAL11_L','NIA_SL_WORD1290_REAL15_R']:continue
  z=dict(np.load(r['raw']));q=read(L/'audit'/(r['id']+'.json'))['output'];start=q['events'][0]['start']
  for hand,offset in [('L',91),('R',112)]:
   for size in [1,3,8]:
    raw={k:v.copy()for k,v in z.items()};ids=np.arange(start,start+size);raw['xy_px'][start:start+size,offset+13:offset+21]=np.nan;raw['score'][start:start+size,offset+13:offset+21]=0;a=old.analyze(raw);b=new.analyze(raw);sa=np.array(a['hand_form_evidence'][hand]['states']);sb=np.array(b['hand_form_evidence'][hand]['states']);ob=new.shape.observations[hand];passed=not np.any((sb[ids]==0)&(sa[ids]!=0))and not ob['available'][ids,3:].any();missing.append(dict(id=r['id'],hand=hand,frames=size,passed=bool(passed),before=sa[ids].tolist(),after=sb[ids].tolist()))
 write(out/'RESULT.json',dict(rows=rows,controls=controls,missing=missing,passed=all(r['passed']for r in rows+controls+missing),private_data=False));print('EXTRA',read(out/'RESULT.json')['passed'],rows,flush=True)
if __name__=='__main__':main()
