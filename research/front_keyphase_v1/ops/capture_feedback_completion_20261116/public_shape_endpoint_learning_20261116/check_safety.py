from form_candidate import *
from form_v5 import QualifiedFormLesson
from form_presentation import present
def main():
 torch.set_num_threads(2);old,new=ThanksCandidate(),QualifiedFormLesson();rows=[]
 sources=read(H/'PUBLIC_INPUTS.json')['rows']
 for r in sources:
  if not r['id'].endswith('_F')or r['person']not in[13,14,15,16,17,18]:continue
  raw=dict(np.load(r['raw']));base=old.analyze(raw)
  if len(base['events'])<2:continue
  first=base['events'][0]['arrival_index'];second=base['events'][1]['arrival_index'];n=len(raw['frame_id'])
  for kind in['first_only','second_before_arrival','missing','constant','fast','slow','scaled','long_gap']:
   z={k:v.copy()for k,v in raw.items()}
   if kind in['first_only','second_before_arrival']:
    end=first+1 if kind=='first_only'else second-1
    for k in['xy_px','score','timestamp_us','frame_id']:z[k]=z[k][:end]
   elif kind=='missing':z['xy_px'][:,91:]=np.nan
   elif kind=='constant':z['xy_px']=np.repeat(z['xy_px'][first:first+1],n,axis=0)
   elif kind in['fast','slow']:z['timestamp_us']=(z['timestamp_us']*(.65 if kind=='fast'else 1.5)).astype(np.int64)
   elif kind=='scaled':z['xy_px']=z['xy_px']*.75+np.array([90,40],np.float32)
   else:z['xy_px'][base['events'][0]['start']:first+1,112:]=np.nan
   a=old.analyze(z);b=new.analyze(z);aa={c['id']:c['status']for c in a['criteria']};bb={c['id']:c['status']for c in b['criteria']};passed=len(a['events'])==len(b['events'])and not any(v=='difference'and aa[k]!='difference'for k,v in bb.items())and all(bb[k]=='supported'for k,v in aa.items()if v=='supported')
   rows.append(dict(id=r['id'],kind=kind,before=aa,after=bb,events_before=len(a['events']),events_after=len(b['events']),passed=passed))
 write(D/'safety/RESULT.json',dict(rows=rows,passed=all(r['passed']for r in rows),synthetic=True,private_data=False));print('SAFETY',len(rows),sum(not r['passed']for r in rows),flush=True)
 q=read(D/'qualified_confirmation/confirmation/NIA_SL_WORD2633_REAL01_F_pattern.json')['after'];cards,coaching=present(q,[1,2]);c=next(c for c in cards if c['id']=='right_shape');cue=next(c for c in coaching['cues']if'right_shape'in c['card_ids']);assert c['display_status']=='difference'and cue['kind']=='shape'and c['query_spans']==next(c for c in q['criteria']if c['id']=='right_shape')['difference_spans']
 write(D/'safety/PRESENTATION.json',dict(card=c,cue=cue,passed=True));assert all(r['passed']for r in rows)
if __name__=='__main__':main()
