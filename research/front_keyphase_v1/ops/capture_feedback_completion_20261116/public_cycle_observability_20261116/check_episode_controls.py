from cycle_candidate import *
from verify_scope import verify
def main(factory=CycleLesson,output='QUALIFIED_EPISODE_CONTROLS.json'):
 torch.set_num_threads(2);old=TemporalVisibilityLesson();new=factory();by={r['id']:r for r in all_sources()};rows=[]
 # Native public truncations before the independently visible second arrival,
 # stationary geometry, and missing keypoint channels. No private clips.
 for suffix,cut,hold in [('06_L',51,47),('06_R',51,46),('13_R',65,59),('16_L',61,51)]:
  cid='NIA_SL_WORD1290_REAL'+suffix;z=dict(np.load(by[cid]['raw']));n=len(z['frame_id'])
  for kind in ['partial','hold','missing','far','slow','fast','rescale']:
   raw={k:v.copy()for k,v in z.items()}
   if kind=='partial':
    for k in ['xy_px','score','timestamp_us','frame_id']:raw[k]=raw[k][:cut]
   elif kind=='hold':raw['xy_px']=np.repeat(raw['xy_px'][hold:hold+1],n,axis=0)
   elif kind=='missing':raw['xy_px'][:,91:]=np.nan
   elif kind=='far':raw['xy_px'][:,112:,0]+=200
   elif kind in ['slow','fast']:raw['timestamp_us']=(raw['timestamp_us']*(1.5 if kind=='slow'else .65)).astype(np.int64)
   else:raw['xy_px']=raw['xy_px']*.75+np.array([90,40],np.float32)
   a,b=old.analyze(raw),new.analyze(raw);aa={c['id']:c['status']for c in a['criteria']};bb={c['id']:c['status']for c in b['criteria']};limited=kind in ['partial','hold','missing','far'];detail=verify(a,b);passed=len(b['events'])<2 if limited else detail['passed'];rows.append(dict(id=cid,kind=kind,events_before=len(a['events']),events_after=len(b['events']),before=aa,after=bb,passed=passed,coverage_extended_unknown=detail['coverage_extended_unknown']))
 write(M/output,dict(rows=rows,passed=all(r['passed']for r in rows)));print('EPISODE_CONTROLS',sum(not r['passed']for r in rows),flush=True)
if __name__=='__main__':main()
