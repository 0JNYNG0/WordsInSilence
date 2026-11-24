from arrival_candidate import *
def brief(q):return dict(events=len(q['events']),criteria={c['id']:c['status']for c in q['criteria']})
def main(factory=ArrivalLesson,output='comparison'):
 torch.set_num_threads(2);net=factory();old=PlaneViewLesson();rows=[];controls=[];out=K/output;out.mkdir(exist_ok=True)
 for r in sources()+read(J/'confirmation/INPUTS.json')['rows']:
  raw=dict(np.load(r['raw']));a=read(J/'final_checks'/(r['id']+'.json'))['after'];b=net.analyze(raw);aa,bb=brief(a),brief(b);lost=[k for k,v in aa['criteria'].items()if v=='supported'and bb['criteria'][k]!='supported'];warning=[k for k,v in bb['criteria'].items()if v=='difference'and aa['criteria'][k]!='difference'];rows.append(dict(id=r['id'],before=aa,after=bb,lost=lost,new_warning=warning));write(out/(r['id']+'.json'),dict(before=a,after=b))
  if aa!=bb:print('CHANGE',rows[-1],flush=True)
  if not(r['id'].endswith('_F')and r.get('person')in[13,14,15,16,17,18]):continue
  first=a['events'][0]['arrival_index'];second=a['events'][1]['arrival_index'];n=len(raw['frame_id'])
  for kind in['first_only','before_second','constant','missing','fast','slow','scaled','long_gap']:
   z={k:v.copy()for k,v in raw.items()}
   if kind in['first_only','before_second']:
    end=first+1 if kind=='first_only'else second-1
    for k in['xy_px','score','timestamp_us','frame_id']:z[k]=z[k][:end]
   elif kind=='constant':z['xy_px']=np.repeat(z['xy_px'][first:first+1],n,axis=0)
   elif kind=='missing':z['xy_px'][:,91:]=np.nan
   elif kind in['fast','slow']:z['timestamp_us']=(z['timestamp_us']*(.65 if kind=='fast'else 1.5)).astype(np.int64)
   elif kind=='scaled':z['xy_px']=z['xy_px']*.75+np.array([90,40],np.float32)
   else:z['xy_px'][a['events'][0]['start']:first+1,112:]=np.nan
   x=brief(old.analyze(z));y=brief(net.analyze(z));limited=kind in['first_only','before_second','constant','missing','long_gap'];passed=y['events']<2 if limited else y['events']>=x['events']and all(y['criteria'][k]=='supported'for k,v in x['criteria'].items()if v=='supported')and not any(v=='difference'and x['criteria'][k]!='difference'for k,v in y['criteria'].items());controls.append(dict(id=r['id'],kind=kind,before=x,after=y,passed=passed))
 write(out/'RESULT.json',dict(rows=rows,controls=controls,new_warning=sum(bool(r['new_warning'])for r in rows),lost=sum(bool(r['lost'])for r in rows),control_failed=sum(not r['passed']for r in controls)));print('ARRIVAL_CHECK',len(rows),sum(bool(r['new_warning'])for r in rows),sum(bool(r['lost'])for r in rows),sum(not r['passed']for r in controls),flush=True)
if __name__=='__main__':main()
