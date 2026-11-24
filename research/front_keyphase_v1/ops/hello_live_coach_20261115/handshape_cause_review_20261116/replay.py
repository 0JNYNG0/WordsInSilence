from diagnose import *
sys.path.insert(0,str(U));from coach import Coach
sys.path.insert(0,str(U/'usability_repair_20261116'));from public_compare import cases
def main():
 raw=load(U/'source/reference.npz');c=Coach();rows=[]
 examples=cases(raw)
 for name,path in [('letterbox',U/'seated_diagnosis_20261116/public_video_640_letterbox.npz'),('upper',U/'seated_diagnosis_20261116/public_video_upper_body_640.npz'),('heldout_view',U/'seated_diagnosis_20261116/heldout_crop.npz')]:examples.append((name,load(path),[True,True],True))
 z={k:v.copy()for k,v in raw.items()};idx=np.r_[np.arange(52),np.repeat(51,480),np.arange(52,len(raw['timestamp_us']))]
 z['xy_px']=raw['xy_px'][idx].copy();z['score']=raw['score'][idx].copy()
 for j in range(52,532):z['xy_px'][j,:,1]+=np.sin(j*.17)*.5
 z['frame_id']=np.arange(len(idx));z['timestamp_us']=np.rint(z['frame_id']*1e6/30).astype(np.int64);examples.append(('delayed_repeat',z,[True,True],True))
 for h,start in [('L',91),('R',112)]:
  z={k:v.copy()for k,v in raw.items()};z['xy_px'][49:81,start+5:start+9]=z['xy_px'][49:81,start+5:start+6]
  examples.append((h+'_collapsed_index_while_repeating',z,[True,False],True))
 for name,z,expected,_ in examples:
  c.reset();counts=collections.Counter();times=[];completion=[None,None];detail=[]
  for i,t in enumerate(z['timestamp_us']):
   ob=c.update(z['xy_px'][i],z['score'][i],int(t),int(z['source_width']),int(z['source_height']));counts[ob['diagnostic']['code']]+=1
   for j,d in enumerate(ob['completed']):
    if d and completion[j]is None:completion[j]=i
   if 'shape_diagnostic'in ob:detail.append(dict(frame=i,diagnostic=ob['diagnostic'],shape=ob['shape_diagnostic']))
  row=dict(case=name,completed=c.completed.copy(),expected=expected,passed=c.completed==expected,stage_received_frame=completion,codes=dict(counts));rows.append(row);print(name,row['passed'],flush=True)
  if 'collapsed'in name:write(V/(name+'_trace.json'),detail)
 write(V/'REPLAY.json',dict(passed=all(r['passed']for r in rows),rows=rows));assert all(r['passed']for r in rows)
if __name__=='__main__':main()
