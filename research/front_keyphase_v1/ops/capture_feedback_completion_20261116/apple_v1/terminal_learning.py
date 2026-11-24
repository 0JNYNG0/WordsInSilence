"""Separate observable end pose from direction; never label 'half time' as wrong."""
from train_progress_axis import *
ENDS={1:70,2:77,3:75,4:98,6:57,7:81,8:93,9:81}
def terminal_features(ch):
 p=ch['body_relative_xy'];a=ch['point_available'];ids=[112,117,121,125,129,120,124,128,132,96,100,104,108]
 x=p[:,ids]-p[:,91:92];valid=a[:,ids].all(1)&a[:,91];return np.nan_to_num(x).reshape(len(p),-1).astype('float32'),valid
def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not(U/'TERMINAL_FROZEN.json').exists();rows=read(U/'INPUTS.json')['rows'];truth={r['person']:r for r in read(U/'SOURCE_TRUTH.json')['rows']};tx=[];ty=[];px=[];py=[];records=[]
 for r in rows:
  if r['role']!='fit':continue
  t=truth[r['person']];ch=dict(np.load(r['channels']));x,av=terminal_features(ch);a=t['pair_anchors'][0];b=ENDS[r['person']]
  # Reviewed original panels: above-hand versus palm passed down around fist.
  for fid,label in[(a-3,0),(a,0),(b,1),(t['pair_anchors'][-2],1),(t['pair_anchors'][-1],1)]:
   if av[fid]:tx.append(x[fid]);ty.append(label);records.append(dict(person=r['person'],frame=fid,terminal=label))
  z=np.load(U/'features'/f'p{r["person"]:02d}.npz')
  for axis,(start,end) in enumerate([t['mouth_sweep'],(a,b)]):
   v=z['mouth'if axis==0 else'pair'][start:end+1];y=1 if axis==0 else 2
   for sc in[.8,1,1.2]:
    for offset in[-.06,0,.06]:
     w=v*sc+offset
     for vals,label in[(w,y),(w[::-1],3-y),(np.repeat(w[:1],len(w),0),0),(np.repeat(w[-1:],len(w),0),0)]:px.append(axis_input(vals,axis));py.append(label)
 write(U/'TERMINAL_FROZEN.json',dict(reason='A 50-percent temporal prefix is not a semantic incomplete label; some short wipe is already complete. Replace provisional duration-fraction target with reviewed visible endpoint.',
   provisional_targets_superseded=['AXIS_FROZEN partial class','SOURCE_TRUTH wipe_sweeps approximate endpoints'],source_end_corrections=ENDS,terminal_records=records,
   steps=500,seeds=[42,43,44],confidence=dict(direction=.75,terminal=.85),
   preserves='Existing finger models, original100 backbone, gratitude and old4 unchanged',no_user_data=True,
   gate='Before visible descent must not complete; inspected terminal may complete regardless duration; full/reversed/missing/shape controls and held public person.'))
 write(U/'TERMINAL_TRAINED.json',dict(terminal=train('terminal',np.array(tx),np.array(ty)),direction=train('direction',np.array(px),np.array(py))))
 print('TERMINAL_DIRECTION',len(tx),len(px))
if __name__=='__main__':main()
