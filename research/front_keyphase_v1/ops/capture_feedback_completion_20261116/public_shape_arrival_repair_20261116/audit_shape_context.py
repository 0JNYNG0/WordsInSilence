"""Find public review candidates near known open-hand anchors; assign NO labels."""
from audit_public import *
from view_conditioned_shape import ViewConditionedShape
def main():
 torch.set_num_threads(2);rows=read(O/'word100_finger_repair_20261115/INPUTS.json')['rows']
 public=[r for r in rows if r['targets'][1:]==[1,1,1,1]]
 extras={r['clip_id']:r for r in read(O/'word100_feature_expansion_20261113/INPUTS_NEW.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']}
 net=ViewConditionedShape();cache={};found=[];checked=[]
 for r in public:
  key=r['clip_id'],r['side']
  if key not in cache:
   record=O/'word100_rtmw_joint_learning_20261108/records'/f'{key[0]}.json'
   source=read(record)if record.exists()else extras.get(key[0])
   if source is None:continue
   path=source.get('raw_path',source.get('raw'));assert path and 'Temp'not in path
   raw=dict(np.load(path));ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));state=net.sequence(ch,r['side'])
   cache[key]=(raw,ch,state,source);checked.append(key)
  raw,ch,state,source=cache[key];a=max(0,r['frame']-12);b=min(len(state['states']),r['frame']+13)
  ids=np.flatnonzero(state['states'][a:b]==1)+a
  if len(ids):
   item=dict(anchor=r,candidate_frames=ids.tolist(),source=source,warning='Nearby open anchor is not evidence the candidate frame is open. RGB review required.')
   found.append(item);print(r['clip_id'],r['side'],r['frame'],ids.tolist(),flush=True)
 write(HERE/'SHAPE_REVIEW_CANDIDATES.json',dict(rows=found,checked_hand_sequences=len(checked),labels_assigned=False))
if __name__=='__main__':main()
