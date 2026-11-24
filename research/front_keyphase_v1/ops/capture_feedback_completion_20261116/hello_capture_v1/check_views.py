from feedback_lesson import *
sys.path.insert(0,str(O/'word100_observation_guard_20261108'))
from guard import quarantine
def main():
 out=U/'VIEW_CHECK.json';assert not out.exists();net=FeedbackLesson();rows=[]
 for name in ['public_video_upper_body_640','public_video_640_letterbox','heldout_crop']:
  raw=dict(np.load(LIVE/'seated_diagnosis_20261116'/f'{name}.npz'))
  for mode in ['raw','runtime_guard']:
   z=dict(raw)
   if mode=='runtime_guard':z['xy_px']=quarantine(raw['xy_px'],raw['score'],raw['timestamp_us'],int(raw['source_width']),int(raw['source_height']))['xy_px']
   q=net.analyze(z);row=dict(view=name,mode=mode,completed=q['completed'],criteria={c['id']:c['status']for c in q['criteria']},all_supported=all(c['status']=='supported'for c in q['criteria']),seconds=q['seconds'])
   rows.append(row);write(U/f'VIEW_{name}_{mode}.json',q)
 write(out,dict(rows=rows,source_people=1,new_heldout_people=0,training=False));print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':
 sys.path.insert(0,str(O/'word100_rtmw_live_guard_20261109'))
 main()
