from verified_candidate import *
def main():
 torch.set_num_threads(2);net=VerifiedCandidate();out=[]
 for key in['full_frame_640','upper_body_640']:
  q=read(U/'web'/f'capture_{key}.json');xy=np.array(q['query_xy'],np.float32);xy[xy<0]=np.nan
  z=dict(xy_px=xy,score=np.ones(xy.shape[:2],np.float32),timestamp_us=np.array(q['timestamp_us']),frame_id=np.array(q['frame_id']),source_width=np.array(q['width']),source_height=np.array(q['height']))
  # Display coordinates are already quarantined; use only to identify region
  # input dependencies. Scores are dummy and this is not a new raw extraction.
  r=net.analyze(z);v=r['observations'];out.append(dict(kind=key,regions=v['regions'],samples=[dict(frame=i,rel=v['rel'][i],available=v['available'][i],state=v['region_states'][i])for i in range(36,56,3)]))
 write(U/'CROP_DIAGNOSIS.json',dict(rows=out,scope='Consumer coordinates for feature dependency diagnosis, no raw score inference or model fitting'));print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
