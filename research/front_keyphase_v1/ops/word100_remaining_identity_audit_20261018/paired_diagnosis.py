"""Descriptive native-vs-cropped sensor evidence only. No cross-view transfer."""
from audit import *
def main():
    rows={r['clip_id']:r for r in read(U/'CONTEXT.json')['rows']};out=[]
    for x in read(U/'REMAINING9.json'):
        r=rows[x['clip_id']];z=load(r['views']['sensor']['raw']);n=load(r['views']['masked']['raw']);a=load(P/'results'/f"{x['clip_id']}_sensor.npz");b=load(P/'results'/f"{x['clip_id']}_masked.npz")
        f=x['index'];i=x['raw_index'];matches=np.flatnonzero(n['frame_id']==z['frame_id'][f]);assert len(matches)==1;t=int(matches[0]);candidates=[]
        for j in range(int(n['raw_count'][t])):
            m=a['point_mask'][f,i]&b['point_mask'][t,j];idx=np.flatnonzero(m)
            d=float(np.linalg.norm(z['hand_xy_px'][f,i,idx]-n['hand_xy_px'][t,j,idx],axis=1).mean()) if len(idx) else None
            candidates.append({'raw_index':j,'side_hint':int(b['sides'][t,j]),'identity_side':int(b['identity_side'][t,j]),'observed_points':int(b['point_mask'][t,j].sum()),'common_sensor_visible_points':len(idx),'mean_distance_px':d})
        d={'id':x['id'],'word':x['word'],'clip_id':x['clip_id'],'frame':f,'sensor_points':x['all_observed_points'],'sensor_identity':x['identity_side'],'native_candidates':candidates,'native_pose_wrist_confidence':n['pose_conf'][t,[15,16]].astype(float).tolist(),'lead_before_annotated_core_ms':(min(q[0] for q in x['core_intervals'])-x['timestamp_us']/1e6)*1000,'restriction':'Native/full-image pixels outside sensor field are diagnostics only, not usable evidence for a real cropped webcam; no identity/coordinate transfer performed.'};out.append(d)
    write(U/'PAIRED_VIEW_DIAGNOSIS.json',out);print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
