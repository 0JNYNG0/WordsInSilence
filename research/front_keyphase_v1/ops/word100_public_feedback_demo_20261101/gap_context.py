from common import *
def main():
 audit=read(U/'AUDIT.json');data=read(S/'DATASET.json')
 selected=sorted(set((r['clip_id'],r['frame_id']) for r in audit['records'] if r['seed']==42 and r['role']=='development' and r['stratum']=='geometry_gate_failed'))
 sources=[]
 for cid,f in selected:
  record=O/f'word100_body_boundary_20261017/data/records/{cid}.json'
  d=read(record);obs=Path(d['views']['native']['observation'])
  sources.append(dict(clip_id=cid,frame_id=f,record=str(record),record_sha256=sha(record),observation=str(obs),observation_sha256=sha(obs)))
 save('GAP_CONTEXT_SCOPE.json',dict(source='AUDIT predetermined failed-gate development centers only',rows=sources,
  note='Supplementary original-point count and contiguous absence length; no new predictions/scores/labels'))
 result=[]
 for s in sources:
  with np.load(s['observation'],allow_pickle=False) as z:obs=z['observed'].copy();xy=z['xy_px'].copy();ids=z['frame_id'].copy();ts=z['timestamp_us'].copy()
  ix=np.flatnonzero(ids==s['frame_id']);assert len(ix)==1;i=int(ix[0]);valid=obs&np.isfinite(xy).all(-1);sides={}
  for side,off in [('L',33),('R',54)]:
   count=valid[:,off:off+21].sum(1);missing=count==0;lo=hi=i
   if missing[i]:
    while lo>0 and missing[lo-1]:lo-=1
    while hi+1<len(ids) and missing[hi+1]:hi+=1
   sides[side]=dict(observed_points=int(count[i]),zero_point_run_frames=hi-lo+1 if missing[i] else 0,
    zero_point_run_ids=[int(ids[lo]),int(ids[hi])] if missing[i] else None)
  result.append(dict(**s,seconds=float(ts[i]/1e6),sides=sides))
 save('DEVELOPMENT_GAP_CONTEXT.json',dict(rows=result,note='Original full-frame observed points; gate may fail for ROI/support reasons without zero-point absence. Not image visibility truth.'))
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
