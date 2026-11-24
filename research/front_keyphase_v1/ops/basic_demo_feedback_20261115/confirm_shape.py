from shape_consensus import *
import time
GOLD={16:{'R':[46,52,58,63,69,75],'L':[46,58,69]},17:{'R':[95,102,109,116],'L':[81,95,109,116]}}

def main():
 torch.set_num_threads(2);FRESH=V/'fresh';assert not(FRESH/'RESULTS.json').exists();records=read(FRESH/'EXTRACTED.json')['rows'];truth=[]
 for r in records:
  for h,frames in GOLD[r['person']].items():
   for f in frames:truth.append(dict(clip_id=r['clip_id'],frame=f,hand=h,state=0,source_sha=r['sha256'],review=str(FRESH/f'{r["person"]}_shape_review.jpg')))
 write(FRESH/'SOURCE_TRUTH.json',dict(rows=truth,reviewer='Codex RGB before predictions; source-visible four long fingers extended; thumb not labelled. Occluded/contact/preparation ambiguity excluded, not inferred from dictionary.'))
 net=ShapeConsensus();sys.path.insert(0,str(O/'word100_rtmw_shared_input_20261108'));import adapter
 source=read(O/'word100_rtmw_joint_learning_20261108/records/NIA_SL_WORD0739_REAL01_F.json');donor=np.load(source['raw_path'])['xy_px'][79,112:133].copy();donor-=donor[0];out=[];anchors=[]
 for r in records:
  raw=dict(np.load(r['raw_path']));lo,hi=r['source_interval'];ids=np.flatnonzero((raw['timestamp_us']/1e6>=lo)&(raw['timestamp_us']/1e6<=hi));original=None
  for kind in ['original','right_fist','left_fist','missing']:
   z={k:v.copy()for k,v in raw.items()}
   if kind=='missing':z['xy_px'][:,91:]=np.nan
   elif kind!='original':
    start=112 if kind=='right_fist'else 91
    for i in ids:
     p=z['xy_px'][i,start:start+21];u=p[9]-p[0];v=donor[9];theta=np.arctan2(u[1],u[0])-np.arctan2(v[1],v[0]);c,sn=np.cos(theta),np.sin(theta);rot=np.array([[c,-sn],[sn,c]],np.float32);z['xy_px'][i,start:start+21]=donor@rot.T*np.linalg.norm(u)/max(np.linalg.norm(v),1e-6)+p[0]
   t=time.perf_counter();ch=adapter.convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']));ss={h:net.sequence(ch['tcn280'],h)for h in ['L','R']};dt=time.perf_counter()-t
   if original is None:original=ss
   opposite=None
   if kind in ['right_fist','left_fist']:h='L'if kind=='right_fist'else'R';opposite=bool(np.array_equal(ss[h]['states'],original[h]['states']))
   out.append(dict(clip_id=r['clip_id'],kind=kind,seconds=dt,opposite_unchanged=opposite,hands={h:dict(extended=int((q['states'][ids]==0).sum()),flexed=int((q['states'][ids]==1).sum()),mixed=int((q['states'][ids]==2).sum()),unknown=int((q['states'][ids]<0).sum()),frames=len(ids))for h,q in ss.items()}))
   np.savez_compressed(FRESH/f'{r["clip_id"]}_{kind}.npz',**{f'{h}_{k}':v for h,q in ss.items()for k,v in q.items()})
   if kind=='original':
    for a in truth:
     if a['clip_id']==r['clip_id']:
      ix=int(np.flatnonzero(z['frame_id']==a['frame'])[0]);anchors.append(dict(**a,predicted=int(ss[a['hand']]['states'][ix]),baseline=int(ss[a['hand']]['baseline'][ix])))
 summary={}
 for key in ['baseline','predicted']:summary[key]=dict(total=len(anchors),correct=sum(a[key]==a['state']for a in anchors),wrong=sum(a[key]>=0 and a[key]!=a['state']for a in anchors),unknown=sum(a[key]<0 for a in anchors))
 write(FRESH/'RESULTS.json',dict(summary=summary,rows=out,anchors=anchors,scope='People16/17 withheld from this candidate fit/selection; two source clips, not broad user accuracy. Fist/missing are engineered controls. Runtime excludes RTMW/capture/DTW/render.',whole_word_pass_qualified=False));print(summary);print(out)
if __name__=='__main__':main()
