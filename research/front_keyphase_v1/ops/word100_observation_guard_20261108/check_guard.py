from prepare import *
from guard import quarantine

def main():
    rows=read(U/'INPUT_PLAN.json')['rows'];checks=[]
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
    idx={r['clip_id']:r for r in rows}
    review=read(O/'word100_rtmw_mediapipe_compare_20261108/VISUAL_REVIEW.json')
    cross=[]
    for r in review['continuous_crossing_windows']:
        z=np.load(idx[r['clip_id']]['raw']);g=quarantine(z['xy_px'],z['score'],z['timestamp_us'],int(z['source_width']),int(z['source_height']))
        ids=r['native_frames'];check('full crossing '+r['clip_id'],not g['hand_quarantined'][ids].any());cross.append(dict(clip_id=r['clip_id'],frames=ids,masked=int(g['hand_quarantined'][ids].sum())))
        z=np.load(U/'crops'/(r['clip_id']+'.npz'));g=quarantine(z['xy_px'],z['score'],z['timestamp_us'],640,480)
        check('crop crossing '+r['clip_id'],not g['hand_quarantined'][ids].any())
    # Examine all candidate masks, no learned model inference.
    for r in rows:
        p=Path(r['cache']) if r['cache'] else U/'crops'/(r['clip_id']+'.npz');z=np.load(p);q=z['xy_px'].copy();s=z['score'].copy();ts=z['timestamp_us'];g=quarantine(q,s,ts,640,480)
        check(r['clip_id']+' raw retained',np.array_equal(q,z['xy_px'],equal_nan=True) and np.array_equal(s,z['score'],equal_nan=True))
        keep=np.ones((len(q),133),bool)
        for side,start in enumerate([91,112]):keep[g['hand_quarantined'][:,side],start:start+21]=False;keep[g['wrist_quarantined'][:,side],9+side]=False
        check(r['clip_id']+' no smoothing or other body edits',np.array_equal(g['xy_px'][keep],q[keep],equal_nan=True))
        doubled=quarantine(q*2,s,ts,1280,960)
        check(r['clip_id']+' scale invariant',np.array_equal(g['hand_quarantined'],doubled['hand_quarantined']) and np.array_equal(g['wrist_quarantined'],doubled['wrist_quarantined']))
        mapping=np.arange(133)
        for a,b in [(5,6),(7,8),(9,10)]:mapping[a],mapping[b]=mapping[b],mapping[a]
        mapping[91:112]=np.arange(112,133);mapping[112:133]=np.arange(91,112)
        mirrored=q[:,mapping].copy();mirrored[:,:,0]=640-mirrored[:,:,0]
        mg=quarantine(mirrored,s[:,mapping],ts,640,480)
        check(r['clip_id']+' left right symmetric',np.array_equal(g['hand_quarantined'][:,::-1],mg['hand_quarantined']) and np.array_equal(g['wrist_quarantined'][:,::-1],mg['wrist_quarantined']))
    # Preserve quick legitimate transitions at existing manually reviewed centers.
    visuals=read(O/'word100_rtmw_mediapipe_compare_20261108/VISUAL_INDEX.json')
    known=[]
    for v in visuals:
        if v['clip_id'] not in idx:continue
        z=np.load(idx[v['clip_id']]['raw']);g=quarantine(z['xy_px'],z['score'],z['timestamp_us'],int(z['source_width']),int(z['source_height']))
        fi=v['frame'];mask=v.get('target_mask',[])
        # Positive handshape labels only, never a negative/hidden target as visible truth.
        for side,cols in [(0,[0,2]),(1,[1,3])]:
            if any(j<len(mask) and mask[j] and v['targets'][j]>0.5 for j in cols):
                ok=not g['hand_quarantined'][fi,side];check(v['clip_id']+f' visible center{fi} side{side}',ok);known.append(dict(clip_id=v['clip_id'],frame=fi,side=side,preserved=bool(ok)))
    summary=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',n=len(checks),checks=checks,known_visible_centers=known,crossing=cross,reviewer='same Codex code/saved data tests; inherited RGB annotations are not independent expert truth')
    write(U/'ENGINEERING_CHECKS.json',summary);print(json.dumps(dict(status=summary['status'],n=len(checks),failed=[x for x in checks if not x['passed']]),ensure_ascii=False));assert summary['status']=='PASS'
if __name__=='__main__':main()
