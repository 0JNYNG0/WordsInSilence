from prepare_review import *
from collections import Counter
rows=rd(U/'EXTRA_LABELS.json')['rows'];meta={m['clip_id']:m for m in rd(O/'word100_feature_expansion_20261113/DATA_READY.json')['rows'] if m['role']=='fit'}
checked=set();seen=set();fail=[];observed=[]
for r in rows:
 key=(r['clip_id'],r['frame'],r['side']);assert key not in seen;seen.add(key)
 assert r['role']=='fit' and r['person'] in range(1,10)
 for path,expected in [(r['source_video'],r['source_sha256']),(r['image'],r['image_sha256']),(r['raw_path'],r['raw_sha256'])]:
  if path not in checked:assert sha(path)==expected,path;checked.add(path)
 assert len(r['y'])==5 and all(v in [-1,0,1,2] for v in r['y'])
 assert not(r['y'][0] in [1,2]),'unsupported thumbB/F'
 assert all({'E':0,'B':1,'F':2,'U':-1}[s]==y for s,y in zip(r['states'],r['y']))
 m=meta[r['clip_id']];assert m['source_sha256']==r['source_sha256']
 z=np.load(m['path']);ix=np.flatnonzero(z['frame_id']==r['frame']);assert len(ix)==1;ii=int(ix[0]);side=0 if r['side']=='L' else 1
 assert int(z['timestamp_us'][ii])==r['pts_us']
 h=z['hand'][ii,side];valid=bool(z['hand_valid'][ii,side]);point_mask=h[42:]>0.5
 raw=np.load(r['raw_path']);ri=int(np.flatnonzero(raw['frame_id']==r['frame'])[0]);sl=slice(91,112) if side==0 else slice(112,133)
 pts=raw['xy_px'][ri,sl];conf=raw['score'][ri,sl]
 obs=dict(clip_id=r['clip_id'],frame=r['frame'],side=r['side'],states=r['states'],cache=m['path'],cache_sha256=m['sha256'],training_hand_valid=valid,training_available_points=int(point_mask.sum()),finite_raw_points=int(np.isfinite(pts).all(1).sum()),raw_score_min=float(np.min(conf)),raw_score_median=float(np.median(conf)),model_prediction_available=False,model_failure_not_claimed=True,visual_geometry_note=r['geometry_review_note'])
 observed.append(obs)
 if not valid:fail.append(dict(key=key,reason='training hand featureinvalid; do not silently trainzero observation as visible state'))
 if m['path'] not in checked:assert sha(m['path'])==m['sha256'];checked.add(m['path'])
# No changes to previous labels; exact old-frame duplication is accounted separately.
counts=Counter(v for r in rows for v in r['states'])
wr(U/'INPUT_OBSERVATION_AUDIT.json',dict(rows=observed,rows_with_invalid_hand=len(fail),invalid_rows=fail,scope='Cached training-path hand normalization/mask and nativeframe/SHA check; not new inference',model_errors='Unassessed: completed experiment did not save FIT finger predictions; DEV predictions not opened'))
wr(U/'VERIFICATION.json',dict(status='PASS' if not fail else 'INPUT_REVIEW_NEEDED',native_hand_rows=len(rows),unique_clip_count=len({r['clip_id'] for r in rows}),known_cells=sum(v!='U' for r in rows for v in r['states']),counts=dict(counts),verified_paths=len(checked),source_and_raw_sha_verified=True,exact_native_frame_pts=True,all_fit_only=True,no_new_inference=True,no_new_training=True,no_prior_result_edits=True,reviewer='Same Codex direct RGB review plus arithmetic script; independent root RGB review pending',duplicate_rechecks=rd(U/'DUPLICATE_RECHECKS.json')['rows']))
rr,b=make('NIA_SL_WORD2523_REAL01_F',[53,55,57,59,61,63],'boundary_recheck')
wr(U/'EGYPT_BOUNDARY_RECHECK.json',dict(rows=rr,panel=b,saved_guide_last_end=55,native_59_old_label=['U','B','U','U','U'],observation='At53/55 final side-facing hand shape is held. Around57-61 hands lower away from final shoulder-level pose;59 is completion/lowering vicinity rather than evidence to extend mainstage. Exact end boundary is not relabelled by this audit.',resolution='Keep old59 physical label provenance; do not force it into saved mainstage. New main-shape labels49/55 are directly inside old final segment. Root handles stage linkage.',model_predictions_used=False))
print(json.dumps(dict(rows=len(rows),invalid=len(fail),known=sum(v!='U' for r in rows for v in r['states']),counts=dict(counts),boundary_panel=b),ensure_ascii=False))

