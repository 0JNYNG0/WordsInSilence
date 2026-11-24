"""Public-only, sparse-label handshape transfer audit. No production imports/mutations."""
from pathlib import Path
import json, hashlib, sys
from collections import Counter
import numpy as np
sys.dont_write_bytecode = True
U=Path(__file__).resolve().parent; O=U.parent
B=O/'word100_shared_motion_learning_20261109'
J=O/'word100_rtmw_joint_learning_20261108'
A=O/'demo_handshape_learning_20261113'
F=O/'word100_broad_factorized_learning_20261110'
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')

def feature(xy,mask):
    xy=np.asarray(xy,np.float32); mask=np.asarray(mask,bool)
    # Same intrinsic shape convention for either hand; orientation is a separate task.
    axis=xy[9]-xy[0]; length=float(np.linalg.norm(axis))
    ok=bool(mask[np.r_[0,5:21]].all() and length>1 and np.isfinite(xy).all())
    if not ok: return np.zeros(63,np.float32),False
    ey=axis/length; ex=np.array([ey[1],-ey[0]],np.float32)
    if float((xy[5]-xy[17])@ex)<0: ex=-ex
    scale=max(float(np.median(np.linalg.norm(xy[[5,9,13,17]]-xy[0],axis=1))),1.)
    local=np.stack([(xy-xy[0])@ex,(xy-xy[0])@ey],1)/scale
    local[~mask]=0
    return np.r_[local.ravel(),mask.astype(np.float32)].astype(np.float32),True

def main():
    assert not (U/'FROZEN.json').exists(), 'Completed preparation must not be overwritten'
    ready=read(B/'READY.json'); source={r['clip_id']:r for r in ready['rows']}
    labels={}; pins={}; conflicts=[]
    def pin(p): pins[str(p)]=sha(p)
    def add(cid,frame,side,attr,val,origin):
        key=(cid,int(frame),side); q=labels.setdefault(key,dict(y=[-1]*3,origins=[]))
        if q['y'][attr] not in (-1,int(val)): conflicts.append([cid,frame,side,attr,q['y'][attr],val])
        else: q['y'][attr]=int(val)
        q['origins'].append(origin)
    # Only sparse RGB/visible-feature labels. DTW stage arrays are never targets here.
    mapping={0:('R',0),1:('R',1),2:('L',1),3:('R',2),4:('L',2)}
    for r in ready['rows']:
        if not r['has_labels']: continue
        assert sha(r['path'])==r['sha256']; pin(r['path'])
        z=np.load(r['path'],allow_pickle=False)
        for i,j in zip(*np.where(z['ym'][:,:5]>0)):
            side,a=mapping[int(j)]
            add(r['clip_id'],int(z['frame_id'][i]),side,a,z['y'][i,j],'shared_sparse_rgb')
    more=read(F/'ADDITIONAL_LABELS.json')
    for cid,rr in more['shape'].items():
        assert source[cid]['role']=='fit'
        for r in rr:
            for j,y in r['features'].items():
                side,a=mapping[int(j)]; add(cid,r['frame'],side,a,y,'additional_sparse_rgb')
    extra=read(B/'EXTRA_EVAL_LABELS.json')
    for r in extra['rows']:
        assert source[r['clip_id']]['role']=='development'
        for j,y in enumerate(r['targets']):
            if y is not None:
                side,a=mapping[j];add(r['clip_id'],r['frame'],side,a,y,'extra_dev_sparse_rgb')
    assert not conflicts, conflicts
    rows=[];xx=[];rawcache={}
    for (cid,frame,side),q in sorted(labels.items()):
        meta=source[cid]
        if cid not in rawcache:
            p=J/'records'/f'{cid}.json'; rec=read(p); pin(p)
            assert sha(rec['raw_path'])==rec['raw_sha256'] and sha(rec['channel_path'])==rec['channel_sha256']
            pin(rec['raw_path']);pin(rec['channel_path'])
            raw=np.load(rec['raw_path'],allow_pickle=False);chan=np.load(rec['channel_path'],allow_pickle=False)
            assert np.array_equal(raw['frame_id'],chan['frame_id'])
            rawcache[cid]=(raw,chan)
        raw,chan=rawcache[cid];ii=np.flatnonzero(raw['frame_id']==frame);assert len(ii)==1;i=int(ii[0])
        sl=slice(112,133) if side=='R' else slice(91,112)
        x,ok=feature(raw['xy_px'][i,sl],chan['point_available'][i,sl]);xx.append(x)
        rows.append(dict(clip_id=cid,word=meta['word_id'],person=meta['person'],role=meta['role'],frame=frame,pts_us=int(raw['timestamp_us'][i]),side=side,y=q['y'],input_available=ok,source_sha256=meta['source_sha256'],origins=sorted(set(q['origins']))))
    digitrows=[];dx=[];admit=read(A/'ADMISSION.json');byid={r['clip_id']:r for r in read(A/'INPUTS.json')['rows']}
    for r in admit['rows']:
        # Never load numeric DEV for this transfer training or test.
        if r['role']!='fit': continue
        rec=byid[r['clip_id']];p=Path(rec['channels']);assert sha(p)==rec['channels_sha256'];pin(p)
        z=np.load(p,allow_pickle=False);ii=np.flatnonzero(z['frame_id']==r['frame']);assert len(ii)==1;i=int(ii[0])
        x,ok=feature(z['raw_xy_px'][i,112:133],z['point_available'][i,112:133]);dx.append(x)
        digitrows.append(dict(r,input_available=ok))
    fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development']
    assert {r['person'] for r in fit+digitrows}.isdisjoint({r['person'] for r in dev})
    assert not ({r['source_sha256'] for r in fit+digitrows}&{r['source_sha256'] for r in dev})
    assert {r['word'] for r in digitrows}.isdisjoint({r['word'] for r in rows})
    def counts(rr):
        return dict(clips=len({r['clip_id'] for r in rr}),words=len({r['word'] for r in rr}),hand_anchors=len(rr),label_cells=sum(v>=0 for r in rr for v in r['y']),unavailable_anchors=sum(not r['input_available'] for r in rr),by_feature={name:{str(v):sum(r['y'][j]==v for r in rr) for v in (0,1)} for j,name in enumerate(['index_only','four_open','four_closed'])},by_hand={s:sum(v>=0 for r in rr if r['side']==s for v in r['y']) for s in ('R','L')})
    coverage=dict(common_word_training=899,common_word_scope=100,scope='Audit of shared100 sparse shape pack plus its verified additions; not an exhaustive count of every later motion/episode annotation.',ordinary_fit=counts(fit),ordinary_development=counts(dev),auxiliary_fit_clips=len({r['clip_id'] for r in digitrows}),auxiliary_fit_anchors=len(digitrows),auxiliary_finger_cells=sum(v>=0 for r in digitrows for v in r['finger_states']),auxiliary_index_state_counts=dict(Counter(str(r['finger_states'][1]) for r in digitrows)),missing=['No individual-finger ordinary-word evaluation truth in this pack','Digits 2/4/6 all extend index: no index-folded auxiliary contrast','No finger spread/contact continuous truth','No left-hand numeric source anchors','No verified educational error/acceptable-boundary labels'],unknown_is_not_negative=True,expert_gold=False)
    write(U/'COVERAGE.json',coverage);write(U/'INPUT_MANIFEST.json',dict(ordinary=rows,auxiliary=digitrows))
    np.savez_compressed(U/'INPUTS.npz',ordinary=np.array(xx),auxiliary=np.array(dx))
    for p in [B/'READY.json',B/'ANNOTATION_REVIEW.json',B/'EXTRA_EVAL_LABELS.json',F/'ADDITIONAL_LABELS.json',A/'ADMISSION.json',U/'INPUT_MANIFEST.json',U/'INPUTS.npz',U/'COVERAGE.json']:pin(p)
    prot=read(A/'FROZEN.json')['protection']
    assert all(sha(p)==h for p,h in prot.items())
    cfg=dict(status='FROZEN_BEFORE_NEW_SCORES',authorization='2026-11-13 user approved evidence review and execution if justified; individual-finger feedback wording added as desired output',purpose='Isolate transfer from existing public numeric anchors to ordinary KSL shape attributes; no number meaning assigned to ordinary signs.',conditions=['ordinary_only','plus_numeric_name','plus_finger_attributes'],seeds=[42,43,44],steps=600,lr=.001,shared_width=32,aux_weight=1.,normalization='Ordinary FIT only, same across conditions; intrinsic hand position/scale/in-plane orientation removed, orientation and movement are not evaluated or modified.',same_ordinary_updates=True,inputs='63 values: hand-local 21 xy + observation masks; no word, body location, person, frame/time',labels=['index-only (inherited definition)','four long fingers open','four long fingers closed'],evaluation='All frozen ordinary DEV known cells including unavailable; per-hand/per-word/feature metrics; historical development not new independent test; no retraining/threshold selection from results.',threshold=.5,confidence=.85,primary='paired results across all three seeds; no best seed',gate='No aggregate or per-hand/per-attribute regression in any seed; even PASS only licenses temporal integration study, never automatic web adoption.',scope_limits=['Frame component probe, not replacement of TCN+GRU','No temporal/transition/occlusion inference gain claim','No direct individual-finger correction claim on ordinary signs','No full feedback false-warning or learner-error rates','No comparison of 100 vs 3000 word training'],preserve_100=True,source_pins=pins,protected=prot)
    write(U/'FROZEN.json',cfg);write(U/'CURRENT_EXECUTION.json',dict(status='PREPARED',running=None))
    print(json.dumps(coverage,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
