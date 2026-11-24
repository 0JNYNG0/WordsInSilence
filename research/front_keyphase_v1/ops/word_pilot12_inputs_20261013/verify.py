"""Independent arithmetic, metadata and original pixel checks; no detector rerun."""
import hashlib
import json
import math
from pathlib import Path
import sys
from collections import Counter
import av
import numpy as np
from PIL import Image

UNIT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def independent_channels(o,rows):
    ids=[0,11,12,13,14,15,16,23,24]+list(range(33,75))
    xy=o['xy_px'][rows][:,ids]; valid=o['usable'][rows][:,ids];ts=o['timestamp_us'][rows];n=len(ts)
    both=valid[:,1]&valid[:,2];scale=float(np.median(np.linalg.norm(xy[both,1]-xy[both,2],axis=1)))
    origin=(xy[:,1]+xy[:,2])/2
    body=np.zeros_like(xy);bv=valid&both[:,None]
    for j in range(51):body[bv[:,j],j]=(xy[bv[:,j],j]-origin[bv[:,j]])/scale
    hands=np.zeros((n,42,2),np.float32);hv=np.zeros((n,42),bool)
    for side,base in enumerate([9,30]):
        lengths=[]
        for t in range(n):
            for j in [5,9,13,17]:
                if valid[t,base] and valid[t,base+j]:lengths.append(float(np.linalg.norm(xy[t,base+j]-xy[t,base])))
        hs=float(np.median(lengths)) if lengths else float('nan')
        if not np.isfinite(hs) or hs<=1e-6:continue
        for j in range(21):
            ok=valid[:,base]&valid[:,base+j];hv[:,side*21+j]=ok
            hands[ok,side*21+j]=(xy[ok,base+j]-xy[ok,base])/hs
    dt=np.zeros((n,1),np.float32);dt[1:,0]=np.diff(ts).astype(np.float32)/1e6
    return np.concatenate([body.reshape(n,102),hands.reshape(n,84),bv.astype(np.float32),hv.astype(np.float32),dt],axis=1)

def main():
    scope=read(UNIT/'SCOPE.json');out=Path(scope['output']);r=read(out/'RESULT.json');cnt=Counter();worst=0.0
    for c in r['clips']:
        for path,hs in [('observation','observation_sha256'),('channels','channels_sha256'),('rgb_cache','rgb_sha256')]:assert sha(c[path])==c[hs]
        o=load(c['observation']);ch=load(c['channels']);a=load(c['rgb_cache']);n=c['frames'];raws=read(out/'rgb'/f"{c['clip_id']}.frames.json")
        assert o['xy_px'].shape==(n,75,2) and not np.isfinite(o['xy_px'][~o['observed']]).any()
        assert not (o['usable']&~o['observed']).any()
        assert np.array_equal(o['frame_id'],np.arange(n)) and (np.diff(o['timestamp_us'])>0).all()
        hand=o['usable'][:,33:].any(1);where=np.flatnonzero(hand);lo,hi=map(int,(where[0],where[-1]))
        assert c['web_span']==[lo,hi] and np.array_equal(ch['span'],[lo,hi])
        for key,rows in [('x_full',np.arange(n)),('x_web',np.arange(lo,hi+1))]:
            expected=independent_channels(o,rows);error=float(np.abs(expected-ch[key]).max());worst=max(worst,error)
            np.testing.assert_allclose(expected,ch[key],atol=1e-6,rtol=1e-6)
            np.testing.assert_array_equal(expected[:,186:279],ch[key][:,186:279])
        for k in ('timestamp_us','frame_id'):assert np.array_equal(a[k],o[k]) and np.array_equal(ch[k],o[k])
        assert np.array_equal(ch['web_frame_id'],o['frame_id'][lo:hi+1])
        assert np.array_equal(ch['web_timestamp_us'],o['timestamp_us'][lo:hi+1])
        assert np.array_equal(a['pose_x'],ch['x_full'])
        assert not a['region_estimated'].any() and np.array_equal(a['available'],a['observed_rgb'])
        assert not a['rgb'][~a['available']].any() and not a['pixel_valid'][~a['available']].any()
        assert (a['bbox_xyxy'][~a['available']]==-1).all()
        core=np.zeros(n,bool)
        for start,end in c['intervals']:core|=(o['timestamp_us']>=start*1e6)&(o['timestamp_us']<=end*1e6)
        assert int(core.sum())==c['annotation_core']['frames']
        reasons=Counter()
        with av.open(c['source'],mode='r') as con:
            st=con.streams.video[0];first=None;got=0
            for fid,f in enumerate(con.decode(st)):
                first=f.pts if first is None else first;ts=int(round(float((f.pts-first)*st.time_base)*1e6))
                assert ts==int(o['timestamp_us'][fid])==raws[fid]['timestamp_us']
                rgb=f.to_ndarray(format='rgb24');assert hashlib.sha256(rgb.tobytes()).hexdigest()==raws[fid]['rgb_sha256']
                assert raws[fid]['frame_id']==fid
                h,w=rgb.shape[:2]
                for side,sl in enumerate([slice(33,54),slice(54,75)]):
                    xy=o['xy_px'][fid,sl].astype(np.float64);valid=o['usable'][fid,sl]
                    assert a['landmark_observed_count'][fid,side]==valid.sum()
                    if not valid.any():reason='missing'
                    elif not valid.all():reason='partial'
                    elif not np.isfinite(xy).all():reason='nonfinite'
                    elif ((xy<0).any() or (xy[:,0]>=w).any() or (xy[:,1]>=h).any()):reason='outside'
                    elif np.ptp(xy,axis=0).max()<=0:reason='degenerate'
                    else:reason='valid'
                    reasons[reason]+=1;assert reason==str(a['reason'][fid,side])
                    assert bool(a['available'][fid,side])==(reason=='valid')
                    if reason!='valid':continue
                    size=max(2,math.ceil(1.5*float(np.ptp(xy,axis=0).max())))
                    corner=np.floor((xy.min(0)+xy.max(0))/2-size/2).astype(int);box=(*corner,*(corner+size))
                    assert np.array_equal(box,a['bbox_xyxy'][fid,side])
                    assert ((xy>=corner)&(xy<corner+size)).all()
                    im=np.array(Image.fromarray(rgb).crop(box).resize((224,224),Image.Resampling.BILINEAR))
                    pm=np.array(Image.new('L',(w,h),255).crop(box).resize((224,224),Image.Resampling.NEAREST))>0
                    np.testing.assert_array_equal(im,a['rgb'][fid,side]);np.testing.assert_array_equal(pm,a['pixel_valid'][fid,side])
                    assert abs(a['frame_relative_size'][fid,side]-size/w)<1e-12
                    if o['usable'][fid,[11,12]].all():
                        shoulders=o['xy_px'][fid,[11,12]].astype(np.float64);s=float(np.linalg.norm(shoulders[1]-shoulders[0]))
                        assert abs(a['shoulder_relative_size'][fid,side]-size/s)<1e-12
                    cnt['rgb_crops_pixel_exact']+=1;cnt['padded_crops']+=int(not pm.all())
                cnt['source_frames_time_pixel_exact']+=1;got+=1
            assert got==n
        assert dict(reasons)==c['crop_reasons'];cnt['clips']+=1
        print(json.dumps({'verified_clips':cnt['clips'],'frames':cnt['source_frames_time_pixel_exact']}),flush=True)
    assert cnt['clips']==12 and cnt['source_frames_time_pixel_exact']==1444
    assert all(sha(p)==h for p,h in scope['source_sha256'].items())
    result={'status':'PASS','counts':dict(cnt),'max_channel_abs_error':worst,
        'source_pins_unchanged':len(scope['source_sha256']),
        'method':'Codex independent source pixels/PTS/crop arithmetic/channel calculation; not an independent human or agent review.',
        'detector_reruns':0,'training':0,'recognition_accuracy_measured':False,
        'limitation':'Verifies linkage and arithmetic, not anatomical correctness of MediaPipe or true sign form.'}
    (UNIT/'INDEPENDENT_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
