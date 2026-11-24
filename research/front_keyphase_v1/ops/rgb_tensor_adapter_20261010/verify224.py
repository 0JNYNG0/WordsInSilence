"""Independent source-to-cache reconstruction. Never imports the producer."""
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw
import torch
import rgb_input as adapter

UNIT=Path(__file__).resolve().parent
BASE=UNIT.parents[1]
OUT=BASE/'runs/rgb_tensor_adapter_v1'
torch.set_num_threads(2)


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)


def main():
    manifest=read(OUT/'CACHE_MANIFEST.json')
    index=read(BASE/'ops/rgb_comparison_contract_20261010/INPUT_INDEX.json')
    by_id={s['sample_id']:s for s in index['samples']}
    groups=defaultdict(list)
    for record in manifest: groups[record['source_video']].append(record)
    counts=Counter(); norm_error=0.; border_cases=0; visual={}; raw_digests=0
    for group_no,(video,records) in enumerate(sorted(groups.items()),1):
        needed=defaultdict(list); data={}
        for record in records:
            sid=record['sample_id']; source=by_id[sid]
            assert digest(record['cache_path'])==record['cache_sha256']
            a=load(record['cache_path']); data[sid]=a
            old=load(source['observed_rgb_path']); pose=load(source['shared_pose_path'])
            for dest,src in [('pose_x','x'),('frame_id','frame_id'),('timestamp_us','timestamp_us')]:
                assert np.array_equal(a[dest],pose[src])
            expected_observed=old['crop_valid']; expected_estimated=np.zeros_like(expected_observed)
            expected_boxes=np.full(a['bbox_xyxy'].shape,-1,np.int32)
            for row in source['rows']:
                j=row['row']
                for side,hand in enumerate(row['hands']):
                    assert hand['side']==('left','right')[side]
                    if hand['observed_rgb'] is not None:
                        expected_boxes[j,side]=old['bbox_xyxy'][j,side]
                    elif hand['estimated_region_rgb'] is not None:
                        expected_estimated[j,side]=True
                        expected_boxes[j,side]=load(hand['estimated_region_rgb']['path'])['bbox_xyxy']
                needed[row['frame_id']].append((sid,j,record['frame_digests'][j]))
            assert np.array_equal(a['observed_rgb'],expected_observed)
            assert np.array_equal(a['region_estimated'],expected_estimated)
            assert np.array_equal(a['available'],expected_observed | expected_estimated)
            assert np.array_equal(a['bbox_xyxy'],expected_boxes)
            assert np.array_equal(a['landmark_observed_count'],old['hand_usable'].sum(2))
            observed_points=old['hand_xy_px'][expected_observed]
            observed_boxes=expected_boxes[expected_observed]
            assert ((observed_points >= observed_boxes[:,None,:2]) & (observed_points < observed_boxes[:,None,2:])).all()
            counts['observed21_inside_saved_square']+=int(expected_observed.sum())
            np.testing.assert_allclose(a['shoulder_relative_size'][expected_observed],old['shoulder_relative_size'][expected_observed],rtol=0,atol=0,equal_nan=True)
            np.testing.assert_allclose(a['frame_relative_size'][expected_observed],old['frame_relative_size'][expected_observed],rtol=0,atol=0,equal_nan=True)
            assert not a['rgb'][~a['available']].any()
            assert not a['pixel_valid'][~a['available']].any()
            for arm in 'ABC':
                p=adapter.pack(a,arm)
                mask=np.zeros_like(expected_observed) if arm=='A' else expected_observed if arm=='B' else expected_observed | expected_estimated
                np.testing.assert_array_equal(p['slots'].numpy(),np.argwhere(mask))
                np.testing.assert_array_equal(p['available'].numpy(),mask)
                if mask.any():
                    expected=(a['rgb'][mask].astype(np.float32)/np.float32(255)-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
                    error=float(np.abs(p['images'].numpy()-expected.transpose(0,3,1,2)).max())
                    assert error<=1e-6; norm_error=max(norm_error,error)
                features=torch.arange(int(mask.sum())*3,dtype=torch.float32).reshape(-1,3)
                scattered=adapter.scatter_features(features,p['slots'],len(mask)).numpy()
                assert np.array_equal(scattered[mask],features.numpy())
                assert not scattered[~mask].any()
                for k,oldkey in [('pose_x','x'),('frame_id','frame_id'),('timestamp_us','timestamp_us')]:
                    np.testing.assert_array_equal(p[k].numpy(),pose[oldkey])
            counts.update(rows=len(a['frame_id']),slots=int(a['available'].size),observed=int(expected_observed.sum()),estimated=int(expected_estimated.sum()),available=int(a['available'].sum()))
        seen=set()
        with av.open(video,mode='r') as con:
            stream=con.streams.video[0]
            for fid,frame in enumerate(con.decode(stream)):
                if fid in needed:
                    rgb=frame.to_ndarray(format='rgb24'); full=Image.fromarray(rgb)
                    pts=int(round(frame.pts*stream.time_base*1_000_000)); h,w=rgb.shape[:2]
                    for sid,row,frame_record in needed[fid]:
                        a=data[sid]
                        assert pts==int(a['timestamp_us'][row])==frame_record['timestamp_us']
                        assert frame_record['frame_id']==fid
                        assert hashlib.sha256(rgb.tobytes()).hexdigest()==frame_record['raw_rgb_sha256']
                        raw_digests+=1
                        for side in range(2):
                            if not a['available'][row,side]: continue
                            box=tuple(map(int,a['bbox_xyxy'][row,side])); x0,y0,x1,y1=box
                            expected=np.array(full.crop(box).resize((224,224),Image.Resampling.BILINEAR))
                            np.testing.assert_array_equal(a['rgb'][row,side],expected)
                            original_mask=Image.new('L',(w,h),255).crop(box)
                            expected_mask=np.array(original_mask.resize((224,224),Image.Resampling.NEAREST))>0
                            np.testing.assert_array_equal(a['pixel_valid'][row,side],expected_mask)
                            area=max(0,min(w,x1)-max(0,x0))*max(0,min(h,y1)-max(0,y0))
                            assert abs(a['padding_fraction'][row,side]-(1-area/(x1-x0)**2))<1e-12
                            border_cases+=int(not expected_mask.all())
                            counts['pixels_checked']+=1
                            categories=['observed' if a['observed_rgb'][row,side] else 'estimated']
                            if not expected_mask.all(): categories.append('padding')
                            for kind in categories:
                                if kind not in visual:
                                    thumb=full.copy(); thumb.thumbnail((600,338)); d=ImageDraw.Draw(thumb)
                                    sx,sy=thumb.width/w,thumb.height/h
                                    d.rectangle((x0*sx,y0*sy,x1*sx,y1*sy),outline='red',width=3)
                                    visual[kind]=(thumb,Image.fromarray(expected),sid,row,side)
                    seen.add(fid)
                if fid>=max(needed): break
        assert seen==set(needed)
        if group_no%30==0: print(json.dumps({'independent_videos_checked':group_no,'total':len(groups)}),flush=True)
    expected={'observed21_inside_saved_square':5315,'rows':2789,'slots':5578,'observed':5315,'estimated':78,'available':5393,'pixels_checked':5393}
    assert dict(counts)==expected and len(manifest)==113
    pins=read(OUT/'SOURCE_PINS.json')
    assert all(digest(p)==h for p,h in pins.items())
    canvas=Image.new('RGB',(870,400*len(visual)),'white'); draw=ImageDraw.Draw(canvas)
    for j,(kind,(thumb,crop,sid,row,side)) in enumerate(visual.items()):
        y=j*400; draw.text((10,y+5),f'{kind} | {sid} | row{row} | '+('left','right')[side],fill='black')
        canvas.paste(thumb,(10,y+40)); canvas.paste(crop,(630,y+70))
        draw.text((630,y+310),'Original-frame224',fill='black')
    canvas.save(OUT/'VISUAL_CHECK.png')
    write(UNIT/'INDEPENDENT_CHECK.json',{'status':'PASS','segments':113,'counts':dict(counts),
        'normalization_max_abs_error':norm_error,'raw_frame_checks':raw_digests,'padding_crops':border_cases,
        'sources_unchanged':len(pins),'producer_imported':False,
        'method':'Independent PIL crop/resize from original frames and independently reconstructed masks/timeline; adapter packing is the code under test. Same Codex author, not external human review.',
        'accuracy_evaluation':False})
    print(json.dumps({'status':'PASS','counts':dict(counts),'norm_error':norm_error}),flush=True)


if __name__=='__main__': main()
