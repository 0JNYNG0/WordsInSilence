"""Preview a fixed resting-wrist boundary on two public training source clips.

No detector, classifier, training, or production change. This is a geometry
proposal, not a selected training parameter or measured recognition gain.
"""
from pathlib import Path
import hashlib
import json
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

U=Path(__file__).resolve().parent
P=U.parent/'word100_fullframe_roi_pair_20261016'/'PLAN.json'
F='C:/Windows/Fonts/malgun.ttf'
B='C:/Windows/Fonts/malgunbd.ttf'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def draw(r, active, name):
    path=r['cached']['native']['observation']
    with np.load(path,allow_pickle=False) as z:
        xy=z['xy_px'].copy(); valid=z['usable'].copy()
    first=np.arange(5)
    assert valid[first[:,None],np.array([11,12,33,54])[None,:]].all()
    wrist_y=np.median(xy[first][:,[33,54],1],axis=0)
    sw=float(np.median(np.linalg.norm(xy[first,11]-xy[first,12],axis=1)))
    margin=0.02*sw
    proposed=int(np.floor(min(wrist_y)-margin))
    old=r['bottom_box'][3]
    w,h=r['original_size']
    assert old<proposed<h
    assert sha(r['video_path'])==r['source_sha256']
    frames={}; ts={}
    with av.open(r['video_path']) as c:
        for i,f in enumerate(c.decode(video=0)):
            if i in [0,active]: frames[i]=f.to_image().convert('RGBA');ts[i]=float(f.pts*f.time_base)
            if i>=active: break
    imout=Image.new('RGB',(2544,984),'#f3f6fa');d=ImageDraw.Draw(imout)
    d.text((48,20),r['label']+' | 손을 내린 자세의 손목 가까이로 하단 확장',font=ImageFont.truetype(B,37),fill='#152a43')
    d.text((48,76),f'기존 경계 {old}px → 검토안 {proposed}px  ·  아래로 {proposed-old}px 확장  ·  영상 안에서는 고정',font=ImageFont.truetype(F,28),fill='#304c69')
    stats={}
    for col,idx in enumerate([0,active]):
        im=frames[idx];assert im.size==(w,h)
        overlay=Image.new('RGBA',im.size)
        od=ImageDraw.Draw(overlay);od.rectangle((0,proposed,w-1,h-1),fill=(25,35,48,180))
        im=Image.alpha_composite(im,overlay);di=ImageDraw.Draw(im)
        di.rectangle((4,4,w-5,proposed-1),outline='#ffe14c',width=8)
        for x in range(0,w,65): di.line((x,old,min(w-1,x+38),old),fill='#ff806f',width=6)
        x=48+col*1248;y=165
        d.text((x,125),f'{"손을 내린 시작 자세" if idx==0 else "동작 중"} · {ts[idx]:.2f}초',font=ImageFont.truetype(B,25),fill='#152a43')
        imout.paste(im.resize((1200,675),Image.Resampling.LANCZOS).convert('RGB'),(x,y))
        d.text((x+20,y+int(old*0.625)-37),f'기존 {old}px',font=ImageFont.truetype(B,24),fill='#ffb0a4',stroke_width=2,stroke_fill='#152a43')
        d.text((x+20,y+int(proposed*0.625)-39),f'손목 기준 검토안 {proposed}px',font=ImageFont.truetype(B,24),fill='#ffe14c',stroke_width=2,stroke_fill='#152a43')
    oldmask=valid & (xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<old)
    newmask=valid & (xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<proposed)
    assert not np.any(oldmask & ~newmask)
    stats['initial_5_old_hand_points']=int(oldmask[:5,33:75].sum())
    stats['initial_5_proposed_hand_points']=int(newmask[:5,33:75].sum())
    stats['last_5_old_hand_points']=int(oldmask[-5:,33:75].sum())
    stats['last_5_proposed_hand_points']=int(newmask[-5:,33:75].sum())
    stats['additional_retained_hand_point_frames']=int(newmask[:,33:75].sum()-oldmask[:,33:75].sum())
    d.text((48,863),'노란 실선: 확장한 입력 범위   /   붉은 점선: 이전 경계   /   어두운 하단: 입력 제외',font=ImageFont.truetype(B,27),fill='#152a43')
    d.text((48,909),'상단·좌우 원본 유지. 검토용 사진이며, 모델 재학습이나 운영 설정 변경은 하지 않았습니다.',font=ImageFont.truetype(F,25),fill='#304c69')
    out=U/name;assert not out.exists();imout.save(out)
    return dict(clip_id=r['clip_id'],label=r['label'],source=r['video_path'],source_sha256=r['source_sha256'],observation=path,observation_sha256=sha(path),
                original_size=[w,h],old_box=r['bottom_box'],proposed_box=[0,0,w,proposed],
                resting_frames_zero_based=first.tolist(),wrist_median_y=wrist_y.tolist(),shoulder_width_px=sw,margin_px=margin,
                selected_frames=[0,active],timestamp_seconds=ts,geometry_only_stats=stats,image=name,image_sha256=sha(out))

if __name__=='__main__':
    plan=json.loads(P.read_text('utf-8-sig'));items={r['clip_id']:r for r in plan['items']}
    rule='For preview only: take first 5 verified arms-down frames. Median each anatomical hand wrist y. Boundary=floor(min(two wrist medians)-0.02*median shoulder width). Fixed per clip, same rule both examples. Review before expanding to more clips.'
    # Freeze the proposal before generating comparative point counts.
    (U/'SCOPE.json').write_text(json.dumps(dict(status='PREVIEW_ONLY_NO_TRAINING',rule=rule,plan_sha256=sha(P),source_role='public TRAIN examples only'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[draw(items['NIA_SL_WORD0106_REAL01_F'],52,'coldcoffee_wrist_boundary.png'),draw(items['NIA_SL_WORD1545_REAL01_F'],36,'swimming_wrist_boundary.png')]
    data=dict(status='PREVIEW_READY_NOT_SELECTED_FOR_TRAINING',rule=rule,rows=rows,
              limitations=['Two public training examples only; not all-clip validation.','Cached full-frame detections only; this does not measure detection on an actually cropped camera image.','More included points are not recognition accuracy or correctness.','No single boundary guarantees all active motion visible or all idle hands absent.'],new_detection=False,new_training=False,production_changes=False)
    (U/'PREVIEW.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([dict(clip=r['clip_id'],proposed=r['proposed_box'],stats=r['geometry_only_stats']) for r in rows]))
