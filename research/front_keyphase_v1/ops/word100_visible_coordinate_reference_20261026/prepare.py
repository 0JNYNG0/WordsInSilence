"""Fixed public RGB evaluation sample; candidate coordinates are not shown during marking."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone,timedelta
import cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
P=U.parent/'word100_aihub_coordinate_alignment_20261026'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(name,x):
    p=U/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def font(n):return ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
def run():
    f=read(P/'FROZEN.json');check(f['protected']);assert read(P/'CURRENT_EXECUTION.json')['active_sessions']==[]
    records={r['clip_id']:r for r in read(P/'DONE.json')['rows']}
    sources={r['clip_id']:r for r in f['rows']}
    rows=[];pins={str(P/n):sha(P/n) for n in ['FROZEN.json','DONE.json','VERIFIED.json','VISUAL_REVIEW.json','NEXT_PLAN.json']}
    for old in f['visuals']:
        cid=old['clip_id'];s=sources[cid];a,b=s['intervals'][0]
        frames=[round(30*(a+(b-a)*t)) for t in [.10,.40,.90]]
        assert len(set(frames))==3 and all(t>=15 and t not in old['frames'] for t in frames)
        rec=records[cid]
        for p,h in [(s['array'],s['array_sha']),(s['mp'],s['mp_sha']),(rec['array'],rec['array_sha']),(old['video'],f['pins'][old['video']])]:pins[p]=h
        rows.append(dict(clip_id=cid,label=old['label'],video=old['video'],frames=frames,array=rec['array'],source_array=s['array'],mp=s['mp'],global_translation=rec['global_translation']))
    check(pins)
    write('AUTHORIZATION.json',dict(at=now(),request='Proceed with next visible RGB reference check and selective position correction quality validation.',training=False,operating_change=False,personal_data_access=False,downloads=False))
    write('FROZEN.json',dict(at=now(),rows=rows,pins=pins,protected=f['protected'],sample='6 prior diagnostic public TRAIN REAL02 clips; three new action frames each at .10/.40/.90 of provided interval. All 18 differ from prior visual review. This is not an unseen-person or unbiased population sample.',keypoints=[0,4,8],keypoint_names={'0':'wrist connection center','4':'thumb tip','8':'index fingertip'},annotation='Same Codex marks plausible 2D rectangular regions on RGB-only views. No copied detector coordinates. Prior clip outcomes are known, so not blind independent ground truth. Unclear identity, occlusion, blur, missing landmark => no rectangle and a reason.',criteria=dict(distance='Report minimum and maximum Euclidean distance from each candidate point to marked rectangle. B robustly closer only if its maximum distance < A minimum distance; robustly worse if B minimum > A maximum; otherwise unresolved.',wrist_support='At least 2 marked frames for each hand, all marked wrists inside B rectangles, no robust wrist regression, and >half wrists robustly closer. Passing means sampled wrist support only, never all-21-point correctness.',tips='Report thumb/index separately. Marking one tip or wrist cannot certify hidden or unmarked fingers.',preservation='No coordinate changes or model inference in this unit; evaluate exact previous A/B/C arrays. Missing/unclear points remain excluded from correctness numerator with counts reported; never filled.',adoption='No automatic corrected training labels or deployment from these approximate same-Codex visual regions. Need source-coordinate correspondence and broader validation first.'),viewport_crop=[600,140,1400,1080]))
    (U/'images').mkdir(exist_ok=True);out=[]
    for r in rows:
        cap=cv2.VideoCapture(r['video']);n=0;got=[]
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if n in r['frames']:
                rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));assert rgb.size==(1920,1080)
                stem=r['clip_id']+f'_{n:04d}';raw='images/'+stem+'_raw.png';rgb.save(U/raw)
                x0,y0,x1,y1=[600,140,1400,1080];crop=rgb.crop((x0,y0,x1,y1))
                canvas=Image.new('RGB',(840,1000),'#f5f7fb');canvas.paste(crop,(40,60));d=ImageDraw.Draw(canvas)
                d.text((8,8),r['label']+f' | frame {n} | RGB ONLY (no predictions)',font=font(20),fill='#17253b')
                for x in range(600,1400,50):
                    d.line([(40+x-x0,60),(40+x-x0,1000)],fill=(100,130,150),width=1)
                    d.text((40+x-x0,38),str(x),font=font(13),fill='#17253b')
                for y in range(150,1080,50):
                    d.line([(40,60+y-y0),(840,60+y-y0)],fill=(100,130,150),width=1)
                    d.text((0,54+y-y0),str(y),font=font(13),fill='#17253b')
                view='images/'+stem+'_grid.png';canvas.save(U/view)
                got.append(dict(frame=n,raw=raw,grid=view,raw_sha=sha(U/raw),grid_sha=sha(U/view)))
            n+=1
        cap.release();assert len(got)==3
        out.append(dict(clip_id=r['clip_id'],label=r['label'],frames=got))
    write('PREPARED.json',dict(status='RGB_ONLY_SAMPLE_READY',clips=out,frames=18,annotation_points_requested=108))
    print(json.dumps({'status':'PREPARED','frames':18,'rows':[(r['label'],r['frames']) for r in rows]},ensure_ascii=False),flush=True)
if __name__=='__main__':run()
