from pathlib import Path
import av, json, hashlib
import numpy as np
from PIL import Image, ImageDraw
U=Path(__file__).resolve().parent
P=U.parent/'avatar_motion_pilot_20261113'
checks=[]
def check(name,ok,value=None): checks.append(dict(name=name,passed=bool(ok),measured=value))
def decode(path):
    with av.open(str(path)) as c:
        return [(float(f.pts*f.time_base),f.to_ndarray(format='rgb24')) for f in c.decode(video=0)]
source=decode(P/'video_only/NIA_SL_WORD1567_REAL02_F.mp4')
for filename in ['SOURCE_BASELINE_CANDIDATE_60FPS.mp4','CANDIDATE_60FPS.mp4']:
    frames=decode(U/filename)
    check(filename+'_frame_count',len(frames)==254)
    check(filename+'_PTS',max(abs(t-i/60) for i,(t,_) in enumerate(frames))<1e-6)
    for i in [0,65,128,176,192,220,253]:
        actual=frames[i][1]
        if filename.startswith('SOURCE'):
            src=source[i//2][1];h,w=src.shape[:2]
            expected=[np.asarray(Image.fromarray(src[:1000,w//2-500:w//2+500]).resize((512,512)))]
            for label in ['baseline','candidate']:
                image=Image.open(U/(label+'_frames60')/f'frame{i:03d}.png').convert('RGB')
                iw,ih=image.size
                crop=(round(iw*70/720),round(ih*110/720),round(iw*650/720),round(ih*690/720))
                expected.append(np.asarray(image.crop(crop).resize((512,512))))
            for j,e in enumerate(expected):
                mae=float(np.mean(np.abs(actual[56:568,j*512:(j+1)*512].astype(float)-e)))
                check(f'{filename}_frame{i}_panel{j}_matches',mae<6,mae)
        else:
            expected=np.asarray(Image.open(U/'candidate_frames60'/f'frame{i:03d}.png').convert('RGB'))
            mae=float(np.mean(np.abs(actual.astype(float)-expected)))
            check(f'{filename}_frame{i}_matches',mae<6,mae)
# Contact sheet for review: no aspect-ratio distortion.
canvas=Image.new('RGB',(1440,544),'white');d=ImageDraw.Draw(canvas)
for j,sf in enumerate([0,16,32,48,64,80,88,96,104,110,118,126]):
    im=Image.open(U/'candidate_frames60'/f'frame{sf*2:03d}.png').convert('RGB').resize((240,240))
    x=j%6*240;y=j//6*272;canvas.paste(im,(x,y+26));d.text((x+8,y+5),f'source frame {sf}',fill='black')
canvas.save(U/'ROOT_REVIEW_MOTION_SHEET.png')
report=dict(status='PASS_PACKAGE_TIMING_AND_PANEL_CONTENT' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,
    source_native_frames=len(source),output_frames=254,output_fps=60,source_frame_interpolation=False,
    new_model_inference=False,unit_note='PNG to H264 mean RGB absolute error tolerance is only an encoding check',
    first_audit_correction='First comparison expected earlier uncropped layout while packaging agent updated the same fixed crop on both avatar panels. Preserved FIRST audit; latest explicit crop checked without changing video or tolerance.',
    pins={n:hashlib.sha256((U/n).read_bytes()).hexdigest() for n in ['SOURCE_BASELINE_CANDIDATE_60FPS.mp4','CANDIDATE_60FPS.mp4','SOURCE_BASELINE_CANDIDATE_KO.png','package_comparison.py']})
(U/'ROOT_PACKAGE_VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n','utf-8')
print(report['status'],len(checks),[c for c in checks if not c['passed']])
assert all(c['passed'] for c in checks)
