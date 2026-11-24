"""Freeze one midpoint per actual missing-hand episode, before new RGB annotations.
Reuse every saved baseline/HaMeR/temporal output. No model fitting or user files.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_mano_threeway_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    assert not (U/'FROZEN.json').exists()
    f=read(OLD/'FROZEN_PROTOCOL.json');rows=[];total=0
    (U/'source_views').mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    for k,m in enumerate(f['manifest']):
        z=np.load(m['prepared'])
        for side in [0,1]:
            ids=np.flatnonzero(z['trigger'][:,side]);total+=len(ids)
            for run in np.split(ids,np.where(np.diff(ids)!=1)[0]+1) if len(ids) else []:
                t=int(run[(len(run)-1)//2]);eid=f'E{len(rows)+1:02d}'
                cap=cv2.VideoCapture(m['rgb_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read();cap.release();assert ok
                # Display only: original geometry; no detector output appears on the annotation view.
                crop=[550,250,1400,1000];x0,y0,x1,y1=crop
                rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);image=Image.fromarray(rgb[y0:y1,x0:x1]);canvas=Image.new('RGB',(x1-x0,y1-y0+60),'#172333');canvas.paste(image,(0,60));d=ImageDraw.Draw(canvas)
                d.text((10,5),f'{eid} / clip{k} frame{t} / 대상: 촬영자 '+('왼손' if side==0 else '오른손'),font=font,fill='white')
                d.text((10,30),'원본 픽셀 좌표 격자 · 추정 관절 미표시 · 표시용 확대',font=font,fill='#bdcbdf')
                for x in range(x0,x1,50):
                    d.line([(x-x0,60),(x-x0,y1-y0+60)],fill=(65,90,110),width=1);d.text((x-x0+2,62),str(x),font=font,fill='#fff39a')
                for y in range(y0,y1,50):
                    d.line([(0,y-y0+60),(x1-x0,y-y0+60)],fill=(65,90,110),width=1);d.text((2,y-y0+62),str(y),font=font,fill='#fff39a')
                p=U/'source_views'/f'{eid}_source.jpg';canvas.save(p,quality=98)
                full=U/'source_views'/f'{eid}_full.jpg';cv2.imwrite(str(full),im,[cv2.IMWRITE_JPEG_QUALITY,98])
                rows.append({'id':eid,'clip_index':k,'clip_id':m['clip_id'],'label':m['word_label_for_display_only'],'frame':t,'side':side,'gap_start':int(run[0]),'gap_end':int(run[-1]),'rgb_path':m['rgb_path'],'source_frame':str(full),'source_frame_sha256':sha(full),'source_grid':str(p),'display_crop':crop,'prepared':m['prepared'],'comparison':str(OLD/'comparisons'/f"{m['clip_id']}.npz")})
    pre=read(U.parent/'word100_3d_hand_pilot_preflight_20261025/PREPARED.json')
    pins={str(p):sha(p) for p in OLD.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.py','.html','.txt'] and not set(p.parts)&{'private_assets','external','runtime_env','downloads','cache','__pycache__'}}
    frozen={'at':datetime.datetime.now().astimezone().isoformat(),'task':'Actual missing-hand public KSL accuracy audit, wrist and visible fingers separately','source':'Same prior 7 public videos, 812 frames, 57 missing handframes; previously viewed diagnostic data, not new independent test',
      'cohort':'All contiguous trigger episodes, one central frame (earlier central if even). No acceptance/output-based sample exclusion.',
      'missing_handframes':total,'episodes':len(rows),'rows':rows,'conditions':['B saved full-frame MP redetection then fixed ROI','raw HaMeR same original-frame candidate before gates','H saved baseline plus accepted candidates','HT saved H plus temporal check'],
      'annotation':'Current RGB-only grids viewed, with context frames if needed; wrist and clearly identifiable fingertips represented as small coordinate rectangles. Ambiguous/occluded anatomical points null, never counted correct. Annotator same Codex has seen older outputs: not blinded or independent expert truth.',
      'metrics':['available and inside visual reference rectangle per wrist / visible fingertip','distance to rectangle lower bound and center distance, in original1920x1080pixels','unobserved and ambiguous denominators shown separately; missing not zero error','all57 missing hands kept for availability summary, primary accuracy only16 episode centers'],
      'interpretation':'Reference regions are visual uncertainty bounds, not exact landmark GT or a learner tolerance. Fingertip support is not full finger-angle/hidden3D accuracy.',
      'normal_crossing_preservation':'All812frame arrays. Original baseline valid points must remain bitwise unchanged in H and HT. All20 designated crossing frames explicitly audited.',
      'no_tuning':True,'no_training':True,'no_user_data':True,'C_T3F_change':False,'deployment':False,
      'prior_result_pins':pins,'protected':pre['protected'],'manifest':f['manifest']}
    write(U/'FROZEN.json',frozen);print('PREPARED',total,'missing handframes,',len(rows),'episodes')
if __name__=='__main__':main()
