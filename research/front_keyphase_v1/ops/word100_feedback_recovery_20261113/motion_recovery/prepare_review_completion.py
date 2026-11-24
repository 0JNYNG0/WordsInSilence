"""Preserve direct panel judgments and expose unsampled native times for H/U review."""
from audit_saved import *
import cv2
from PIL import Image,ImageDraw,ImageFont

DIRECT={
 1:['MH','MH','HH','MH'],2:['HH','MH','MH','MH'],
 3:['MH','MH','MH','HH'],4:['MH','MH','MH','MH'],
 5:['HH','MH','HH','MH'],6:['MH','MH','HH','MM'],
 7:['HH','MH','MH','MM'],8:['MM','HH','MH','MH'],
 9:['HH','MM','HH','MH'],10:['MM','MM','MM','HH'],
 11:['MM','HH','MM','MM'],12:['MM','MM','MM','MM'],
 13:['UM','MM','UM','MM'],14:['MM','MM','MM','MM'],
 15:['MM','MM','HH','MM'],16:['HH','MM','MM','MM'],
 17:['MM','HH','MM','UU'],18:['MM','MM','MM','MM'],
 19:['HH','MM','MM','HH'],20:['MM','HH','MM']}

def main():
    manifest=read(U/'PHYSICAL_REVIEW_MANIFEST.json');rows=[];extra=[]
    for r0 in manifest['records']:
        r=dict(r0);name=Path(r['panel']).name
        if name.startswith('physical_review'):
            labels=DIRECT[int(name.split('_')[-1].split('.')[0])][r['panel_row']]
        else:
            labels=(['MH','MH','HH']if r['word_id']=='WORD0739'else['MM','MM','HH'])[r['panel_row']]
            if r['word_id']=='WORD2523'and r['person']==6 and r['kind']=='held_posture_review':labels='UU'
        r['provisional_right_left']=labels
        r['observed_unique_source_frames']=sorted(set(r['review_frames']))
        r['repeated_boundary_images_not_counted']=len(r['review_frames'])-len(set(r['review_frames']))
        rows.append(r)
        # Complete all original native times for H/U, including the stationary opposite
        # wrist in MH windows. No numeric pose or model outputs determine the labels.
        if labels!='MM':extra.append(r)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    for offset in range(0,len(extra),4):
        pack=extra[offset:offset+4];im=Image.new('RGB',(1600,330*len(pack)),'white');draw=ImageDraw.Draw(im);out=U/f'physical_native_completion_{offset//4+1:02}.jpg'
        for row,r in enumerate(pack):
            cap=cv2.VideoCapture(r['source']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=[fr for fr in range(max(0,r['frame']-4),min(n,r['frame']+5))if fr not in r['observed_unique_source_frames']]
            # Full frame, so rest wrists hidden by the old display crop are visible.
            draw.text((5,row*330+3),f"{r['clip_id']} {r['kind']} centre {r['frame']} R/L {r['provisional_right_left']}",font=font,fill='#16304a')
            for col,fr in enumerate(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES,fr);ok,bgr=cap.read();assert ok
                pic=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));pic.thumbnail((390,270));im.paste(pic,(400*col+(400-pic.width)//2,row*330+32));draw.text((400*col+5,row*330+303),str(fr),font=font,fill='#16304a')
            cap.release();r['completion_panel']=str(out);r['completion_row']=row;r['completion_native_frames']=frames
        im.save(out,quality=93)
    write('PHYSICAL_LABEL_DRAFT.json',dict(status='AWAIT_NATIVE_COMPLETION_REVIEW',rows=rows,counts=dict(Counter(r['provisional_right_left']for r in rows)),new_panels=(len(extra)+3)//4))
    print(json.dumps(dict(windows=len(rows),native_completion_windows=len(extra),panels=(len(extra)+3)//4)))

if __name__=='__main__':
    from collections import Counter
    main()
