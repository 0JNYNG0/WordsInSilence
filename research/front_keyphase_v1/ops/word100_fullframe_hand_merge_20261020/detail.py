"""Large evidence cards from saved coordinates, no detection."""
import json,sys
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from render import skeleton,fit
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf8'))
rows={r['clip_id']:r for r in read(U/'FROZEN.json')['rows']}
targets=read(U/'REVIEW_TARGETS.json')
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',23)
for target in map(int,sys.argv[1:]):
    t=targets[target];r=rows[t['clip_id']]
    e=read(U/'records'/f"{t['clip_id']}.json")['frames'][t['index']]
    q=e['decisions'][t['decision']];p=np.array(e['full'][t['decision']]['xy'])
    cap=cv2.VideoCapture(r['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,e['frame_id'])
    ok,bgr=cap.read();cap.release();assert ok
    with np.load(r['raw']) as z:old=z['hand_xy_px'][t['index'],:int(z['raw_count'][t['index']])]
    before=bgr.copy()
    for hand in old:skeleton(before,hand,(255,160,0))
    after=before.copy()
    skeleton(after,p,(0,240,40),q['joint_supported'] if t['kind']=='added' else None)
    for im in (before,after):cv2.line(im,(0,r['box'][3]),(im.shape[1]-1,r['box'][3]),(0,240,255),4)
    lo=np.maximum(p.min(0)-80,0).astype(int);hi=np.minimum(p.max(0)+80,[bgr.shape[1],bgr.shape[0]]).astype(int)
    panel=Image.new('RGB',(1600,1000),'white');d=ImageDraw.Draw(panel)
    d.text((20,10),f"{target:03d} | {r['label']} | 촬영자 {r['signer_id']:02d} | 저장 프레임 {t['index']} | 원본 프레임 {e['frame_id']}",font=font,fill='black')
    d.text((20,50),"기존 전체 검출",font=font,fill='#18558a')
    d.text((820,50),"재검출 확인 후 (초록) · 노란선 아래는 최종 입력에서 제외",font=font,fill='#11662a')
    panel.paste(fit(before,(780,440)),(10,85));panel.paste(fit(after,(780,440)),(810,85))
    d.text((20,540),"원본 손 확대",font=font,fill='black')
    d.text((820,540),"같은 장면의 관절 표시",font=font,fill='black')
    panel.paste(fit(bgr[lo[1]:hi[1],lo[0]:hi[0]],(780,390)),(10,585))
    panel.paste(fit(after[lo[1]:hi[1],lo[0]:hi[0]],(780,390)),(810,585))
    panel.save(U/'evidence'/f'detail_{target:03d}.jpg',quality=97)
print('DETAILS',sys.argv[1:])
