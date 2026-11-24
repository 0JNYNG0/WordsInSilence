from holdout_common import *
import cv2,re
from PIL import Image,ImageDraw,ImageFont
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18);out=U/'left_search';out.mkdir(exist_ok=True);rows=[]
used=set()
for p in read(U/'EXPOSURE_AUDIT.json')['manifests']:used |= set(re.findall(r'NIA_SL_WORD\d+_REAL\d+_F',Path(p).read_text('utf-8-sig')))
for word in ['WORD1486','WORD2309','WORD0678','WORD2572']:
 cid=f'NIA_SL_{word}_REAL17_F';assert cid not in used;path=O/'word100_t3f_heldout_20261019/data/clips'/cid/'source.mp4';meta=read(O/'word100_t3f_heldout_20261019/data/records'/f'{cid}.json');assert sha(path)==meta['source_sha256'];cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ids=np.linspace(round(n*.25),round(n*.8),12).round().astype(int);sheet=Image.new('RGB',(1280,870),'white');dr=ImageDraw.Draw(sheet)
 for k,f in enumerate(ids):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,im=cap.read();assert ok;h,w=im.shape[:2];rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));rgb=rgb.crop((int(w*.3),int(h*.05),int(w*.7),int(h*.85)));rgb.thumbnail((315,250));sheet.paste(rgb,((k%4)*320,(k//4)*290+30));dr.text(((k%4)*320,(k//4)*290),f'{k}: f{f}',font=font,fill='#18314a')
 cap.release();sheet.save(out/f'{cid}.jpg',quality=96);rows.append(dict(clip_id=cid,label=meta['canonical_label'],frames=ids.tolist(),path=str(path),source_sha256=sha(path),predictions_computed=False))
write(U/'LEFT_COVERAGE_SEARCH.json',dict(rows=rows,purpose='RGB-only search for additional genuinely different left long-finger combinations. No checkpoint or threshold changed after reserved evaluation.'))
