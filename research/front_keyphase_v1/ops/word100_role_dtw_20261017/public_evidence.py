import json,hashlib,zipfile
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent; B=U.parents[1]
read=lambda p:json.loads(Path(p).read_text('utf-8'))
sha=lambda b:hashlib.sha256(b).hexdigest()
bank=read(B/'ops/word100_multiref_20261014/BASELINE.json')
manifest=read(B/'ops/word100_training_20261013/MANIFEST.json'); byid={r['clip_id']:r for r in manifest}
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
sources=[]; archives={}
(U/'public_videos').mkdir(exist_ok=True)
for code in ['WORD0949','WORD2177','WORD2551','WORD0500']:
    sheet=Image.new('RGB',(1440,9*148),'#edf3f8');d=ImageDraw.Draw(sheet)
    for j,ref in enumerate(bank['words'][code]['references']):
        rec=read(B/f'ops/word100_body_boundary_20261017/data/records/{ref["clip_id"]}.json'); src=rec['source']
        assert ref['role']=='train_candidate' and ref['signer_id'] in range(1,10)
        path=U/'public_videos'/(ref['clip_id']+'.mp4')
        if not path.exists():
            if src['archive'] not in archives: archives[src['archive']]=zipfile.ZipFile(src['archive'])
            data=archives[src['archive']].read(src['member']);assert sha(data)==src['sha256'];path.write_bytes(data)
        assert sha(path.read_bytes())==src['sha256']
        cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        for k,idx in enumerate(np.linspace(0,n-1,8).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,bgr=cap.read();assert ok
            sheet.paste(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((180,101)),(k*180,j*148+24))
            d.text((k*180+4,j*148+126),f'{idx/fps:.2f}s',font=font,fill='#23415c')
        cap.release();d.text((4,j*148),f'{bank["words"][code]["label"]} TRAIN {ref["signer_id"]:02}',font=font,fill='#23415c')
        sources.append({'word_id':code,'clip_id':ref['clip_id'],'signer_id':ref['signer_id'],'source_sha256':src['sha256'],'path':str(path),'scope':'entire clip8 sampled RGB frames; no user data','original':src})
    sheet.save(U/(code+'_TRAIN9.jpg'),quality=93);print(code,flush=True)
for z in archives.values():z.close()
(U/'PUBLIC_EVIDENCE.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),'utf-8')
