"""Evaluation-only new action labels; no new physical/role truth or training."""
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit import *

def main():
    assert not (D/'DEV_REVIEW_MANIFEST.json').exists()
    source=[r for r in read(P/'INPUT_MANIFEST.json')['rows'] if r['role']!='fit']
    assert len(source)==12 and {r['person'] for r in source}=={11,12,13}
    poses={r['clip_id']:r for r in read(S/'PHYSICAL_STATE_TARGETS.json')['rows']};roles={r['clip_id']:r for r in read(A/'ROLE_TARGETS.json')['rows']}
    plans=[]
    for r in source:
        cid=r['clip_id'];physical=[p['frame'] for p in poses[cid]['points'] if p['mask']];core=[p['frame'] for p in roles[cid]['points'] if p['mask'] and p['label']==1]
        lo=max(0,min(physical+core)-9);hi=min(r['frames']-1,max(physical+core)+12)
        indices=sorted(set([0,r['frames']-1]+np.rint(np.linspace(lo,hi,24)).astype(int).tolist()+physical))
        plans.append(dict(clip_id=cid,word_id=r['word_id'],person=r['person'],label=r['label'],source=r['source'],source_sha256=r['source_sha256'],frames=r['frames'],indices=indices))
    exclusive(D/'DEV_REVIEW_INTENT.json',dict(at=clock(),rows=plans,purpose='Evaluation-only sparse event-family labels before D predictions; no existing physical/role labels changed, no training or candidate selection',selection='Same24uniform rule as FIT plus previous physical anchors, no predictions or error-driven frames',warning='Repeatedly exposed DEV, not independent final holdout; same Codex annotation. H14/15 untouched.'))
    dest=D/'dev_review_sheets';dest.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14);rows=[]
    for r in plans:
        assert sha(r['source'])==r['source_sha256'];cap=cv2.VideoCapture(r['source']);assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS)
        idx=r['indices'];im=Image.new('RGB',(1596,72+190*((len(idx)+5)//6)),'#f4f5f7');draw=ImageDraw.Draw(im)
        draw.text((8,8),f"{r['label']} | {r['clip_id']} | 평가용 공개 DEV 원본",font=font,fill='#111827')
        draw.text((8,35),'후보 예측 없음 · 행동 구간 평가용 주석 · 학습에 사용하지 않음',font=font,fill='#111827')
        for k,f in enumerate(idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok;h,w=bgr.shape[:2]
            small=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop((int(w*.15),0,int(w*.85),h));small.thumbnail((258,161));x=(k%6)*266+4;y=72+(k//6)*190
            im.paste(small,(x,y+25));draw.text((x,y),f'f{f} {f/fps:.3f}s',font=font,fill='#111827')
        cap.release();path=dest/(r['clip_id']+'.jpg');im.save(path,quality=94);rows.append(dict(**r,sheet=str(path),sheet_sha256=sha(path)))
    exclusive(D/'DEV_REVIEW_MANIFEST.json',dict(at=clock(),rows=rows,reviewed=False,evaluation_only=True));print('Prepared12DEV chronological sheets; not yet reviewed',flush=True)
if __name__=='__main__':main()
