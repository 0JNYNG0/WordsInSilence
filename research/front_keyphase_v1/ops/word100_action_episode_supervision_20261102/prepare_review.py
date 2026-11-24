"""Chronological FIT-only review; no predictions, no detector or training."""
import os,shutil,traceback
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit import *

def main():
    assert not (D/'REVIEW_MANIFEST.json').exists();assert shutil.disk_usage(D).free>1024**3
    aud=read(D/'INPUT_AUDIT.json');sr={r['clip_id']:r for r in read(S/'PHYSICAL_STATE_TARGETS.json')['rows']};ar={r['clip_id']:r for r in read(A/'ROLE_TARGETS.json')['rows']}
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14);title=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    plans=[]
    for r in aud['rows']:
        cid=r['clip_id'];physical=[p['frame'] for p in sr[cid]['points'] if p['mask']];core=[p['frame'] for p in ar[cid]['points'] if p['mask'] and p['label']==1]
        endpoints=[f for ep in r['motion_episodes'] for f in ep['frames']]
        lo=max(0,min(physical+core)-9);hi=min(r['frames']-1,max(physical+core)+12)
        indices=sorted(set([0,r['frames']-1]+np.rint(np.linspace(lo,hi,24)).astype(int).tolist()+physical+endpoints))
        plans.append(dict(**r,review_frame_indices=indices,review_bounds=[lo,hi],frame_selection='24uniform times through existing observed-action support plus physical anchors, reviewed motion endpoints and video endpoints. No candidate predictions or DEV errors.'))
    exclusive(D/'REVIEW_INTENT.json',dict(at=clock(),rows=plans,split='FIT01-09 only',new_semantic_labels=0,fields_to_review=['observed action episode','guide phrase support','uncertain temporal edges','same posture in another action'],no_new_dense_truth_from_selection=True))
    (D/'review_frames').mkdir(exist_ok=True);(D/'review_sheets').mkdir(exist_ok=True);rows=[]
    for j,r in enumerate(plans):
        assert sha(r['source'])==r['source_sha256'];cap=cv2.VideoCapture(r['source']);assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert n==r['frames'] and abs(fps-30)<1
        frames=[];cid=r['clip_id'];ix=r['review_frame_indices'];columns=6;cw,ch=266,190;im=Image.new('RGB',(cw*columns,72+ch*((len(ix)+columns-1)//columns)),'#f4f5f7');draw=ImageDraw.Draw(im)
        draw.text((10,8),f"{r['label']} | {cid} | 시간순 공개 FIT 원본",font=title,fill='#111827');draw.text((10,36),'이전 예측 없음 · 화면 중앙 확대는 보기용 · 모든 원본 시간은 보존 · 아직 새 동작 정답 아님',font=font,fill='#111827')
        for k,idx in enumerate(ix):
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,bgr=cap.read();assert ok
            h,w=bgr.shape[:2];rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);full=Image.fromarray(rgb);full.thumbnail((1280,720))
            path=D/'review_frames'/f'{cid}_f{idx:04d}.jpg';assert not path.exists();full.save(path,quality=92)
            # Body/arms context, not a hand-only detection crop. Originals remain
            # separately available; this crop only lays out the contact sheet.
            small=Image.fromarray(rgb).crop((int(w*.15),0,int(w*.85),h));small.thumbnail((cw-8,ch-29));x=(k%columns)*cw+4;y=72+(k//columns)*ch
            im.paste(small,(x,y+25));draw.text((x,y),f'f{idx}  {idx/fps:.3f}s',font=font,fill='#111827')
            frames.append(dict(frame=idx,seconds=idx/fps,path=str(path),sha256=sha(path)))
        cap.release();dest=D/'review_sheets'/f'{cid}.jpg';im.save(dest,quality=94)
        rows.append(dict(clip_id=cid,word_id=r['word_id'],person=r['person'],source=r['source'],source_sha256=r['source_sha256'],frames=frames,sheet=str(dest),sheet_sha256=sha(dest),reviewed=False))
        write(D/'REVIEW_PROGRESS.json',dict(at=clock(),status='PREPARING',pid=os.getpid(),done=j+1,total=36));print('prepared review',j+1,cid,len(frames),flush=True)
    exclusive(D/'REVIEW_MANIFEST.json',dict(at=clock(),status='PREPARED_NOT_VISUALLY_REVIEWED',rows=rows,frames=sum(len(r['frames']) for r in rows),sheets=36,predictions_shown=False,new_labels=0))
    write(D/'REVIEW_PROGRESS.json',dict(at=clock(),status='PREPARED_NOT_VISUALLY_REVIEWED',active=None,done=36,total=36))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (D/'REVIEW_FIRST_FAILURE.txt').exists():(D/'REVIEW_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
