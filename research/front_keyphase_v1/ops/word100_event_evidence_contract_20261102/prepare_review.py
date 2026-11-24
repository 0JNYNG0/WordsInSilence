"""Blind-to-predictions FIT windows; no label changes or neural inference."""
from pathlib import Path
import json,hashlib,math
import cv2
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
OUT=Path('W:/WordsInSilence_derived/word100_event_evidence_contract_20261102')
D=O/'word100_action_episode_supervision_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    assert not (U/'REVIEW_MANIFEST.json').exists() and not (U/'FROZEN.json').exists()
    source=read(D/'ACTION_TARGETS.json')['rows'];plans=[]
    for r in source:
        w=r['word_id'];person=r['person'];pts=r['points']
        if w=='WORD1133':left,right=3,4;purpose='traversal and wrist-region relation'
        elif person in [1,5,9] and w=='WORD0739':left,right=1,2;purpose='relocation and forearm reorientation'
        elif person in [1,5,9] and w=='WORD2523':left,right=6,7;purpose='convergence relation and separation movement'
        elif person in [1,5,9] and w=='WORD1567':left=right=5;purpose='alternating movement inside one recorded event'
        else:continue
        if left==right:
            fs=sorted(p['frame'] for p in pts if p['action_family']==left);lo=max(0,fs[len(fs)//2]-6);hi=min(r['frames']-1,lo+12)
        else:
            lo=max(0,max(p['frame'] for p in pts if p['action_family']==left)-5)
            hi=min(r['frames']-1,min(p['frame'] for p in pts if p['action_family']==right)+5)
        assert r['person'] in range(1,10) and 8<=hi-lo+1<=32
        plans.append(dict(alias=f'A{len(plans)+1:02d}',clip_id=r['clip_id'],word_id=w,person=person,source=r['source'],source_sha256=r['source_sha256'],frames=list(range(lo,hi+1)),purpose=purpose))
    assert len(plans)==18 and sum(p['word_id']=='WORD1133' for p in plans)==9
    write(U/'FROZEN.json',dict(plans=plans,selection='Pre-existing FIT witness spans only; no model prediction/DEV error or confidence in selection or sheets.',
        review_fields=['observed_change','body_relative_relation','main_or_return_context','simultaneous_evidence','boundary_uncertainty'],
        values='yes/no/unknown with native evidence frames; no mandatory pause or new semantic correctness claim',
        pins={str(p):sha(p) for p in [U/'prepare_review.py',D/'ACTION_TARGETS.json',D/'review_1133.py']},whole_goal_complete=False))
    reports=[]
    for p in plans:
        assert sha(p['source'])==p['source_sha256'];cap=cv2.VideoCapture(p['source']);assert cap.isOpened();want=set(p['frames']);images=[];frames=[];f=0
        while f<=max(want):
            ok,bgr=cap.read();assert ok
            if f in want:
                ms=float(cap.get(cv2.CAP_PROP_POS_MSEC));dest=OUT/'frames'/p['alias']/f'f{f:04d}.jpg';dest.parent.mkdir(parents=True,exist_ok=True)
                assert not dest.exists();assert cv2.imwrite(str(dest),bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
                h,w=bgr.shape[:2];crop=bgr[int(h*.10):int(h*.92),int(w*.18):int(w*.86)]
                tile=np.full((470,600,3),245,np.uint8);thumb=cv2.resize(crop,(600,430));tile[40:470]=thumb
                cv2.putText(tile,f"{p['alias']}  f{f}   {ms/1000:.3f}s",(12,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,20),2)
                images.append(tile);frames.append(dict(frame=f,seconds=ms/1000,image=str(dest),sha256=sha(dest),full_extent=[w,h],review_crop=[int(w*.18),int(h*.10),int(w*.86),int(h*.92)]))
            f+=1
        cap.release();assert len(frames)==len(want) and all(b['seconds']>a['seconds'] for a,b in zip(frames,frames[1:]))
        sheets=[]
        for start in range(0,len(images),9):
            batch=images[start:start+9];sheet=np.full((470*math.ceil(len(batch)/3),1800,3),225,np.uint8)
            for i,tile in enumerate(batch):sheet[(i//3)*470:(i//3+1)*470,(i%3)*600:(i%3+1)*600]=tile
            dest=OUT/'sheets'/f"{p['alias']}_{start//9+1}.jpg";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();assert cv2.imwrite(str(dest),sheet,[cv2.IMWRITE_JPEG_QUALITY,93])
            sheets.append(dict(path=str(dest),sha256=sha(dest),frames=p['frames'][start:start+9]))
        reports.append(dict(**p,images=frames,sheets=sheets))
    write(U/'REVIEW_MANIFEST.json',dict(rows=reports,clips=len(reports),native_images=sum(len(r['images']) for r in reports),no_model_inference=True,no_new_targets=True,review_only_crop='Full original frame also retained. Crop is visualization only, not detection/training input.',whole_goal_complete=False))
    print('review windows',len(reports),'native images',sum(len(r['images']) for r in reports))
if __name__=='__main__':main()
