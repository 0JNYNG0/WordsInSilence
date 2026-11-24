"""Finite direct-RGB witness set for a new wrist-translation target."""
from audit_saved import *
import cv2
from PIL import Image,ImageDraw,ImageFont
def main():
    files=[G/'episode_stage_contrast/RGB_CONTRAST_LABELS.json',G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json',G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json']
    co={r['clip_id']:r for r in read(files[0])['rows']};co.update({r['clip_id']:r for r in read(files[1])['contrasts']})
    sources={r['clip_id']:r for p in files[1:]for r in read(p)['rows']if r['role']=='fit'and r['person']in range(2,8)}
    old={r['clip_id']:r for r in read(U/'LABEL_CONTRACT_AUDIT.json')['panels']}
    records=[];pending=[]
    for cid,s in sorted(sources.items()):
        for kind in['entry','returning']:
            values=co[cid][kind]
            if not values:continue
            frame=min(values);records.append(dict(clip_id=cid,person=s['person'],word_id=s['word_id'],kind=kind,frame=frame))
        if s['hold_sample_frames']:
            vals=s['hold_sample_frames'];records.append(dict(clip_id=cid,person=s['person'],word_id=s['word_id'],kind='held_posture_review',frame=vals[len(vals)//2]))
        for k,e in enumerate(s['events']):
            vals=e['progression_witness_frames'];records.append(dict(clip_id=cid,person=s['person'],word_id=s['word_id'],kind=f'main_{k}',frame=vals[len(vals)//2],old_event=e))
    for r in records:
        s=sources[r['clip_id']];r.update(source=s['source'],source_sha256=s['source_sha256'],review_frames=[max(0,min(s['frames']-1,r['frame']+x))for x in[-4,-2,0,2,4]],label_scope='Whether this short native-time window visibly contains movement of this wrist relative to torso; not exact central biological onset, finger activity, wrist rotation, clause identity or correctness.')
        r['key']=f"{r['clip_id']}:{r['kind']}:{r['frame']}"
        previous=old.get(r['clip_id']);a=next((a for a in previous['anchors']if a['kind']==r['kind']and a['frame']==r['frame']),None)if previous else None
        if a:r['panel']=previous['path'];r['panel_row']=previous['anchors'].index(a);r['reused_existing_direct_review']=True
        else:pending.append(r)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
    for offset in range(0,len(pending),4):
        pack=pending[offset:offset+4];canvas=Image.new('RGB',(1600,len(pack)*330),'white');draw=ImageDraw.Draw(canvas);dest=U/f'physical_review_{offset//4+1:02}.jpg'
        for row,r in enumerate(pack):
            draw.text((10,row*330+3),f"{r['clip_id']} {r['kind']} centre {r['frame']}",font=font,fill='#16304a');cap=cv2.VideoCapture(r['source']);assert cap.isOpened();assert sha(r['source'])==r['source_sha256']
            for col,fr in enumerate(r['review_frames']):
                cap.set(cv2.CAP_PROP_POS_FRAMES,fr);ok,im=cap.read();assert ok
                h,w=im.shape[:2];pic=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).crop((int(.14*w),int(.04*h),int(.86*w),int(.90*h)));pic.thumbnail((310,270));canvas.paste(pic,(col*320+(320-pic.width)//2,row*330+32));draw.text((col*320+10,row*330+303),str(fr),font=font,fill='#16304a')
            cap.release();r.update(panel=str(dest),panel_row=row,reused_existing_direct_review=False)
        canvas.save(dest,quality=93)
    write('PHYSICAL_REVIEW_MANIFEST.json',dict(status='AWAIT_DIRECT_RGB_LABELS',records=records,window_count=len(records),hand_cells=2*len(records),panels={r['panel']:sha(r['panel'])for r in records},
        sources={cid:dict(source=s['source'],source_sha256=s['source_sha256'])for cid,s in sources.items()},
        labels='M=clearly translating wrist in body-relative image plane within window; H=visibly maintained wrist placement through window; U=uncertain. Body wobble not automatically M; palm rotation/fingers not automatically H or absence. Annotation independent from pose values and candidate predictions.',
        no_dense_interpolation=True,no_auto_legacy_label_copy=True,no_new_learning=True))
    print(json.dumps(dict(windows=len(records),hand_cells=2*len(records),newpanels=(len(pending)+3)//4,reused=sum(r['reused_existing_direct_review']for r in records))))
if __name__=='__main__':main()
