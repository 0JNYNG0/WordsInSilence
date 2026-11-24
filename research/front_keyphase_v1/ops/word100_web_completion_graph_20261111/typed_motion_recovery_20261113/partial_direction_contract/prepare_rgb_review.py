"""Fixed FIT counterexample/overlap inspection, no scores or new labels."""
from pathlib import Path
import sys
U=Path(__file__).resolve().parent
sys.path.insert(0,str(U.parent))
from common import *
import cv2
from PIL import Image,ImageDraw,ImageFont
R=U/'rgb_review'

def main():
    R.mkdir(exist_ok=True);assert not(R/'MANIFEST.json').exists()
    a=read(U/'RESULTS.json')
    srcs={r['clip_id']:r for p in [GRAPH/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',PRIOR/'additional_source_review/SOURCE_EPISODES_ADDED.json']for r in read(p)['rows']if r['person'] in range(2,8)}
    groups={}
    for r in a['sign_conflicts']:
        cid=r['clip_id'];span=r['original_span'];groups.setdefault(cid,[]).append(span)
    cases=[]
    for cid,spans in sorted(groups.items()):
        aa=max(0,min(s[0]for s in spans)-2);bb=min(srcs[cid]['frames']-1,max(s[1]for s in spans)+2)
        cases.append(dict(id=cid+'_sign_conflict',clip_id=cid,reason='All unique original source scenes underlying8 FIT projected-sign conflicts; continuous every-frame RGB, no keypoint overlay',ranges=[[aa,bb]],target_spans=sorted(set(tuple(s)for s in spans))))
    for cid in a['full_forward']['partial_within_hold_amplitude_clips']:
        ps=[r for r in a['rows']if r['clip_id']==cid and r['view']=='full'and not r['reverse']and r['label_kind']=='partial'and r['measurement']['eligible']and abs(r['measurement']['delta'])<=a['full_forward']['maximum_absolute_reviewed_hold_delta']]
        p=min(ps,key=lambda r:(r['span'][0],r['span'][1]))
        hs=[r for r in a['rows']if r['clip_id']==cid and r['view']=='full'and r['label_kind']=='hold'and r['measurement']['eligible']]
        h=max(hs,key=lambda r:(abs(r['measurement']['delta']),-r['span'][0],-r['span'][1]))
        ha,hb=h['source_span'];aa,bb=h['span'];hspan=[ha+aa,ha+bb]
        cases.append(dict(id=cid+'_amplitude_overlap',clip_id=cid,reason='First lexicographic full-view partial in existing hold-amplitude overlap, paired with same-source largest-absolute-delta reviewed hold; diagnostic selection, not evaluation',ranges=[p['span'],hspan],
            target_spans=[p['span'],hspan],partial_candidate=dict(bag_index=p['bag_index'],candidate_index=p['candidate_index'],delta=p['measurement']['delta']),hold_candidate=dict(bag_index=h['bag_index'],candidate_index=h['candidate_index'],delta=h['measurement']['delta'])))
    selected=dict(at=stamp(),audit_sha256=sha(U/'RESULTS.json'),cases=cases,scope='FIT02..07 only; no training/labels before visual review; original source RGB unmodified; rendered sheets are derivatives')
    write(R/'SELECTION.json',selected)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for case in cases:
        src=srcs[case['clip_id']];assert sha(src['source'])==src['source_sha256'];cap=cv2.VideoCapture(src['source']);assert cap.isOpened()
        fw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));fh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));nf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Same fixed central-half horizontal crop shows torso/arms/hands; also
        # retain a full-frame first/last pair to inspect crop coverage.
        crop=[fw//4,0,3*fw//4,fh]
        tiles=[]
        for rg,(aa,bb) in enumerate(case['ranges']):
            for f in range(aa,bb+1):
                cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);im=Image.fromarray(rgb)
                if f in [aa,bb]:
                    pp=R/f"{case['id']}_range{rg}_frame{f}_full.jpg";im.save(pp,quality=95)
                im=im.crop(tuple(crop));im.thumbnail((300,340));tile=Image.new('RGB',(320,375),'white');tile.paste(im,((320-im.width)//2,30));ImageDraw.Draw(tile).text((10,6),f'range {rg+1} | frame {f}',font=font,fill='#14243a');tiles.append(tile)
        cap.release();cols=6;nr=(len(tiles)+cols-1)//cols
        sheet=Image.new('RGB',(cols*320,55+nr*375),'#e5e9ef');dd=ImageDraw.Draw(sheet);dd.text((12,8),case['id'],font=font,fill='#14243a');dd.text((12,32),'Ordered left to right, top to bottom; no skeleton overlay. Range 1 partial/conflict; range 2 reviewed hold.',font=small,fill='#14243a')
        for k,im in enumerate(tiles):sheet.paste(im,((k%cols)*320,55+(k//cols)*375))
        dest=R/f"{case['id']}.jpg";sheet.save(dest,quality=93)
        case.update(source=src['source'],source_sha256=src['source_sha256'],source_dimensions=[fw,fh],source_frames=nf,crop=crop,path=str(dest),sha256=sha(dest),reviewed=False)
        print(case['id'],case['ranges'],str(dest),flush=True)
    write(R/'MANIFEST.json',dict(at=stamp(),selection_sha256=sha(R/'SELECTION.json'),cases=cases,private_access=False,model_inference=False))

if __name__=='__main__':main()
