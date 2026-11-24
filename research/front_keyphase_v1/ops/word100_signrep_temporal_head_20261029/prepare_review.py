from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent; OPS=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
idx=read(OPS/'word100_shared_feature_bundle_20261027/visual_inputs/INDEX.json')['rows']
draft=read(OPS/'word100_shared_feature_bundle_20261027/DRAFT_LABELS.json')
seqs=read(OPS/'word100_shape_sequence_bundle_20261027/BUNDLE.json')['sequences']
out=U/'review';out.mkdir(exist_ok=False)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
groups={}
for row in idx:groups.setdefault(row['word_id'],[]).append(row)
review=[]
for number,(word,group) in enumerate(groups.items(),1):
    canvas=Image.new('RGB',(1500,1440),'#edf0f4');draw=ImageDraw.Draw(canvas)
    for rr,row in enumerate(group):
        a=row['alias'];crop=row['display_only_crop_xyxy']
        for cc,ordinal in enumerate([5,8,11]):
            fr=next(x for x in row['frames'] if x['ordinal']==ordinal)
            im=Image.open(fr['file']).convert('RGB').crop(crop)
            im.thumbnail((490,440));x=cc*500+(500-im.width)//2;y=rr*480+30
            canvas.paste(im,(x,y));draw.text((cc*500+10,rr*480+5),f'{a} / n{ordinal} / f{fr["frame_id"]}',font=font,fill='black')
        review.append(dict(alias=a,static=[5,8,11],sheet=f'static_{number:02}.jpg'))
    canvas.save(out/f'static_{number:02}.jpg',quality=94)
for number in range(0,len(seqs),3):
    canvas=Image.new('RGB',(1820,780),'#edf0f4');draw=ImageDraw.Draw(canvas)
    for rr,s in enumerate(seqs[number:number+3]):
        frames=s['frames'];ids={0,6,12}
        for i in range(1,len(frames)):
            for feat in ['index_only','open_four']:
                a,b=frames[i-1]['direct'][feat],frames[i]['direct'][feat]
                if a is not None and b is not None and a!=b:ids.update([i-1,i])
        # All13 thumbnails, preserve chronological order; native hand crops.
        for cc,f in enumerate(frames):
            im=Image.open(f['focus_image']).convert('RGB');im.thumbnail((135,210))
            x=cc*140+(140-im.width)//2;y=rr*260+42
            canvas.paste(im,(x,y));draw.text((cc*140+3,rr*260+22),str(f['frame_id']),font=font,fill='black')
        draw.text((10,rr*260+1),f'{s["source_alias"]} {s["case_id"]} side{s["side"]}',font=font,fill='black')
    canvas.save(out/f'seq_{number//3+1:02}.jpg',quality=94)
(U/'REVIEW_INDEX.json').write_text(json.dumps(dict(static=review,sequences=[dict(case_id=s['case_id'],alias=s['source_alias'],side=s['side']) for s in seqs],pins={str(p):sha(p) for p in [OPS/'word100_shared_feature_bundle_20261027/DRAFT_LABELS.json',OPS/'word100_shape_sequence_bundle_20261027/BUNDLE.json',U/'FROZEN.json']}),ensure_ascii=False,indent=2),encoding='utf8')
print('Prepared12static +12sequence sheets; no training or label changes.')
