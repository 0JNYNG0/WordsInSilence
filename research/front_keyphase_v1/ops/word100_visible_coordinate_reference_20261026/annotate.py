import sys
from PIL import Image,ImageDraw
import prepare as S
from marks import MARKS,NOTES
U=S.U
def run():
    f=S.read(U/'FROZEN.json');p=S.read(U/'PREPARED.json');allrows=[];sheets=[]
    for clip in p['clips']:
        key=clip['clip_id'].split('_WORD')[1][:4];sheet=Image.new('RGB',(1440,520),'#f6f8fc');sd=ImageDraw.Draw(sheet)
        sd.text((12,6),clip['label']+' | RGB 기준 위치 영역 (예측 좌표 없음)',font=S.font(22),fill='#17253b')
        for i,imrow in enumerate(clip['frames']):
            t=imrow['frame'];im=Image.open(U/imrow['raw']).convert('RGB');d=ImageDraw.Draw(im);marks=MARKS[key][t];boxes=[]
            for h,arr in enumerate(marks):
                for j,box in zip([0,4,8],arr):
                    status='visible_region' if box else 'unclear_withheld'
                    allrows.append(dict(clip_id=clip['clip_id'],label=clip['label'],frame=t,side=['left','right'][h],joint=j,region=box,status=status,reason=NOTES[key] if box is None else 'Approximate anatomical visible region in source RGB; no detector point used.',source=imrow['raw'],source_sha=imrow['raw_sha']))
                    if box:
                        assert 0<=box[0]<box[2]<1920 and 0<=box[1]<box[3]<1080
                        boxes.append(box);c=['#00ebff','#ffae38'][h];d.rectangle(tuple(box),outline=c,width=2)
                        d.text((box[0],box[1]-18),['L','R'][h]+str(j),font=S.font(15),fill=c)
            bound=[min(b[0] for b in boxes)-65,min(b[1] for b in boxes)-65,max(b[2] for b in boxes)+65,max(b[3] for b in boxes)+65]
            crop=im.crop(tuple(bound));crop.thumbnail((464,420));sheet.paste(crop,(i*480+8+(464-crop.width)//2,80))
            sd.text((i*480+12,42),f'frame {t} / L=왼손 R=오른손 / 0손목 4엄지 8검지',font=S.font(14),fill='#17253b')
            im.save(U/'images'/(clip['clip_id']+f'_{t:04d}_regions.png'))
        dest='images/'+clip['clip_id']+'_regions.jpg';sheet.save(U/dest,quality=95);sheets.append(dest)
    assert len(allrows)==108
    S.write('ANNOTATIONS_DRAFT.json',dict(points=allrows,visible=sum(x['region'] is not None for x in allrows),withheld=sum(x['region'] is None for x in allrows),source='Codex RGB approximate regions, not independent human ground truth',sheets=sheets))
    print('ANNOTATIONS_DRAFT',len(allrows),sum(x['region'] is not None for x in allrows),flush=True)
def freeze():
    d=S.read(U/'ANNOTATIONS_DRAFT.json')
    S.write('ANNOTATIONS.json',{**d,'status':'FROZEN_BEFORE_CANDIDATE_SCORING','mark_code_sha':S.sha(U/'marks.py'),'review':'All six region overlays reviewed against source RGB. Prior clip diagnostics known; no candidate positions shown in these new-frame marking views. Not an independent blinded human annotation.'})
    print('ANNOTATIONS_FROZEN',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
