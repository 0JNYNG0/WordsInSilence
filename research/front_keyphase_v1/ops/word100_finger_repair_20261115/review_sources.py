from common import *
from PIL import Image,ImageDraw,ImageFont
rr=read(E/'annotation_review/EXTERNAL_LABELS.json')['rows'];selected=[r for r in rr if r['side']=='R'and any(y>=0 for y in r['y'])]
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
for offset in range(0,len(selected),12):
 board=Image.new('RGB',(1200,1005),'white');d=ImageDraw.Draw(board)
 for k,r in enumerate(selected[offset:offset+12]):
  assert sha(r['image'])==r['image_sha256'];im=Image.open(r['image']).convert('RGB');box=r['crop_boxes']['R'];crop=im.crop(box);crop.thumbnail((288,260));x=(k%4)*300;y=(k//4)*335;board.paste(crop,(x,y+58));d.text((x+4,y+5),f"{offset+k} {r['word']} ID{r['person']:02} f{r['frame']}\nE/B/F/U={r['y']}",font=font,fill='#163148')
 board.save(U/f'new_source_review_{offset//12}.jpg',quality=96)
write(U/'SOURCE_REVIEW_INPUTS.json',dict(rows=selected,definition=read(E/'annotation_review/ORDINARY_LABELS.json')['schema'],provenance='Previously reviewed public data; directly rechecked before current candidate learning. Separate personas1/4/7FIT versus11/12/13DEV.',reviewer='pending'))
print(len(selected))
