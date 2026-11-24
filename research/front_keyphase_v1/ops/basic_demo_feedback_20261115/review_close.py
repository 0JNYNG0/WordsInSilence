from prepare import *
import av
def main():
 (U/'close_review').mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20)
 for r in read(U/'INPUTS.json')['rows']:
  gold=[a for a in read(U/'SOURCE_TRUTH.json')['rows']if a['clip_id']==r['clip_id'] and a['state']>0]
  todo={a['frame']:a['state']for a in gold};pictures={}
  with av.open(r['video']) as c:
   for i,fr in enumerate(c.decode(video=0)):
    if i in todo:
     rgb=fr.to_ndarray(format='rgb24');h,w=rgb.shape[:2];pictures[i]=Image.fromarray(rgb[int(h*.23):int(h*.58),int(w*.27):int(w*.70)])
  for page in range((len(gold)+7)//8):
   anchors=gold[page*8:page*8+8];board=Image.new('RGB',(1600,600),'white');d=ImageDraw.Draw(board)
   for k,a in enumerate(anchors):
    pic=pictures[a['frame']].copy();pic.thumbnail((397,265));x=k%4*400;y=k//4*300;board.paste(pic,(x,y));d.text((x+5,y+267),f'P{r["person"]:02} f{a["frame"]} initial:{a["state"]}',font=font,fill='#152c46')
   board.save(U/'close_review'/f'{r["person"]:02}_{page}.jpg',quality=96)
 print('SEQUENTIAL PTS-NATIVE RGB REVIEW READY')
if __name__=='__main__':main()
