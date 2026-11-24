from prepare import *
import av

def main():
 V=U/'source_corrected';r=next(r for r in read(V/'CONFIRMATION_INPUTS.json')['rows']if r['person']==15);raw=np.load(r['raw']);frames=[53,59,68,77]
 font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20);im=Image.new('RGB',(1600,820),'white');d=ImageDraw.Draw(im)
 with av.open(r['video'])as c:
  for i,fr in enumerate(c.decode(video=0)):
   if i not in frames:continue
   j=frames.index(i);rgb=fr.to_image();p=raw['xy_px'][i,112:133];valid=np.isfinite(p).all(1);lo=p[valid].min(0)-50;hi=p[valid].max(0)+50
   box=(max(0,int(lo[0])),max(0,int(lo[1])),min(rgb.width,int(hi[0])),min(rgb.height,int(hi[1])));crop=rgb.crop(box);factor=min(390/crop.width,350/crop.height);sz=(int(crop.width*factor),int(crop.height*factor));crop=crop.resize(sz);im.paste(crop,(j*400,35));im.paste(crop,(j*400,440))
   for finger,k in enumerate([1,5,9,13,17]):
    ids=[0]+list(range(k,k+4));xy=[(j*400+float((p[t,0]-box[0])*factor),440+float((p[t,1]-box[1])*factor))for t in ids]
    col=['#a56628','#245aba','#009880','#a24085','#555555'][finger]
    d.line(xy,fill=col,width=3)
    for x,y in xy:d.ellipse((x-3,y-3,x+3,y+3),fill=col)
   d.text((j*400+8,5),f'공개15 / 원본 프레임 {i}',font=font,fill='#193147')
 d.text((10,399),'위: 원본 오른손  /  아래: 같은 프레임의 RTMW 관절',font=font,fill='#193147');im.save(U/'EDGE_FAILURE_REVIEW.jpg',quality=95)
if __name__=='__main__':main()
