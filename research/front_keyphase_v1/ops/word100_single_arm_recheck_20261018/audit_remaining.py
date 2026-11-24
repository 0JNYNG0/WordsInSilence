"""Read-only source/observation diagnosis of all eleven unresolved audit events."""
from run import *
import cv2
from PIL import Image,ImageDraw,ImageFont
F=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22)
def main():
    rows={r['clip_id']:r for r in read(U/'CONTEXT.json')['rows']};ann=read(T/'ANNOTATED_COMPARISON.json')
    events=[x for x in ann if x['verdict']=='error_identity' and x['new_side'] is not None and x['new_side']!=x['actual_side']]
    assert len(events)==11;(U/'remaining').mkdir();facts=[]
    contract=read(O.parent/'config'/'contract.json');threshold=contract['usable_rule']['pose_visibility_threshold']
    for x in events:
        r=rows[x['clip_id']];v=x['view'];f=x['index'];i=x['raw_index'];raw=load(r['views'][v]['raw']);p=raw['pose_xy_px'][f];w=int(raw['width'][f]);h=int(raw['height'][f]);conf=raw['pose_conf'][f];joints=[]
        for s in (0,1):
            k=15+s;inside=bool(np.isfinite(p[k]).all() and 0<=p[k,0]<w and 0<=p[k,1]<h)
            joints.append({'side':s,'xy':p[k].tolist(),'inside_strict_image':inside,'observed_output':bool(raw['pose_observed'][f,k]),'usable':bool(raw['pose_usable'][f,k]),'confidence':float(conf[k]),'below_threshold':bool(conf[k]<threshold)})
        palm=raw['hand_xy_px'][f,i,[0,5,9,13,17]];outside=((palm[:,0]<0)|(palm[:,0]>=w)|(palm[:,1]<0)|(palm[:,1]>=h)).tolist()
        video=next((q/'videos'/f"{x['clip_id']}.mp4" for q in (A,P,T) if (q/'videos'/f"{x['clip_id']}.mp4").exists()),None);assert video and sha(video)==r['source']['sha256']
        cap=cv2.VideoCapture(str(video));page=Image.new('RGB',(2400,540),'#15283a')
        for col,delta in enumerate((-1,0,1)):
            ff=max(0,min(len(raw['frame_id'])-1,f+delta));cap.set(cv2.CAP_PROP_POS_FRAMES,int(raw['frame_id'][ff]));ok,im=cap.read();assert ok
            cv2.line(im,(0,int(raw['height'][ff])-1),(im.shape[1]-1,int(raw['height'][ff])-1),(0,220,255),3)
            if col==1:
                for k in range(int(raw['raw_count'][ff])):
                    for pt in raw['hand_xy_px'][ff,k]:cv2.circle(im,tuple(pt.astype(int)),5,(0,255,90) if k==i else (255,190,30),-1)
                for s in (0,1):
                    for a,b in ((11+s,13+s),(13+s,15+s)):
                        pts=raw['pose_xy_px'][ff,[a,b]]
                        if np.isfinite(pts).all():cv2.line(im,tuple(pts[0].astype(int)),tuple(pts[1].astype(int)),(255,255,255) if raw['pose_usable'][ff,[a,b]].all() else (128,128,128),4)
            tile=Image.new('RGB',(800,540),'#15283a');tile.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((800,450)),(0,90));d=ImageDraw.Draw(tile)
            d.text((10,6),f"{x['word']} · {v} f{int(raw['frame_id'][ff])}",font=F,fill='white');d.text((10,35),f"{'문제 프레임: 초록 raw'+str(i) if col==1 else '앞뒤 원본'}",font=F,fill='#d6e7f8')
            d.text((10,63),'흰선: 사용 가능 팔 / 회색선: 신뢰 부족 좌표',font=F,fill='#d6e7f8');page.paste(tile,(800*col,0))
        cap.release();name=f"remaining_{x['id']}.jpg";path=U/'remaining'/name;page.save(path,quality=96)
        facts.append({**{k:x[k] for k in ('id','set','clip_id','word','view','index','raw_index','actual_side','new_side')},'wrists':joints,'palm_outside':outside,'image':str(path),'source_sha256':sha(video),'source_video':str(video)})
    write(U/'REMAINING_DIAGNOSIS.json',{'threshold':threshold,'threshold_source':str(O.parent/'config'/'contract.json'),'threshold_source_sha256':sha(O.parent/'config'/'contract.json'),'events':facts,'all_eleven_included':True,'images_are_diagnostic_only':True,'full_frame_pixels_never_candidate_input':True})
    print('11 remaining source contexts rendered; threshold',threshold)
if __name__=='__main__':main()
