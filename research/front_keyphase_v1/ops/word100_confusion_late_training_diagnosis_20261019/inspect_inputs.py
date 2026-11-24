"""Read public saved observations and prepare RGB evidence, never run MediaPipe."""
from common import *
from collections import Counter
import zipfile,cv2
from PIL import Image,ImageDraw,ImageFont

def longest(mask):
 best=cur=0
 for x in mask:
  cur=cur+1 if x else 0;best=max(best,cur)
 return best
def stats(obs,ix):
 m=obs['usable'][ix];hands=m[:,33:].reshape(-1,2,21);seen=hands.sum(axis=2)>=3;ts=obs['timestamp_us'][ix]
 return {'frames':len(ix),'duration_seconds':float((ts[-1]-ts[0])/1e6) if len(ix)>1 else 0,
 'left_observed_fraction':float(seen[:,0].mean()),'right_observed_fraction':float(seen[:,1].mean()),'both_observed_fraction':float(seen.all(1).mean()),
 'no_hands_fraction':float((~seen.any(1)).mean()),'longest_no_hands_frames':longest(~seen.any(1)),
 'body_arm_valid_fraction':float(m[:,[11,12,13,14,15,16]].mean()),
 'left_wrist_valid_fraction':float(hands[:,0,0].mean()),'right_wrist_valid_fraction':float(hands[:,1,0].mean())}
def main():
 saved=read(U/'SAVED_ANALYSIS.json');rows=read(FACT/'PREP_CONTEXT.json')['rows'];index={r['clip_id']:r for r in rows};pins={}
 labels={r['label_index'] for r in rows if r['clip_id'] in saved['focus']};needed={r['clip_id'] for r in rows if r['role']=='development_only' or r['label_index'] in labels}
 qualities={}
 for cid in sorted(needed):
  row=index[cid];p=BASE/'data/records'/f'{cid}.json';rec=read(p);pins[str(p)]=sha(p)
  q={'label':row['label'],'role':row['role'],'signer':row['signer_id'],'intervals':row['intervals'],'views':{}}
  for v in ['sensor','masked']:
   op=Path(rec['views'][v]['observation']);obs=load(op);pins[str(op)]=sha(op);cp=Path(rec['views'][v]['channel_file']);ch=load(cp)
   whole=np.flatnonzero(np.isin(obs['frame_id'],ch['whole_frame_id']));ts=obs['timestamp_us']/1e6
   core=np.flatnonzero(np.logical_or.reduce([(ts>=a)&(ts<=b) for a,b in row['intervals']]))
   if not len(core):raise ValueError(('no core',cid))
   q['views'][v]={'whole':stats(obs,whole),'annotation_core':stats(obs,core),'whole_frame_range':[int(ch['whole_frame_id'][0]),int(ch['whole_frame_id'][-1])],
    'core_covered_by_model_fraction':float(np.isin(obs['frame_id'][core],ch['whole_frame_id']).mean()),'input_frames':len(ch['x_whole'])}
  qualities[cid]=q
 requests=[]
 for cid in saved['focus']:
  row=index[cid];labelid=row['label_index']
  targets=[r for r in rows if r['label_index']==labelid]
  requests.extend((r['clip_id'],'target_dev' if r['role']=='development_only' else 'target_train') for r in targets if r['role']=='development_only' or r['signer_id']==min(x['signer_id'] for x in targets if x['role']=='train_candidate'))
  wrongs=saved['clips'][cid]['wrong_predictions'];pred=sorted(wrongs,key=lambda k:(-wrongs[k],k))[0]
  cand=sorted((r for r in rows if r['label']==pred and r['role']=='train_candidate'),key=lambda r:(r['signer_id'],r['clip_id']))[0]
  requests.append((cand['clip_id'],'frequent_confusion_train'))
 requests=list(dict.fromkeys(requests));folder=U/'visual';folder.mkdir(exist_ok=True);videos=U/'public_video_cache';videos.mkdir(exist_ok=True)
 font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
 hand_edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
 sheets=[]
 for cid,reason in requests:
  row=index[cid];rec=read(BASE/'data/records'/f'{cid}.json');src=rec['source'];archive=Path(src['archive'])
  if not archive.exists():archive=Path(row['source']['archive'])
  assert archive.is_file(),str(archive)
  out=videos/f'{cid}.mp4'
  if not out.exists():
   with zipfile.ZipFile(archive) as z:data=z.read(src['member'])
   assert hashlib.sha256(data).hexdigest()==src['sha256'];out.write_bytes(data)
  assert sha(out)==src['sha256'];pins[str(out)]=src['sha256']
  cap=cv2.VideoCapture(str(out));assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  lo=min(a for a,b in row['intervals']);hi=max(b for a,b in row['intervals']);times=[max(0,lo-.3)]+np.linspace(lo,hi,6).tolist()+[min((n-1)/fps,hi+.3)]
  obs=load(rec['views']['sensor']['observation']);sheet=Image.new('RGB',(4*400,2*275+85),'white');dr=ImageDraw.Draw(sheet)
  dr.text((10,5),f"{row['label']} · 촬영자 {row['signer_id']} · {reason}",font=font,fill='black');dr.text((10,35),f"{cid} | RGB + 저장된 sensor 관측 | 좌=초록, 우=자홍 | 노랑=입력 하단 경계",font=small,fill='black')
  used=[]
  for k,t in enumerate(times):
   f=min(n-1,max(0,round(t*fps)));cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok,(cid,f)
   src_h,src_w=img.shape[:2];xy=obs['xy_px'];j=int(np.argmin(abs(obs['timestamp_us']/1e6-f/fps)));mask=obs['usable'][j]
   for side,start in enumerate([33,54]):
    color=(50,255,60) if side==0 else (255,60,240)
    for a,b in hand_edges:
     if mask[start+a] and mask[start+b]:cv2.line(img,tuple(np.rint(xy[j,start+a]).astype(int)),tuple(np.rint(xy[j,start+b]).astype(int)),color,3)
    for a in range(21):
     if mask[start+a]:cv2.circle(img,tuple(np.rint(xy[j,start+a]).astype(int)),3,color,-1)
   for a,b in [(11,13),(13,15),(12,14),(14,16)]:
    if mask[a] and mask[b]:cv2.line(img,tuple(np.rint(xy[j,a]).astype(int)),tuple(np.rint(xy[j,b]).astype(int)),(255,220,60),2)
   bottom=rec['calibration']['bottom'];cv2.line(img,(0,bottom),(src_w-1,bottom),(0,230,255),3)
   im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));im.thumbnail((400,225));x=(k%4)*400;y=85+(k//4)*275;sheet.paste(im,(x,y))
   dr.text((x+5,y+228),f"{f/fps:.2f}s f{f}  L{int(mask[33:54].sum())}/21 R{int(mask[54:75].sum())}/21",font=small,fill='black');used.append({'frame':f,'time':f/fps,'observation_frame':int(obs['frame_id'][j]),'left_points':int(mask[33:54].sum()),'right_points':int(mask[54:75].sum())})
  cap.release();sp=folder/f'{cid}.jpg';sheet.save(sp,quality=88);sheets.append({'clip':cid,'reason':reason,'label':row['label'],'signer':row['signer_id'],'sheet':str(sp),'video':str(out),'source_archive':str(archive),'frames':used})
  print(cid,flush=True)
 write(U/'INPUT_QUALITY.json',{'at':now(),'quality':qualities,'sheets':sheets,'pins':pins,'code_sha256':sha(__file__),'mediapipe_executed':False,'inference':False,'user_data_used':False})
if __name__=='__main__':main()
