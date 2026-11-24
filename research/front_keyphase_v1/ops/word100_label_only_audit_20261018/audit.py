"""Public-only visual audit inventory. No changes to any assigner or model."""
import sys,json,hashlib,zipfile,shutil,os,time,traceback
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;PREV=O/'word100_assignment_conflict_20261018';ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def inventory():
 c=read(PREV/'CONTEXT.json');assert read(PREV/'FINAL_AUDIT.json')['status']=='PASS';assert not (PREV/'RUNNING.json').exists()
 for d in ('before','videos','frames','sheets','details'):(U/d).mkdir(exist_ok=False)
 for p in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json'):shutil.copy2(p,U/'before'/p.name)
 pins=dict(c['pins']);pins[str(PREV/'CHANGES.json')]=sha(PREV/'CHANGES.json');pins[str(U/'audit.py')]=sha(U/'audit.py')
 rows=[];events=[];hist=Counter();changed={(x['clip_id'],x['view'],x['index'],x['raw_index']) for x in read(PREV/'CHANGES.json')}
 for r in c['rows']:
  used=False
  assert r['role']=='train_candidate' and 1<=r['signer_id']<=9
  for v in ('masked','sensor'):
   tr=read(Path(r['views'][v]['assignment']).with_name('trace.json'));raw=None
   for f,d in enumerate(tr):
    for i,why in enumerate(d['reasons']):
     if why!='label_without_close_pose' or (r['clip_id'],v,f,i) in changed:continue
     if raw is None:raw=load(r['views'][v]['raw'])
     used=True;s=d['assigned_sides'][i];n=int(raw['raw_count'][f]);prior=[ff for ff in range(f) if s in tr[ff]['assigned_sides']];future=[ff for ff in range(f+1,len(tr)) if s in tr[ff]['assigned_sides']];xy=raw['hand_xy_px'][f,i];h=int(raw['height'][f]);ts=int(raw['timestamp_us'][f]);pastgap=(ts-int(raw['timestamp_us'][prior[-1]]))/1e6 if prior else None
     event={'id':len(events)+1,'clip_id':r['clip_id'],'word':r['label'],'view':v,'index':f,'frame_id':int(raw['frame_id'][f]),'timestamp_us':ts,'raw_index':i,'side':s,'label_score':float(raw['raw_score'][f,i]),'raw_count':n,'trace':d,'prior_same_side_frame':prior[-1] if prior else None,'prior_gap_s':pastgap,'next_same_side_frame':future[0] if future else None,'height':h,'wrist_xy':xy[0].tolist(),'hand_y_minmax':[float(np.min(xy[:,1])),float(np.max(xy[:,1]))],'inside_points':int(((xy[:,0]>=0)&(xy[:,0]<=raw['width'][f])&(xy[:,1]>=0)&(xy[:,1]<=h)).sum()),'pose_wrist_usable':raw['pose_usable'][f,[15,16]].tolist(),'first_assignment_for_side':not prior,'after_expired_gap':bool(pastgap is not None and pastgap>.5)}
     events.append(event);hist[v]+=1;hist['first_assignment' if not prior else 'reappearance_or_existing']+=1
     if event['after_expired_gap']:hist['after_expired_gap']+=1
  if used:rows.append(r)
 assert len(events)==456
 write(U/'CONTEXT.json',{'authorization':'남은 라벨단독456건 공개원본 분류분석 및 정상교차보존 공통보완안 보고. 기존C유지/사용자미사용/학습대기.','at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'rows':rows,'pins':pins,'training':0,'assignment_code_changes':False,'user_mentor_dev_heldout_reads':0,
 'review_protocol':['Audit every remaining event by original RGB and highlighted raw hypothesis; inspect neighbors when ambiguous.','normal = visible hand belongs to assigned anatomical side; this does not certify every finger coordinate.','error = visible opposite hand/duplicate or landmark hypothesis not on a hand; distinguish identity from landmark/finger quality.','uncertain = occlusion/boundary/motion prevents safe visual identity conclusion; never force binary labels.','Original full RGB below crop may be used for offline anatomical annotation only, never proposed as online webcam evidence.','No classifier output or user footage may select cases or rules. The audit is on known public TRAIN, not independent unseen accuracy.','All456 records must have final reviewed/uncertain labels and evidence paths; no silent exclusion.','Proposed common policies only, no new training or production code in this unit.'],
 'review_operator':'Codex visual review, not independent human anatomical annotations.'})
 write(U/'EVENTS.json',events);write(U/'INVENTORY.json',{'events':len(events),'clips':len(rows),'words':len(set(e['word'] for e in events)),'histogram':dict(hist),'definition':'Each item is one raw hand candidate in one frame/view, not one video or one person.'});print(json.dumps(read(U/'INVENTORY.json'),ensure_ascii=False),flush=True)
def video(r,archives):
 p=U/'videos'/f"{r['clip_id']}.mp4";s=r['source']
 if p.exists():assert sha(p)==s['sha256'];return p
 reuse=PREV/'evidence'/p.name
 if reuse.exists():data=reuse.read_bytes()
 else:
  if s['archive'] not in archives:archives[s['archive']]=zipfile.ZipFile(s['archive'])
  z=archives[s['archive']];zi=z.getinfo(s['member']);assert zi.CRC==s['crc32'] and zi.file_size==s['bytes'];data=z.read(s['member'])
 assert hashlib.sha256(data).hexdigest()==s['sha256'];p.write_bytes(data);return p
FONT=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
def panel(im,e,raw):
 f=e['index'];i=e['raw_index'];pts=raw['hand_xy_px'][f,i];pic=im.copy();h=e['height'];w=im.shape[1]
 # Original below-boundary pixels stay visible for audit, not detector input.
 cv2.line(pic,(0,h),(w-1,h),(0,210,255),3)
 for q in pts:
  if np.isfinite(q).all():cv2.circle(pic,tuple(q.astype(int)),5,(0,255,80),-1)
 cv2.circle(pic,tuple(pts[0].astype(int)),15,(0,255,80),3)
 shoulders=raw['pose_xy_px'][f,[11,12]];cx=int(np.nanmean(shoulders[:,0])) if np.isfinite(shoulders).all() else w//2
 x0=max(0,min(w-1080,cx-540));crop=pic[:,x0:x0+1080];p=Image.new('RGB',(500,560),'#122334');p.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((500,500)),(0,60));d=ImageDraw.Draw(p)
 d.text((8,4),f"#{e['id']:03} {'L' if e['side']==0 else 'R'}  {e['word']}  {e['clip_id'].split('_')[3]} f{e['frame_id']}",font=FONT,fill='white')
 d.text((8,30),f"{e['view']} · 초록=해당후보 · 노랑=검출하단",font=FONT,fill='#b9dae7');return p
def render():
 c=read(U/'CONTEXT.json');ev=read(U/'EVENTS.json');by=defaultdict(list)
 for e in ev:by[e['clip_id']].append(e)
 archives={};write(U/'RUNNING.json',{'pid':os.getpid(),'task':'public RGB evidence render, no inference/training'});start=time.monotonic()
 try:
  for n,r in enumerate(c['rows'],1):
   cap=cv2.VideoCapture(str(video(r,archives)));raws={v:load(r['views'][v]['raw']) for v in set(e['view'] for e in by[r['clip_id']])}
   for e in by[r['clip_id']]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,e['frame_id']);ok,im=cap.read();assert ok,(r['clip_id'],e['frame_id']);panel(im,e,raws[e['view']]).save(U/'frames'/f"{e['id']:03}.jpg",quality=95)
   cap.release()
   if n%50==0:print('Public source rendered',n,'/',len(c['rows']),flush=True)
  for starti in range(0,len(ev),12):
   page=Image.new('RGB',(1500,2240),'#122334')
   for j,e in enumerate(ev[starti:starti+12]):page.paste(Image.open(U/'frames'/f"{e['id']:03}.jpg"),(j%3*500,j//3*560))
   page.save(U/'sheets'/f'{starti//12+1:02}.jpg',quality=95)
  write(U/'RENDER_RESULT.json',{'status':'COMPLETE','clips':len(c['rows']),'events':len(ev),'sheets':38,'seconds':time.monotonic()-start});(U/'RUNNING.json').unlink();print('456 public evidence panels complete',flush=True)
 except BaseException:
  if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'traceback':traceback.format_exc()})
  raise
 finally:
  for z in archives.values():z.close()
if __name__=='__main__':
 if sys.argv[1]=='inventory':inventory()
 elif sys.argv[1]=='render':render()
