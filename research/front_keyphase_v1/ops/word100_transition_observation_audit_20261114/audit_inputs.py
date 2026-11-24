from pathlib import Path
import json,hashlib,collections
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_terminal_relation_targets_20261114';T=P/'transition_shape';S=O/'word100_goal_state_learning_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
write('PLAN.json',dict(goal='Separate detector observation mismatch from model/target confusion before adding public cross-word bending labels',
 input='Saved public RTMW + RGB and past failed outputs; no old model inference/training repeated',
 fit_records=list(range(2,8)),exposed_probe_records=[8,9],
 phases=['Verify pixel/keypoint mapping and inspect all saved regressions','Review additional FIT expressions and projected finger states','Only after reliable targets, freeze a small comparison; do not weaken old endpoint safeguards','Verify arithmetic and synchronize records'],
 constraints=['No private or mentor data','C/T3-F/common100/web unchanged','No download, camera, Git','Final8 stays sealed','No long hold/minimum displacement requirement'],
 evaluation='Exposed developmental diagnosis is not an independent holdout test. Same Codex review is not expert consensus.'))
labels=read(T/'LABELS.json')['rows'];lookup={(r['clip_id'],r['frame'],r['hand']):r for r in labels}
saved=read(T/'EVALUATION.json')['rows'];parallel=read(T/'parallel/EVALUATION.json')['rows'];cases={}
for r in saved:
 if r['split']=='exposed_probe'and r['label']and r['new']!=r['label']and r['origin']=='original_visible_state':
  key=r['clip_id'],r['frame'],r['hand'];cases[key]=dict(reason=['endpoint_error_or_regression'],prior_outputs=dict(binary=r['old'],three_class=r['new']))
for r in parallel:
 if r['split']=='exposed_probe'and r['label'] and ((r['label']=='transition'and r['middle_state']!=1)or(r['label']!='transition'and r['middle_state']==1)):
  key=r['clip_id'],r['frame'],r['hand'];case=cases.setdefault(key,dict(reason=[],prior_outputs={}));case['reason'].append('parallel_middle_mismatch');case['prior_outputs']['middle_state']=r['middle_state']
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18);panels=[];out=[];arr=np.load(S/'source_arrays.npz')
for (cid,fid,hand),case in sorted(cases.items()):
 r=lookup[cid,fid,hand];rec=read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json')
 assert rec['source_sha256']==r['source_sha256'];raw=np.load(rec['raw_path']);assert sha(Path(rec['raw_path']))==rec['raw_sha256']
 idx=int(np.flatnonzero(raw['frame_id']==fid)[0]);xy=raw['xy_px'][idx];score=raw['score'][idx]
 cap=cv2.VideoCapture(r['source']);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();cap.release();assert ok
 h,w=im.shape[:2];assert (w,h)==(int(raw['source_width']),int(raw['source_height']))
 start=112 if hand=='R'else 91;pts=xy[start:start+21];radius=max(90,float(np.ptp(pts,axis=0).max())*.8)
 center=pts.mean(0);x1=max(0,int(center[0]-radius));y1=max(0,int(center[1]-radius));x2=min(w,int(center[0]+radius));y2=min(h,int(center[1]+radius))
 crop=im[y1:y2,x1:x2].copy();over=crop.copy();q=np.rint(pts-[x1,y1]).astype(int)
 for base in [1,5,9,13,17]:
  for a,b in zip([0,base,base+1,base+2],[base,base+1,base+2,base+3]):cv2.line(over,tuple(q[a]),tuple(q[b]),(0,180,255),1,cv2.LINE_AA)
 for j,pt in enumerate(q):
  cv2.circle(over,tuple(pt),2,(255,80,20),-1);cv2.putText(over,str(j),tuple(pt+2),cv2.FONT_HERSHEY_SIMPLEX,.24,(20,20,20),1)
 panel=Image.new('RGB',(840,395),'white');d=ImageDraw.Draw(panel);d.text((8,6),f'{cid} f{fid} {hand} / RGB: {r["label"]}',font=font,fill='#17283f')
 for j,mat in enumerate([crop,over]):
  rgb=Image.fromarray(cv2.cvtColor(mat,cv2.COLOR_BGR2RGB));rgb.thumbnail((398,330));panel.paste(rgb,(j*420+10,45))
 panels.append(panel)
 # Verify common normalized coordinates by a shoulder-frame similarity transform.
 pi=arr[cid+'_x'][r['array_index'],:102].reshape(51,2);ph=pi[30:51]if hand=='R'else pi[9:30]
 scale=np.linalg.norm(xy[6]-xy[5])/np.linalg.norm(pi[2]-pi[1]);origin=(xy[5]+xy[6])/2
 expected=ph*scale+origin;err=float(np.max(np.abs(expected-pts)))
 assert err<.01,err
 out.append(dict(**r,**case,raw_path=rec['raw_path'],raw_sha256=rec['raw_sha256'],raw_scores=score[start:start+21].tolist(),
  all_points_in_frame=bool(((pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)).all()),mapping_max_px_error=err,
  detector_correctness=None,RGB_annotation_recheck=None,hand_crop=[x1,y1,x2,y2]))
for i in range(0,len(panels),3):
 sheet=Image.new('RGB',(840,395*len(panels[i:i+3])),'white')
 for j,p in enumerate(panels[i:i+3]):sheet.paste(p,(0,j*395))
 sheet.save(U/f'observation_review_{i//3+1}.jpg',quality=95)
write('AUDIT_INPUTS.json',dict(rows=out,source_count=len({r['clip_id']for r in out}),raw_predictions_reused=True,
 source_pins={str(P/'EVALUATION.json'):sha(P/'EVALUATION.json'),str(T/'EVALUATION.json'):sha(T/'EVALUATION.json'),str(T/'parallel/EVALUATION.json'):sha(T/'parallel/EVALUATION.json')}))
print('audit cases',len(out),'sheets',(len(panels)+2)//3)
