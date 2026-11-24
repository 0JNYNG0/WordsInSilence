from pathlib import Path
import hashlib,json,cv2,numpy as np
U=Path(__file__).resolve().parent;O=U.parent
P=O/'word100_feedback_recovery_20261113/motion_recovery'
A=O/'word100_actionable_feedback_20261113'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
def put(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
z=np.load(P/'PHYSICAL_INPUTS.npz');rows=read(P/'PHYSICAL_INDEX.json')['rows'];labels=read(P/'PHYSICAL_SOURCE_LABELS.json')['rows']
losskeys=['NIA_SL_WORD2523_REAL03_F:held_posture_review:82:R','NIA_SL_WORD0739_REAL02_F:main_1:76:R','NIA_SL_WORD1133_REAL06_F:main_0:48:R']
stats=[]
for i,r in enumerate(rows):
 if r['view']!='full' or r['reverse']:continue
 q=z['q'][i].astype(float);v=z['valid'][i]&z['exists'][i];p=q[v];d=np.diff(q,axis=0)[v[:-1]&v[1:]]
 path=np.linalg.norm(d,axis=1).sum();net=np.linalg.norm(p[-1]-p[0]);maxd=np.linalg.norm(p[:,None]-p[None,:],axis=2).max()
 row=dict(source_key=r['source_key'],truth=r['label'],q=p.tolist(),max_displacement=float(maxd),path=float(path),net=float(net),net_path=float(net/max(path,1e-8)),
  observed=r['observation'],source_frame_ids=r['source_frames'],row_index=i)
 for name,directory in [('previous',P/'physical_runs'),('invariant',A/'motion_runs')]:
  row[name]=[float(np.load(directory/f"{r['person']}_{seed}"/'predictions.npz')['logits'][i]) for seed in [42,43,44]]
 stats.append(row)
targets=[]
for key in losskeys:
 r=next(s for s in stats if s['source_key']==key)
 others=[s for s in stats if s['truth']=='H' and s['source_key'].split(':')[0]!=key.split(':')[0]]
 nearest=sorted(others,key=lambda s:abs(s['max_displacement']-r['max_displacement']))[:2]
 targets.append(dict(target=r,held_nearest_displacement=nearest))
put(U/'OBSERVATION_DIAGNOSTIC.json',dict(scope='Previously exposed FIT source diagnostics only; no new score or label tuning',targets=targets,all_source_stats=stats))
panels=[]
for target in targets:
 key=target['target']['source_key'];lab=next(l for l in labels if l['key']==key.rsplit(':',1)[0])
 cap=cv2.VideoCapture(lab['source']);fps=cap.get(cv2.CAP_PROP_FPS);frames=lab['reviewed_unique_frames'];imlist=[]
 for f in frames:
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
  im=cv2.resize(im,(320,180));cv2.putText(im,f'{f} / {f/fps:.3f}s',(7,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA);imlist.append(im)
 cap.release()
 while len(imlist)%3:imlist.append(np.zeros_like(imlist[0]))
 grid=np.vstack([np.hstack(imlist[j:j+3]) for j in range(0,len(imlist),3)])
 head=np.full((50,960,3),245,np.uint8);cv2.putText(head,key,(12,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv2.LINE_AA)
 path=U/(key.replace(':','_')+'.jpg');cv2.imwrite(str(path),np.vstack([head,grid]))
 panels.append(dict(source_key=key,panel=str(path),source=lab['source'],source_sha256=hashlib.sha256(Path(lab['source']).read_bytes()).hexdigest(),label=lab.get('right'),label_note=lab.get('note'),native_frames=frames))
put(U/'DIAGNOSTIC_PANELS.json',panels)
for t in targets:
 r=t['target'];print(json.dumps({k:v for k,v in r.items() if k not in ['q','observed']},ensure_ascii=False))
print('panels',json.dumps(panels,ensure_ascii=False))
