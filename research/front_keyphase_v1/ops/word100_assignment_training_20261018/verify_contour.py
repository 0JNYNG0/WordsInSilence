"""Independent current-FOV contour arithmetic for all newly integrated frames."""
from common import *
import cv2,time,traceback,argparse
ref=module('_independent_forearm_reference',F/'verify.py')

def main(role):
 c=read(U/'PREP_CONTEXT.json');p=c['parameters']['contour'];count=0;hypotheses=0;start=time.monotonic()
 for r in c['rows']:
  if r['role']!=role:continue
  rec=read(U/'records'/f"{r['clip_id']}.json")
  if not rec['source_video']:continue
  assert sha(rec['source_video'])==r['source']['sha256']
  cap=cv2.VideoCapture(rec['source_video'])
  for view in ['masked','sensor']:
   z=load(r['views'][view]['raw']);b=load(U/'observations'/f"{r['clip_id']}_{view}.npz")
   trace=read(U/'observations'/f"{r['clip_id']}_{view}_contour.json")
   for x in trace:
    t=x['index'];actual=x['contour'];n=int(z['raw_count'][t]);width=int(z['width'][t]);height=int(z['height'][t])
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(z['frame_id'][t]));ok,full=cap.read();assert ok
    w=min(width,640);h=round(height*w/width);im=cv2.resize(full[:height,:width],(w,h),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float)
    smooth=ref.correlate1d(ref.correlate1d(gray,[.25,.5,.25],0,mode='mirror'),[.25,.5,.25],1,mode='mirror')
    gx=ref.correlate1d(ref.correlate1d(smooth,[.25,.5,.25],0,mode='mirror'),[-.5,0,.5],1,mode='mirror')
    gy=ref.correlate1d(ref.correlate1d(smooth,[.25,.5,.25],1,mode='mirror'),[-.5,0,.5],0,mode='mirror')
    threshold=max(p['min_edge_gradient'],p['contrast_gradient_ratio']*float(np.percentile(gray,95)-np.percentile(gray,5)))
    def pix(q):return (min(w-1,round(float(q[0])*w/width)),min(h-1,round(float(q[1])*h/height)))
    elbows=[pix(z['pose_xy_px'][t,k]) for k in [13,14]];scale=float(np.linalg.norm(z['pose_xy_px'][t,11]-z['pose_xy_px'][t,12]))*w/width
    evidence=[];supported=[]
    for i in range(n):
     wrist=pix(z['hand_xy_px'][t,i,0]) if b['point_mask'][t,i,0] else None
     passed=[];counts=[]
     for side,elbow in enumerate(elbows):
      yes,rows=ref.reference(gx,gy,threshold,elbow,wrist,scale,p)
      if yes:passed.append(side)
      counts.append(sum(q['pair'] is not None for q in rows));hypotheses+=1
     evidence.append(passed[0] if len(passed)==1 else -1);supported.append(counts)
    before=b['base_sides'][t,:n].tolist();expected=before.copy()
    for i,s in enumerate(evidence):
     if b['identity_side'][t,i]>=0 or before[i]<0 or s<0:continue
     if sum(k==s for k in evidence)>1 or any(j!=i and k==s for j,k in enumerate(before)):continue
     expected[i]=s
    assert supported==actual['supported'],(r['clip_id'],view,t,'supported')
    assert evidence==actual['evidence_side'] and expected==actual['sides'],(r['clip_id'],view,t,'decision')
    count+=1
    if count%200==0:print('CONTOUR_VERIFY',count,flush=True)
  cap.release()
 write(U/f'{role}_CONTOUR_INDEPENDENT.json',{'status':'PASS','frames':count,'hypotheses':hypotheses,'seconds':time.monotonic()-start,'code_sha256':sha(__file__),'reference_sha256':sha(F/'verify.py'),'method':'Separate scalar sampling/contour pair decisions plus scipy filters; shared decoding/resize; same Codex, not independent human ground truth.'},new=True)
 print('PASS_CONTOUR',count,hypotheses,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--role',default='train_candidate');a=p.parse_args()
 try:main(a.role)
 except BaseException:
  path=U/('FIRST_CONTOUR_VERIFY_ERROR_'+a.role+'.json')
  if not path.exists():write(path,{'at':now(),'error':traceback.format_exc()},new=True)
  raise
