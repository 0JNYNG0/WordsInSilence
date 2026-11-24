"""Independent sequential decode verifies source, native frame and timestamp."""
from pathlib import Path
import json,hashlib,cv2,numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'MEDIA_VERIFIED.json').exists();m=read(U/'REVIEW_MANIFEST.json');plan=read(U/'FROZEN.json');count=0;errs=[]
    for p,h in plan['pins'].items():assert sha(p)==h
    for r,p in zip(m['rows'],plan['plans']):
        assert r['frames']==p['frames'] and r['person'] in range(1,10) and sha(r['source'])==r['source_sha256']
        items={z['frame']:z for z in r['images']};assert list(items)==r['frames'];cap=cv2.VideoCapture(r['source']);assert cap.isOpened()
        for f in range(max(items)+1):
            ok,bgr=cap.read();assert ok
            if f not in items:continue
            z=items[f];assert abs(float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000-z['seconds'])<1e-6
            assert sha(z['image'])==z['sha256'];im=cv2.imread(z['image']);assert im.shape==bgr.shape
            error=float(np.abs(im.astype(np.float32)-bgr.astype(np.float32)).mean());assert error<5,(r['alias'],f,error)
            errs.append(error);count+=1
        cap.release()
        for s in r['sheets']:assert sha(s['path'])==s['sha256']
        assert [f for s in r['sheets'] for f in s['frames']]==r['frames']
    assert len(m['rows'])==18 and count==m['native_images']==248
    out=dict(status='PUBLIC_NATIVE_REVIEW_MEDIA_VERIFIED',windows=18,frames=248,max_jpeg_mae=float(max(errs)),
        native_time_and_every_selected_frame=True,source_read_only=True,model_inference=0,scope='Source/frame/hash/time integrity, not validation of physical annotation or KSL correctness.',whole_goal_complete=False)
    with (U/'MEDIA_VERIFIED.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
