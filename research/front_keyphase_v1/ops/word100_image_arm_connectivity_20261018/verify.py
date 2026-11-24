"""Independent numeric reference: scipy components, separate shortest-path BFS,
separate decision logic; no import from candidate.py or run.py.
"""
import json,hashlib,math,traceback,time
from pathlib import Path
from collections import deque,Counter
import numpy as np,cv2
from scipy import ndimage
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_uncertain_hand_preservation_20261018'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def distance(mask,start,end,limit):
    start=tuple(start);end=tuple(end);front={start};visited={start};h,w=mask.shape
    for step in range(limit+1):
        if end in front:return step
        new=set()
        for x,y in front:
            for a in range(max(0,x-1),min(w,x+2)):
                for b in range(max(0,y-1),min(h,y+2)):
                    if (a,b) not in visited and mask[b,a]:new.add((a,b))
        visited.update(new);front=new
        if not front:break
    return None
def main():
    c=read(U/'CONTEXT.json');rows={r['clip_id']:r for r in c['rows']};out=read(U/'RESULTS.json');counts=Counter();cache={};start=time.monotonic()
    for ri,x in enumerate(out):
        image=cv2.imread(x['image']);assert image is not None
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)[:,:,1:].astype(float);radius=max(1,int(round(x['scale']*.02)));h,w=image.shape[:2];centers=[]
        for xy in x['elbows']+x['wrists']:
            if xy is None:continue
            px,py=xy;vals=lab[max(0,py-radius):min(h,py+radius+1),max(0,px-radius):min(w,px+radius+1)].reshape(-1,2)
            vals=np.sort(vals,axis=0);nn=len(vals);centers.append((vals[(nn-1)//2]+vals[nn//2])/2)
        # If anchors unusable candidate intentionally did not collect any seed.
        if not all(q is not None for q in x['elbows']):centers=[]
        assert np.array_equal(np.asarray(centers).reshape(-1,2),np.asarray(x['centers_ab']).reshape(-1,2))
        expected=np.zeros((h,w),bool)
        for ab in centers:expected|=np.einsum('ijk,ijk->ij',lab-ab,lab-ab)<=400.
        saved=load(x['map']);assert np.array_equal(expected,saved['support']),('support',ri)
        cc,nc=ndimage.label(expected,structure=np.ones((3,3),int))
        # Component numeric labels may differ; their complete partitions must
        # agree, including background. Use unique pairs to prove one-to-one.
        pairs=np.unique(np.stack([cc.ravel(),saved['components'].ravel()],axis=1),axis=0)
        assert len(pairs)==len(np.unique(cc))==len(np.unique(saved['components'])),('components',ri)
        assert np.array_equal(cc==0,saved['components']==0)
        get=lambda p:0 if p is None else int(cc[p[1],p[0]])
        ec=[get(p) for p in x['elbows']];wc=[get(p) for p in x['wrists']];rgb=[]
        for i,pt in enumerate(x['wrists']):
            side=-1;comp=wc[i]
            possibilities=[j for j,e in enumerate(ec) if comp!=0 and e==comp]
            if pt is not None and len(possibilities)==1 and wc.count(comp)==1:
                j=possibilities[0];a=x['elbows'][j];lim=int(math.floor(1.5*max(abs(a[0]-pt[0]),abs(a[1]-pt[1]))));d=distance(expected,a,pt,lim)
                if d is not None:
                    side=j;path=x['paths'][i];assert len(path)-1==d and path[0]==a and path[-1]==pt
                    assert all(expected[q[1],q[0]] for q in path)
                    assert all(max(abs(p[0]-q[0]),abs(p[1]-q[1]))==1 for p,q in zip(path,path[1:]))
            rgb.append(side)
        assert rgb==x['rgb_side'],('rgbside',ri,rgb,x['rgb_side'])
        decision=x['old_sides'].copy()
        for i,s in enumerate(rgb):
            if x['old_identity'][i]>=0 or decision[i]<0 or s<0:continue
            if any(j!=i and old==s for j,old in enumerate(x['old_sides'])):continue
            decision[i]=s
        assert decision==x['sides'],('decision',ri)
        key=(x['clip_id'],x['view'])
        if key not in cache:
            cache={key:load(B/'results'/f'{key[0]}_{key[1]}.npz')}
        p=cache[key];f=x['index'];n=x['raw_count'];assert p['sides'][f,:n].tolist()==x['old_sides'];assert p['identity_side'][f,:n].tolist()==x['old_identity'];assert p['point_mask'][f,:n].sum(1).tolist()==x['observed_points']
        counts['frames']+=1;counts['hand_candidates']+=n;counts['observed_points_preserved']+=sum(x['observed_points']);counts['proposed_changes']+=sum(a!=b for a,b in zip(x['old_sides'],x['sides']))
        if (ri+1)%200==0:print('VERIFY',ri+1,'/',len(out),flush=True)
    for pins in (c['pins'],read(U/'OUTPUT_PINS.json')):
        for p,h in pins.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    summary=read(U/'SUMMARY.json');assert counts['proposed_changes']==summary['changed_events'];assert counts['frames']==summary['counts']['frames'];assert counts['hand_candidates']==summary['counts']['raw_candidates']
    ann=read(U/'ANNOTATED.json');normal=[x for x in ann if x['verdict']=='normal_identity'];rem=[x for x in ann if x['remaining9']];cross=read(U/'CROSSING.json')
    assert len(normal)==441 and len(rem)==9 and len(cross)==315
    assert sum(x['after']==x['actual_side'] for x in rem)==summary['remaining9_proposed_correct']
    write(U/'INDEPENDENT.json',{'status':'PASS','counts':dict(counts),'normal_preserved':sum(x['before']==x['after']==x['actual_side'] for x in normal),'crossing_preserved':sum(x['before']==x['after'] for x in cross),'remaining9_numeric_correct':sum(x['after']==x['actual_side'] for x in rem),'protected_pins':len(c['pins']),'elapsed_seconds':time.monotonic()-start,'independence':'Separate scipy connectivity and BFS numeric code by same Codex, not separate human visual labels'})
    print('PASS',dict(counts),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (U/'FIRST_VERIFY_ERROR.json').exists():write(U/'FIRST_VERIFY_ERROR.json',{'error':traceback.format_exc()})
        raise
