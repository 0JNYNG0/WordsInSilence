"""Research-only paired forearm contour evidence; accepts cropped pixels only.

No label/word/person features, future frames, hidden pixels, or learned weights.
Two possible elbow-to-hand paths are measured symmetrically. A visible tube is
supporting evidence, NOT an anatomical guarantee. Original points are retained.
"""
import cv2
import numpy as np

PARAMS=dict(slices=12,start=.15,end=.85,search_halfwidth=.30,
 min_width=.06,max_width=.40,center_tolerance=.08,
 min_edge_gradient=3.,contrast_gradient_ratio=.04,normal_alignment=.70,
 min_supported_slices=10,max_missing_run=1,max_width_cv=.35,
 max_adjacent_edge_jump=.08,min_length=.15,max_length=2.5)

def fields(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image.copy()
    g=cv2.GaussianBlur(gray.astype(np.float64),(3,3),0,borderType=cv2.BORDER_REFLECT_101)
    gx=cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3,scale=1/8,borderType=cv2.BORDER_REFLECT_101)
    gy=cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3,scale=1/8,borderType=cv2.BORDER_REFLECT_101)
    threshold=max(PARAMS['min_edge_gradient'],PARAMS['contrast_gradient_ratio']*float(np.percentile(gray,95)-np.percentile(gray,5)))
    return gray,gx,gy,threshold

def sample(a,xy):
    h,w=a.shape;x=xy[:,0];y=xy[:,1];valid=(x>=2)&(x<w-3)&(y>=2)&(y<h-3)
    ix=np.clip(np.floor(x).astype(int),0,w-2);iy=np.clip(np.floor(y).astype(int),0,h-2)
    dx=x-ix;dy=y-iy
    value=(a[iy,ix]*(1-dx)*(1-dy)+a[iy,ix+1]*dx*(1-dy)+a[iy+1,ix]*(1-dx)*dy+a[iy+1,ix+1]*dx*dy)
    return value,valid

def pair(profile,offsets,scale,threshold):
    q=np.asarray(profile);n=len(q);strength=np.abs(q[:,0]);ok=q[:,2].astype(bool)
    peaks=[]
    for i in range(2,n-2):
        if not ok[i] or strength[i]<threshold or strength[i]<PARAMS['normal_alignment']*q[i,1]:continue
        # Fixed local-maximum definition, leftmost peak for an equal plateau.
        if strength[i]>max(strength[i-2:i]) and strength[i]>=max(strength[i+1:i+3]):peaks.append(i)
    pairs=[]
    for i in peaks:
        for j in peaks:
            if i>=j or offsets[i]>=0 or offsets[j]<=0 or q[i,0]*q[j,0]>=0:continue
            width=offsets[j]-offsets[i];mid=(offsets[i]+offsets[j])/2
            if PARAMS['min_width']*scale<=width<=PARAMS['max_width']*scale and abs(mid)<=PARAMS['center_tolerance']*scale:
                pairs.append((min(strength[i],strength[j]),-abs(mid),-width,-i,-j,i,j))
    if not pairs:return None
    p=max(pairs);return [float(offsets[p[-2]]),float(offsets[p[-1]]),float(p[0])]

def tube(gx,gy,threshold,elbow,wrist,scale):
    if elbow is None or wrist is None or scale<=0:return {'pass':False,'reason':'endpoint_unavailable','slices':[]}
    a=np.asarray(elbow,float);b=np.asarray(wrist,float);v=b-a;length=float(np.linalg.norm(v));ratio=length/scale
    if not PARAMS['min_length']<=ratio<=PARAMS['max_length']:return {'pass':False,'reason':'length_outside_contract','slices':[],'length':length}
    direction=v/length;normal=np.array([-direction[1],direction[0]])
    offsets=np.arange(-int(np.ceil(PARAMS['search_halfwidth']*scale)),int(np.ceil(PARAMS['search_halfwidth']*scale))+1,dtype=float)
    slices=[]
    for t in np.linspace(PARAMS['start'],PARAMS['end'],PARAMS['slices']):
        center=a+t*v;xy=center+offsets[:,None]*normal
        xx,valid=sample(gx,xy);yy,_=sample(gy,xy)
        profile=np.stack((xx*normal[0]+yy*normal[1],np.hypot(xx,yy),valid.astype(float)),axis=1)
        p=pair(profile,offsets,scale,threshold)
        slices.append({'center':center.tolist(),'normal':normal.tolist(),'pair':p,'profile':profile.tolist()})
    selected=[(i,x['pair']) for i,x in enumerate(slices) if x['pair'] is not None]
    gap=0;maxgap=0
    for x in slices:
        gap=0 if x['pair'] else gap+1;maxgap=max(maxgap,gap)
    widths=np.array([p[1]-p[0] for _,p in selected]);cv=float(widths.std()/widths.mean()) if len(widths) else None
    jumps=[max(abs(p[0]-q[0]),abs(p[1]-q[1]))/((j-i)*scale) for (i,p),(j,q) in zip(selected,selected[1:])]
    jump=max(jumps,default=0.)
    passed=len(selected)>=PARAMS['min_supported_slices'] and maxgap<=PARAMS['max_missing_run'] and cv is not None and cv<=PARAMS['max_width_cv'] and jump<=PARAMS['max_adjacent_edge_jump']
    return {'pass':bool(passed),'reason':'paired_contour' if passed else 'incomplete_or_incoherent_contour','length':length,'offsets':offsets.tolist(),'slices':slices,'supported':len(selected),'max_missing_run':maxgap,'width_cv':cv,'max_edge_jump':jump}

def decide(tests,sides,identity):
    evidence=[]
    for h in tests:
        supported=[s for s,t in enumerate(h) if t['pass']]
        evidence.append(supported[0] if len(supported)==1 else -1)
    out=list(sides);why=[]
    for i,s in enumerate(evidence):
        if identity[i]>=0:why.append('keep_confirmed');continue
        if sides[i]<0:why.append('keep_withheld');continue
        if s<0:why.append('no_unique_contour');continue
        if evidence.count(s)>1:why.append('multiple_candidates_for_one_arm');continue
        if any(j!=i and old==s for j,old in enumerate(sides)):why.append('destination_occupied');continue
        out[i]=s;why.append('same_hint' if s==sides[i] else 'changed_hint')
    return out,evidence,why

def propose(image,elbows,wrists,scale,sides,identity):
    gray,gx,gy,threshold=fields(image)
    tests=[[tube(gx,gy,threshold,e,w,scale) for e in elbows] for w in wrists]
    out,evidence,why=decide(tests,sides,identity)
    return {'sides':out,'evidence_side':evidence,'reason':why,'trials':tests,'threshold':threshold}
