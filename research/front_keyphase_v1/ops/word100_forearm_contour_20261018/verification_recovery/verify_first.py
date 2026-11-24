"""Independent scalar contour extraction and decision verification.

Shares OpenCV grayscale conversion only; filters use scipy separable kernels,
bilinear sampling and peak/pair logic are independently expressed. Does not
import candidate.py or alter any outputs. Numerical agreement is not efficacy.
"""
import json,gzip,hashlib,time,traceback,math,itertools
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import correlate1d
U=Path(__file__).resolve().parent;P=U.parent/'word100_image_arm_connectivity_20261018'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def bilinear(a,x,y):
    iy=max(0,min(a.shape[0]-2,math.floor(y)));ix=max(0,min(a.shape[1]-2,math.floor(x)))
    return sum(float(a[iy+dy,ix+dx])*(1-abs((x-ix)-dx))*(1-abs((y-iy)-dy)) for dy in [0,1] for dx in [0,1])

def reference(gx,gy,threshold,a,b,scale,p):
    if a is None or b is None or scale<=0:return False,[]
    dx=b[0]-a[0];dy=b[1]-a[1];length=math.hypot(dx,dy)
    if length/scale<p['min_length'] or length/scale>p['max_length']:return False,[]
    nx=-dy/length;ny=dx/length;extent=math.ceil(p['search_halfwidth']*scale);offsets=list(range(-extent,extent+1));rows=[]
    for t in np.linspace(p['start'],p['end'],p['slices']):
        cx=a[0]+t*dx;cy=a[1]+t*dy;profile=[]
        for d in offsets:
            x=cx+d*nx;y=cy+d*ny;xx=bilinear(gx,x,y);yy=bilinear(gy,x,y)
            profile.append([xx*nx+yy*ny,math.hypot(xx,yy),float(2<=x<gx.shape[1]-3 and 2<=y<gx.shape[0]-3)])
        grad=[abs(z[0]) for z in profile];peaks=[]
        for k in range(2,len(offsets)-2):
            if profile[k][2] and grad[k]>=threshold and grad[k]>=p['normal_alignment']*profile[k][1] and all(grad[k]>grad[q] for q in [k-2,k-1]) and all(grad[k]>=grad[q] for q in [k+1,k+2]):peaks.append(k)
        options=[]
        for i,j in itertools.combinations(peaks,2):
            lo,hi=offsets[i],offsets[j];width=hi-lo;center=(lo+hi)/2
            if lo<0<hi and profile[i][0]*profile[j][0]<0 and p['min_width']*scale<=width<=p['max_width']*scale and abs(center)<=p['center_tolerance']*scale:
                options.append((min(grad[i],grad[j]),-abs(center),-width,-i,-j,lo,hi))
        selected=max(options) if options else None
        pair=None if selected is None else [float(selected[-2]),float(selected[-1]),selected[0]]
        rows.append({'profile':profile,'pair':pair})
    valid=[(i,x['pair']) for i,x in enumerate(rows) if x['pair'] is not None]
    zero_runs=[sum(1 for _ in group) for value,group in itertools.groupby(x['pair'] is None for x in rows) if value]
    widths=[v[1]-v[0] for _,v in valid]
    cv=float(np.std(widths)/np.mean(widths)) if widths else float('inf')
    jumps=[max(abs(a[0]-b[0]),abs(a[1]-b[1]))/((j-i)*scale) for (i,a),(j,b) in zip(valid,valid[1:])]
    ok=len(valid)>=p['min_supported_slices'] and max(zero_runs,default=0)<=p['max_missing_run'] and cv<=p['max_width_cv'] and max(jumps,default=0)<=p['max_adjacent_edge_jump']
    return bool(ok),rows

def main():
    start=time.monotonic();c=read(U/'CONTEXT.json');p=c['parameters'];out=read(U/'RESULTS.json');old=read(P/'RESULTS.json');counts={'frames':0,'hypotheses':0,'profiles':0,'hands':0,'points':0,'changes':0}
    for k,(x,before) in enumerate(zip(out,old)):
        assert x['old_sides']==before['old_sides'] and x['old_identity']==before['old_identity'] and x['observed_points']==before['observed_points']
        im=cv2.imread(x['image']);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float)
        smooth=correlate1d(correlate1d(gray,[.25,.5,.25],0,mode='mirror'),[.25,.5,.25],1,mode='mirror')
        gx=correlate1d(correlate1d(smooth,[.25,.5,.25],0,mode='mirror'),[-.5,0,.5],1,mode='mirror')
        gy=correlate1d(correlate1d(smooth,[.25,.5,.25],1,mode='mirror'),[-.5,0,.5],0,mode='mirror')
        threshold=max(p['min_edge_gradient'],p['contrast_gradient_ratio']*float(np.percentile(gray,95)-np.percentile(gray,5)));assert threshold==x['threshold']
        with gzip.open(x['trials_file'],'rt',encoding='utf8') as f:trials=json.load(f)
        evidence=[]
        for i,hand in enumerate(trials):
            passed=[]
            for side,t in enumerate(hand):
                expected,rows=reference(gx,gy,threshold,x['elbows'][side],x['wrists'][i],x['scale'],p)
                assert expected==t['pass'],('tube',k,i,side,expected,t['pass'])
                assert len(rows)==len(t['slices'])
                for a,b in zip(rows,t['slices']):
                    # Different evaluation order may differ at floating roundoff.
                    assert np.allclose(a['profile'],b['profile'],rtol=0,atol=1e-10),('profile',k,i,side)
                    assert (a['pair'] is None)==(b['pair'] is None),('pair_none',k,i,side)
                    if a['pair'] is not None:assert np.allclose(a['pair'],b['pair'],rtol=0,atol=1e-10),('pair',k,i,side)
                    counts['profiles']+=1
                if expected:passed.append(side)
                counts['hypotheses']+=1
            evidence.append(passed[0] if len(passed)==1 else -1)
        assert evidence==x['evidence_side']
        decision=x['old_sides'].copy()
        for i,side in enumerate(evidence):
            if x['old_identity'][i]>=0 or decision[i]<0 or side<0:continue
            if sum(z==side for z in evidence)>1:continue
            if any(j!=i and z==side for j,z in enumerate(x['old_sides'])):continue
            decision[i]=side
        assert decision==x['sides'],('decision',k)
        counts['frames']+=1;counts['hands']+=len(evidence);counts['points']+=sum(x['observed_points']);counts['changes']+=sum(a!=b for a,b in zip(decision,x['old_sides']))
        if (k+1)%100==0:print('VERIFY',k+1,flush=True)
    for pinset in [c['pins'],read(U/'OUTPUT_PINS.json')]:
        for path,h in pinset.items():assert sha(path)==h,path
    ann=read(U/'ANNOTATED.json');cross=read(U/'CROSSING.json');s=read(U/'SUMMARY.json')
    normal=[a for a in ann if a['verdict']=='normal_identity'];rem=[a for a in ann if a['remaining9']]
    assert len(out)==785 and len(normal)==441 and len(rem)==9 and len(cross)==315
    assert counts['changes']==s['changed_events']
    assert sum(a['after']==a['actual_side'] for a in rem)==s['remaining9_correct']
    assert sum(a['after']!=a['actual_side'] for a in normal)==s['normal_regressions']
    assert sum(a['before']!=a['after'] for a in cross)==s['crossing_changed_frames']
    write(U/'INDEPENDENT.json',{'status':'PASS','counts':counts,'protected_pins':len(c['pins']),'elapsed_seconds':time.monotonic()-start,'implementation':'Separate scipy filters + scalar sampling/pair/decision code by same Codex. Shared cv2 grayscale conversion. Not independent human anatomical judgment.'})
    print('PASS',counts,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (U/'FIRST_VERIFY_ERROR.json').exists():write(U/'FIRST_VERIFY_ERROR.json',{'error':traceback.format_exc()})
        raise
