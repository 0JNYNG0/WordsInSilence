from prepare import *
from geometry import features
def main():
    rows=[r for r in read(U/'INPUT_PLAN.json')['rows'] if r['role']=='calibration'];out=[]
    for r in rows:
        p=U/'crops'/(r['clip_id']+'.npz')
        if not p.exists():continue
        new=np.load(p);old=np.load(r['raw']);l,t,rr,b=r['crop']
        for view,z in [('full',old),('crop',new)]:
            fs=features(z['xy_px'],z['score'])
            for side,start in enumerate([91,112]):
                orig=old['xy_px'][:,start:start+21]
                outside=np.count_nonzero((orig[:,:,0]<l)|(orig[:,:,0]>=rr)|(orig[:,:,1]<t)|(orig[:,:,1]>=b),axis=1)>=18
                lo,hi=r['intervals'][0];main=(z['timestamp_us']>=lo*1e6)&(z['timestamp_us']<=hi*1e6)
                for group,m in [('main_projected_inside',main&~outside),('projected_outside',outside)]:
                    if view=='full' and group=='projected_outside':continue
                    n=int(m.sum())
                    if not n:continue
                    stats={k:np.nanpercentile(v[m],[10,50,90,99]).round(3).tolist() for k,v in fs[side].items() if k!='palm_spokes_arm'}
                    out.append(dict(clip=r['clip_id'],view=view,side=side,group=group,n=n,stats=stats))
    write(U/'CALIBRATION_GEOMETRY.json',dict(rows=out,caveat='Projection-based grouping is a diagnostic proxy, not human visibility ground truth. No classifier scores used.'))
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
