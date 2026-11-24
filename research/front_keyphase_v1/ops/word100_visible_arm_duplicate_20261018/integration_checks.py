"""Actual wrapper replay on changed/conflict clips, order/prefix/hidden input."""
import numpy as np
from run import U,read,write,load
from candidate import HandAssigner,geometry
def main():
    c=read(U/'CONTEXT.json');ids={x['clip_id'] for x in read(U/'ARM_CONFLICTS.json')};rows=[r for r in c['rows'] if r['clip_id'] in ids]
    nf=nh=prefix=0
    for r in rows:
        for v in ('masked','sensor'):
            z=load(r['views'][v]['raw']);saved=load(U/'results'/f"{r['clip_id']}_{v}.npz");m=HandAssigner();rev=HandAssigner();short=HandAssigner();N=len(z['raw_count'])
            for f in range(N):
                n=int(z['raw_count'][f]);h=z['hand_xy_px'][f,:n];la=z['raw_label'][f,:n];sc=z['raw_score'][f,:n];p=z['pose_xy_px'][f];pu=z['pose_usable'][f];tail=(p,pu,int(z['timestamp_us'][f]),int(z['width'][f]),int(z['height'][f]))
                out,d=m.assign(h,la,sc,*tail);out=[-1 if out[i] is None else out[i] for i in range(n)]
                assert out==saved['sides'][f,:n].tolist(),(r['clip_id'],v,f,'wrapper')
                rr,_=rev.assign(h[::-1],la[::-1],sc[::-1],*tail);rr=[-1 if rr[i] is None else rr[i] for i in reversed(range(n))]
                assert out==rr,(r['clip_id'],v,f,'permutation')
                if f<N//2:
                    ss,_=short.assign(h,la,sc,*tail);assert out==[-1 if ss[i] is None else ss[i] for i in range(n)];prefix+=1
                q=p.copy();q[~pu]=[1e6,-1e6]
                a=geometry(h,p,pu,tail[-2],tail[-1]);b=geometry(h,q,pu,tail[-2],tail[-1])
                for key in ('arm_cost','elbow_distance','distal_cost','coverage','preferred','arms_visible','valid'):assert np.array_equal(a[key],b[key]),(r['clip_id'],v,f,key)
                nf+=1;nh+=int(sum(~pu))
    write(U/'INTEGRATION_CHECKS.json',{'status':'PASS','clips':len(rows),'streams':len(rows)*2,'actual_wrapper_and_order_frames':nf,'prefix_frames':prefix,'hidden_unusable_pose_points_perturbed':nh,'training':0})
    print(nf,'wrapper/order,',prefix,'prefix,',nh,'hidden points PASS')
if __name__=='__main__':main()
