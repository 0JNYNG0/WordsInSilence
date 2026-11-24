"""Real raw stream API versus saved-state replay, without model inference."""
from run import U,P,read,write,load
from candidate import HandAssigner,prior
import numpy as np
def main():
    c=read(U/'CONTEXT.json');ids={x['clip_id'] for x in read(U/'DELTA_PREVIOUS.json')} | {x['clip_id'] for x in read(P/'ARM_CONFLICTS.json')};rows=[r for r in c['rows'] if r['clip_id'] in ids]
    nf=nprefix=nh=0
    for r in rows:
        for v in ('masked','sensor'):
            z=load(r['views'][v]['raw']);saved=load(U/'results'/f"{r['clip_id']}_{v}.npz");m=HandAssigner();rev=HandAssigner();short=HandAssigner();N=len(z['raw_count'])
            changed=[x['index'] for x in read(U/'CHANGES.json') if x['clip_id']==r['clip_id'] and x['view']==v]
            stop=min(N,max(N//2,max(changed,default=0)+1))
            for f in range(N):
                n=int(z['raw_count'][f]);h=z['hand_xy_px'][f,:n];la=z['raw_label'][f,:n];sc=z['raw_score'][f,:n];p=z['pose_xy_px'][f];pu=z['pose_usable'][f];tail=(p,pu,int(z['timestamp_us'][f]),int(z['width'][f]),int(z['height'][f]))
                for x in (h,la,sc,p,pu):x.flags.writeable=False
                out,d=m.assign(h,la,sc,*tail);out=[-1 if out[i] is None else out[i] for i in range(n)]
                assert out==saved['sides'][f,:n].tolist(),(r['clip_id'],v,f,'actual API')
                rr,_=rev.assign(h[::-1],la[::-1],sc[::-1],*tail);assert out==[-1 if rr[i] is None else rr[i] for i in reversed(range(n))],(r['clip_id'],v,f,'order')
                if f<stop:
                    ss,_=short.assign(h,la,sc,*tail);assert out==[-1 if ss[i] is None else ss[i] for i in range(n)];nprefix+=1
                q=p.copy();q[~pu]=[1e6,-1e6]
                ga=prior.geometry(h,p,pu,tail[-2],tail[-1]);gb=prior.geometry(h,q,pu,tail[-2],tail[-1])
                for key in ('arm_cost','elbow_distance','distal_cost','coverage','preferred','arms_visible','valid'):assert np.array_equal(ga[key],gb[key]),(r['clip_id'],v,f,key)
                nf+=1;nh+=int(sum(~pu))
    write(U/'INTEGRATION_CHECKS.json',{'status':'PASS','public_clips':len(rows),'streams':2*len(rows),'api_and_order_frames':nf,'prefix_frames_including_all_new_changes':nprefix,'hidden_unusable_pose_points_perturbed':nh,'inputs_readonly':True,'training':0})
    print(nf,'API/order,',nprefix,'prefix,',nh,'hidden points PASS')
if __name__=='__main__':main()
