"""Separate scalar verifier, never imports candidate or runs detector/classifier."""
from run import U,P,G,V,read,write,load,check
from collections import Counter
import math,time
import numpy as np
def main():
    c=read(U/'CONTEXT.json');check(c['pins']);check(read(U/'OUTPUT_PINS.json'));counts=Counter();start=time.monotonic()
    for rn,r in enumerate(c['rows'],1):
        for v in ('masked','sensor'):
            z=load(r['views'][v]['raw']);old=load(P/'results'/f"{r['clip_id']}_{v}.npz");g=load(G/'results'/f"{r['clip_id']}_{v}.npz");pr=load(V/'results'/f"{r['clip_id']}_{v}.npz");out=load(U/'results'/f"{r['clip_id']}_{v}.npz");last_mask=None
            assert np.array_equal(out['frame_id'],z['frame_id']) and np.array_equal(out['timestamp_us'],z['timestamp_us'])
            for f,n0 in enumerate(z['raw_count']):
                n=int(n0);us=int(z['timestamp_us'][f]);h=z['hand_xy_px'][f,:n];p=z['pose_xy_px'][f];pu=z['pose_usable'][f];w=int(z['width'][f]);ht=int(z['height'][f]);key=(r['clip_id'],v,f)
                mask=np.zeros((n,21),bool)
                for i in range(n):
                    for k in range(21):
                        x,y=map(float,h[i,k]);score=float(z['raw_score'][f,i]);mask[i,k]=math.isfinite(x) and math.isfinite(y) and 0<=x<w and 0<=y<ht and math.isfinite(score) and score>=.5
                expected=old['sides'][f,:n].tolist();blocked=[False]*n;anchor=[-1]*n
                if n==2:
                    for i in (0,1):
                        j=1-i;s=expected[i];other=expected[j]
                        if s<0 or other!=1-s or pr['state'][f,i]!=1 or pr['state'][f,j]!=2:continue
                        root=int(pr['origin_frame'][f,j]);recent=root>=0 and 0<=us-int(z['timestamp_us'][root])<=100000
                        if not (bool(pr['current_pose'][f,j]) or recent):continue
                        if not all(g['strict_hand_inside'][f]) or g['coverage'][f,i,j]<.8:continue
                        own=float(pr['confirmed_track_distance'][f,i,s]);opp=float(pr['confirmed_track_distance'][f,i,other])
                        if not math.isfinite(own) and math.isfinite(opp) and opp<=.4:expected[i]=-1;blocked[i]=True;anchor[i]=j
                assert expected==out['sides'][f,:n].tolist(),key
                assert blocked==out['blocked'][f,:n].tolist() and anchor==out['anchor'][f,:n].tolist(),key
                assert np.array_equal(mask,out['point_mask'][f,:n]),key
                expected_xy=np.full((n,21,2),np.nan);expected_local=expected_xy.copy();localmask=np.zeros_like(mask)
                for i in range(n):
                    for k in range(21):
                        if mask[i,k]:
                            expected_xy[i,k]=h[i,k]
                            if mask[i,0]:
                                expected_local[i,k]=[float(h[i,k,0])-float(h[i,0,0]),float(h[i,k,1])-float(h[i,0,1])];localmask[i,k]=True
                for actual,k in ((expected_xy,'observed_xy'),(expected_local,'local_xy'),(localmask,'local_mask')):assert np.array_equal(actual,out[k][f,:n],equal_nan=True),(key,k)
                ids=[-1]*n;states=[0]*n
                for i,s in enumerate(expected):
                    if s>=0 and (pr['state'][f,i]==2 or old['action'][f,i] in (2,3,4)):ids[i]=s;states[i]=1
                    if blocked[i]:states[i]=2
                assert ids==out['identity_side'][f,:n].tolist() and states==out['identity_state'][f,:n].tolist(),key
                assert np.array_equal(out['suggested_side'][f,:n],old['sides'][f,:n]) and np.array_equal(out['effective_side'][f,:n],out['sides'][f,:n]),key
                wrist=[bool(pu[k]) and all(math.isfinite(float(q)) for q in p[k]) and 0<=p[k,0]<w and 0<=p[k,1]<ht for k in (15,16)];assert wrist==out['pose_wrist_usable'][f].tolist(),key
                shgood=all(pu[k] and np.isfinite(p[k]).all() and 0<=p[k,0]<w and 0<=p[k,1]<ht for k in (11,12));scale=math.hypot(float(p[11,0])-float(p[12,0]),float(p[11,1])-float(p[12,1])) if shgood else 0.
                links=np.zeros((n,2),bool);cost=np.full((n,2),np.nan);common=np.zeros((n,2),np.int8)
                if f and 0<us-int(z['timestamp_us'][f-1])<=100000 and scale>1e-6:
                    for i in range(n):
                        for j in range(int(z['raw_count'][f-1])):
                            indices=[k for k in (0,5,9,13,17) if mask[i,k] and last_mask[j,k]];common[i,j]=len(indices)
                            if len(indices)>=3:
                                cost[i,j]=sum(math.hypot(float(h[i,k,0])-float(z['hand_xy_px'][f-1,j,k,0]),float(h[i,k,1])-float(z['hand_xy_px'][f-1,j,k,1])) for k in indices)/len(indices)/scale;links[i,j]=cost[i,j]<=.4
                assert np.array_equal(links,out['possible_links'][f,:n]) and np.array_equal(common,out['common_palm_points'][f,:n]),key
                assert np.allclose(cost,out['link_cost'][f,:n],rtol=8*np.finfo(np.float32).eps,atol=8*np.finfo(np.float32).eps,equal_nan=True),key
                last_mask=mask;counts['frames']+=1;counts['raw_candidates']+=n;counts['usable_hand_points']+=int(mask.sum());counts['usable_local_points']+=int(localmask.sum());counts['new_duplicate_withheld']+=sum(blocked);counts['possible_link_edges']+=int(links.sum())
        if rn%200==0:print('VERIFY',rn,'/899',flush=True)
    summary=read(U/'SUMMARY.json')
    for k,v in counts.items():assert summary['counts'].get(k,0)==v,(k,v)
    check(c['pins']);check(read(U/'OUTPUT_PINS.json'));write(U/'INDEPENDENT.json',{'status':'PASS','counts':dict(counts),'float_cost_tolerance':float(8*np.finfo(np.float32).eps),'boolean_link_decisions_exact':True,'recovery':'FIRST_VALIDATION_FAILURE.json; original candidate/replay/results untouched','protected_pins':len(c['pins']),'output_pins':len(read(U/'OUTPUT_PINS.json')),'seconds':time.monotonic()-start,'method':'Independent scalar per-point geometry/quality masks, local coordinates, possible links, identity certainty and duplicate guard. Prior geometry independently verified before and reused by hash. Same operator; not independent human labels.'});print('PASS',dict(counts),flush=True)
if __name__=='__main__':main()
