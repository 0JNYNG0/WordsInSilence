"""Independent scalar evidence ledger; no import or execution of candidate."""
import math,time
from collections import Counter
import numpy as np
from run import U,P,V,A,read,write,load,pinscheck
PALM=[0,5,9,13,17]
def dist(a,b):return math.hypot(float(a[0])-float(b[0]),float(a[1])-float(b[1]))
def mean_dist(hand,old,scale):return sum(dist(hand[k],old[j]) for j,k in enumerate(PALM))/5/scale if scale>1e-6 else math.inf
def unique(h,i,q,scale):
    d=mean_dist(h[i],q['points'],scale)
    return d<=.4 and all(d<.7*mean_dist(h[j],q['points'],scale) for j in range(len(h)) if j!=i)
def main():
    c=read(U/'CONTEXT.json');pinscheck(c['pins']);pinscheck(read(U/'OUTPUT_PINS.json'));counts=Counter();start=time.monotonic()
    for rn,r in enumerate(c['rows'],1):
        for view in ('masked','sensor'):
            raw=load(r['views'][view]['raw']);old=load(P/'results'/f"{r['clip_id']}_{view}.npz");pr=load(V/'results'/f"{r['clip_id']}_{view}.npz");out=load(U/'results'/f"{r['clip_id']}_{view}.npz")
            pending={};accepted={};last=None
            assert np.array_equal(raw['frame_id'],out['frame_id']) and np.array_equal(raw['timestamp_us'],out['timestamp_us'])
            for f,n0 in enumerate(raw['raw_count']):
                n=int(n0);us=int(raw['timestamp_us'][f]);hands=raw['hand_xy_px'][f,:n];pose=raw['pose_xy_px'][f];pu=raw['pose_usable'][f];w=int(raw['width'][f]);ht=int(raw['height'][f]);score=raw['raw_score'][f,:n]
                assert last is None or us>last;last=us
                scale=dist(pose[11],pose[12]) if all(pu[k] and all(math.isfinite(float(x)) for x in pose[k]) and 0<=pose[k,0]<w and 0<=pose[k,1]<ht for k in (11,12)) else 0.
                accepted={s:q for s,q in accepted.items() if 0<=us-q['seen_arm']<=100000}
                previous_pending=pending;pending={};sides=list(map(int,old['sides'][f,:n]));states=list(map(int,pr['state'][f,:n]));target=[-1]*n;action=[0]*n;votes=[0]*n;roots=[-1]*n;ex=sides.copy();proposal={}
                cost=old['arm_cost'][f,:n].copy();cost[~np.isfinite(cost)]=math.inf
                preference=list(map(int,old['preference'][f,:n]));valid=old['strict_hand_inside'][f,:n].tolist()
                for i,s in enumerate(sides):
                    want=preference[i]
                    possible=s>=0 and states[i]==1 and valid[i] and math.isfinite(float(score[i])) and score[i]>=.5 and want>=0 and all(old['arms_visible'][f])
                    if possible and all(float(cost[i,want])<.7*float(cost[j,want]) for j in range(n) if j!=i) and not any(j!=i and sides[j]==want and states[j]==2 for j in range(n)):target[i]=want
                for i,s in enumerate(sides):
                    if s<0 or states[i]!=1:continue
                    want=target[i]
                    if want>=0 and want!=s:
                        q=accepted.get(want)
                        if q is not None and unique(hands,i,q,scale):proposal[i]=want;action[i]=3;roots[i]=q['origin'];votes[i]=2
                        else:
                            q=previous_pending.get(want)
                            linked=q is not None and q['old']==s and 0<us-q['time']<=100000 and unique(hands,i,q,scale)
                            votes[i]=q['count']+1 if linked else 1;roots[i]=q['origin'] if linked else f
                            if votes[i]>=2:proposal[i]=want;action[i]=2
                            else:
                                ex[i]=-1;action[i]=1;pending[want]={'points':[tuple(map(float,hands[i,k])) for k in PALM],'time':us,'old':s,'count':votes[i],'origin':roots[i]}
                    elif valid[i] and math.isfinite(float(score[i])) and score[i]>=.5:
                        possible=[s0 for s0,q in accepted.items() if s0!=s and unique(hands,i,q,scale)]
                        if len(possible)==1 and preference[i] in (-1,possible[0]):
                            want=possible[0]
                            if not any(j!=i and sides[j]==want and states[j]==2 for j in range(n)):proposal[i]=want;action[i]=4;roots[i]=accepted[want]['origin'];votes[i]=2
                destinations=ex.copy()
                for i,s in proposal.items():destinations[i]=s
                success={}
                for i,s in proposal.items():
                    if destinations.count(s)>1:ex[i]=-1;action[i]=5
                    else:ex[i]=s;success[i]=s
                for i,s in success.items():
                    arm_us=us if target[i]==s else accepted[s]['seen_arm']
                    accepted[s]={'points':[tuple(map(float,hands[i,k])) for k in PALM],'time':us,'seen_arm':arm_us,'origin':roots[i]}
                for i,s in enumerate(sides):
                    if s>=0 and states[i]==2:accepted.pop(s,None);pending.pop(s,None)
                key=(r['clip_id'],view,f)
                for calculated,k in [(ex,'sides'),(action,'action'),(target,'target'),(votes,'votes'),(roots,'origin')]:assert calculated==out[k][f,:n].tolist(),(key,k,calculated,out[k][f,:n].tolist())
                assert len([s for s in ex if s>=0])==len(set(s for s in ex if s>=0)),key
                counts['frames']+=1;counts['raw_candidates']+=n;counts['changed']+=sum(a!=b for a,b in zip(ex,sides))
                counts['reassigned']+=sum(a!=b and a>=0 for a,b in zip(ex,sides));counts['new_withheld']+=sum(a!=b and a<0 for a,b in zip(ex,sides))
        if rn%200==0:print('VERIFY',rn,'/899',flush=True)
    summary=read(U/'SUMMARY.json');assert counts['changed']==len(read(U/'CHANGES.json'))
    assert counts['new_withheld']==summary['counts'].get('new_withheld',0) and counts['reassigned']==summary['counts'].get('reassigned',0)
    pinscheck(c['pins']);pinscheck(read(U/'OUTPUT_PINS.json'))
    write(U/'INDEPENDENT.json',{'status':'PASS','counts':dict(counts),'protected_pins':len(c['pins']),'output_pins':len(read(U/'OUTPUT_PINS.json')),'seconds':time.monotonic()-start,'method':'Separate scalar ledger, integer microsecond expiry, raw palm distances, full decision/action/vote/origin reconstruction. Prior geometry independent proof reused by pinned hashes. Same operator, not human ground truth.'})
    print(dict(counts),'PASS',flush=True)
if __name__=='__main__':main()
