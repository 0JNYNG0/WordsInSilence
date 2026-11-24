"""Separate scalar/reference checker; does not import candidate probe."""
import json,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_uncertain_hand_preservation_20261018';V=O/'word100_assignment_provenance_20261018';S=O/'word100_single_arm_recheck_20261018';G=O/'word100_visible_arm_duplicate_20261018'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def ref(z,p,pr,s,g,f,i):
    if p['identity_side'][f,i]!=-1 or p['sides'][f,i]<0 or not any(p['point_mask'][f,i]):return None
    def bad(q):return z['raw_count'][q]==2 and any(g['coverage'][q,a,1-a]>=.8 for a in (0,1))
    if bad(f):return None
    path=[[f,i]];current=i;seen=set();prev_anchor=-1;prev_frame=-2
    for q in range(f+1,len(z['raw_count'])):
        delta=int(z['timestamp_us'][q])-int(z['timestamp_us'][f])
        if not(0<delta<=100000 and z['timestamp_us'][q]>z['timestamp_us'][q-1]):break
        if bad(q) or z['raw_count'][q]==0 or z['raw_count'][q-1]==0:break
        edges=[(a,b) for a in range(int(z['raw_count'][q])) for b in range(int(z['raw_count'][q-1])) if p['possible_links'][q,a,b]]
        following=[a for a,b in edges if b==current]
        if len(following)!=1:break
        current=following[0]
        if len([b for a,b in edges if a==current])!=1:break
        path.append([q,current]);e=int(p['identity_side'][q,current])
        if e>=0:seen.add(e)
        if len(seen)>1:break
        direct=e>=0 and ((pr['current_pose'][q,current] and pr['sides'][q,current]==e) or s['action'][q,current] in (2,3))
        if direct and prev_anchor==e and prev_frame==q-1:
            return {'side':e,'delay_us':delta,'available_at_us':int(z['timestamp_us'][q]),'anchor_frames':[q-1,q],'path':path}
        prev_anchor=e if direct else -1;prev_frame=q if direct else -2
    return None

def main():
    c=read(U/'CONTEXT.json');saved=read(U/'PROPOSALS.json');expected={(x['clip_id'],x['view'],x['index'],x['raw_index']):x for x in saved};anns=read(U/'ANNOTATED.json');ak={(x['clip_id'],x['view'],x['index'],x['raw_index']):x for x in anns};found={};frames=hands=points=elbows=0
    for r in c['rows']:
        assert r['role']=='train_candidate' and 1<=r['signer_id']<=9
        for v in ('masked','sensor'):
            clip=r['clip_id'];z=load(Path(r['views'][v]['raw']));p=load(P/'results'/f'{clip}_{v}.npz');pr=load(V/'results'/f'{clip}_{v}.npz');s=load(S/'results'/f'{clip}_{v}.npz');g=load(G/'results'/f'{clip}_{v}.npz')
            for f,n in enumerate(z['raw_count']):
                frames+=1;hands+=int(n);points+=int(p['point_mask'][f,:n].sum());candidates={}
                for i in range(int(n)):
                    x=ref(z,p,pr,s,g,f,i)
                    if x is not None:candidates[i]=x
                    key=(clip,v,f,i)
                    if key in ak:
                        yy=ak[key]['elbow_only'];ok=bool(p['point_mask'][f,i,0]);ds=[]
                        for a in (13,14):
                            xx,y=z['pose_xy_px'][f,a];ok=ok and bool(z['pose_usable'][f,a] and math.isfinite(float(xx)) and math.isfinite(float(y)) and 0<=xx<z['width'][f] and 0<=y<z['height'][f]);hx,hy=z['hand_xy_px'][f,i,0];ds.append(math.hypot(float(xx)-hx,float(y)-hy))
                        if ok and ds[0]!=ds[1]:
                            assert yy is not None and yy['side']==(0 if ds[0]<ds[1] else 1);assert np.allclose(ds,yy['distance_px'],rtol=1e-12,atol=1e-12);elbows+=1
                        else:assert yy is None
                accepted={i:x for i,x in candidates.items() if all(j==i or (candidates[j]['side'] if j in candidates else int(p['identity_side'][f,j]))!=x['side'] for j in range(int(n)))}
                for i,x in accepted.items():
                    key=(clip,v,f,i);assert key in expected,key
                    for k,y in x.items():assert y==expected[key][k],(key,k)
                    found[key]=True
    assert set(found)==set(expected)
    summary=read(U/'SUMMARY.json');assert frames==summary['counts']['frames'] and hands==summary['counts']['raw_candidates'] and points==summary['counts']['observed_points_preserved']
    for file,h in c['pins'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==h,file
    normal=[x for x in anns if x['verdict']=='normal_identity'];remain=read(U/'REMAINING9.json');cross=read(U/'CROSSING.json')
    assert len(normal)==441 and all(x['hypothetical_side']==x['actual_side'] for x in normal)
    assert len(remain)==9 and not any(x['delayed_proposal'] for x in remain) and sum(x['all_observed_points'] for x in remain)==72
    assert len(cross)==315 and all(x['original_sides']==x['delayed_sides'] for x in cross)
    assert sum(x['elbow_only'] is not None and x['elbow_only']['side']!=x['actual_side'] for x in normal)==5
    out={'status':'PASS','frames':frames,'raw_candidates':hands,'original_points_unchanged':points,'delayed_proposals_verified':len(found),'elbow_annotations_verified':elbows,'remaining9_recovered':0,'normal441_preserved':441,'crossing315_preserved':315,'normal_elbow_counterexamples':5,'protected_pins_verified':len(c['pins']),'independence':'Separate reference implementation by same Codex; no independent human labels'}
    (U/'INDEPENDENT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
