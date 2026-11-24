import json
import numpy as np
import prepare as S
U=S.U
def bounds(point,box):
    p=np.asarray(point,float);lo=np.asarray(box[:2],float);hi=np.asarray(box[2:],float)
    near=np.minimum(np.maximum(p,lo),hi)
    far=np.maximum(np.abs(p-lo),np.abs(p-hi))
    return dict(min_distance=float(np.linalg.norm(p-near)),max_distance=float(np.linalg.norm(far)),inside=bool(np.all(p>=lo) and np.all(p<=hi)))
def aggregate(rows):
    out={}
    for part,joints in [('wrist',[0]),('thumb',[4]),('index',[8]),('tips',[4,8]),('all',[0,4,8])]:
        rr=[r for r in rows if r['joint'] in joints];marked=[r for r in rr if r['region']]
        x=dict(requested=len(rr),marked=len(marked),withheld=len(rr)-len(marked),conditions={})
        for k in 'ABC':
            vs=[r['metrics'][k] for r in marked if r['metrics'][k] is not None]
            x['conditions'][k]=dict(available=len(vs),missing=len(marked)-len(vs),inside=sum(z['inside'] for z in vs),median_min_distance=float(np.median([z['min_distance'] for z in vs])) if vs else None,median_max_distance=float(np.median([z['max_distance'] for z in vs])) if vs else None)
        x['B_vs_A']={k:sum(r.get('B_relation')==k for r in marked) for k in ['robust_closer','robust_farther','unresolved','not_available']}
        out[part]=x
    return out
def run():
    f=S.read(U/'FROZEN.json');an=S.read(U/'ANNOTATIONS.json');S.check(f['pins']);S.check(f['protected'])
    assert an['status']=='FROZEN_BEFORE_CANDIDATE_SCORING'
    # Known rectangle fixtures test inclusion, orthogonal/corner distance and far bound.
    assert bounds([5,5],[0,0,10,10])==dict(min_distance=0.,max_distance=float(np.sqrt(50)),inside=True)
    assert bounds([13,14],[0,0,10,10])['min_distance']==5
    assert bounds([0,10],[0,0,10,10])['inside']
    arrays={r['clip_id']:np.load(r['array']) for r in f['rows']};out=[]
    for r in an['points']:
        q=arrays[r['clip_id']];h=['left','right'].index(r['side']);j=r['joint'];t=r['frame'];points={};metrics={}
        for k in 'ABC':
            present=bool(q['source_valid'][t,h,j] and np.isfinite(q[k][t,h,j]).all())
            if k=='C':present=present and bool(q['c_available'][t,h])
            points[k]=q[k][t,h,j].tolist() if present else None
            metrics[k]=bounds(points[k],r['region']) if present and r['region'] else None
        relation='withheld'
        if r['region']:
            if metrics['A'] is None or metrics['B'] is None:relation='not_available'
            elif metrics['B']['max_distance']<metrics['A']['min_distance']:relation='robust_closer'
            elif metrics['B']['min_distance']>metrics['A']['max_distance']:relation='robust_farther'
            else:relation='unresolved'
        out.append({**r,'points':points,'metrics':metrics,'B_relation':relation})
    clips=[]
    for s in f['rows']:
        rr=[r for r in out if r['clip_id']==s['clip_id']];w=[r for r in rr if r['joint']==0 and r['region']]
        sufficient=all(len([r for r in w if r['side']==side])>=2 for side in ['left','right'])
        supported=bool(sufficient and all(r['metrics']['B'] and r['metrics']['B']['inside'] for r in w) and sum(r['B_relation']=='robust_closer' for r in w)>len(w)/2 and not any(r['B_relation']=='robust_farther' for r in w))
        clips.append(dict(clip_id=s['clip_id'],label=s['label'],summary=aggregate(rr),sampled_wrist_translation_supported=supported,sufficient_wrist_regions=sufficient,all21points_approved=False,training_target_approved=False))
    result=dict(status='RGB_REGION_DIAGNOSTIC_COMPLETE_NOT_TRAINING_TARGETS',at=S.now(),annotation_sha=S.sha(U/'ANNOTATIONS.json'),frozen_sha=S.sha(U/'FROZEN.json'),rows=out,clips=clips,summary=aggregate(out),independent_human_groundtruth=False,training=0,operating_changed=False)
    S.write('RESULT.json',result)
    print(json.dumps({'summary':result['summary'],'clips':[(c['label'],c['summary']['all']['conditions'],c['sampled_wrist_translation_supported']) for c in clips]},ensure_ascii=False),flush=True)
if __name__=='__main__':run()
