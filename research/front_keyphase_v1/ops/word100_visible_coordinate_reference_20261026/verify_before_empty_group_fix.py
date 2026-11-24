"""Independent arithmetic route: rectangle distances and frozen point provenance."""
from collections import Counter
import math,numpy as np
import prepare as S
U=S.U
def run():
    f=S.read(U/'FROZEN.json');a=S.read(U/'ANNOTATIONS.json');r=S.read(U/'RESULT.json');p=S.read(U/'PREPARED.json')
    assert r['annotation_sha']==S.sha(U/'ANNOTATIONS.json') and r['frozen_sha']==S.sha(U/'FROZEN.json')
    assert len(r['rows'])==len(a['points'])==108
    pins={**f['pins'],**f['protected']}
    for clip in p['clips']:
        for z in clip['frames']:
            for kind in ['raw','grid']:pins[str(U/z[kind])]=z[kind+'_sha']
    S.check(pins)
    arrays={x['clip_id']:np.load(x['array']) for x in f['rows']};valid=0;unknown=0;inside=Counter();comparisons=Counter();catalog=[];lost=[];gained=[]
    for ann,row in zip(a['points'],r['rows']):
        for key,val in ann.items():assert row[key]==val,key
        q=arrays[row['clip_id']];h=['left','right'].index(row['side']);j=row['joint'];t=row['frame'];box=row['region'];bounds={}
        for k in 'ABC':
            value=q[k][t,h,j];present=bool(q['source_valid'][t,h,j] and np.isfinite(value).all())
            if k=='C':present=present and bool(q['c_available'][t,h])
            assert (row['points'][k] is not None)==present
            if present:np.testing.assert_allclose(value,row['points'][k],atol=0,rtol=0)
            if box is None or not present:
                assert row['metrics'][k] is None;continue
            x,y=value;x0,y0,x1,y1=box
            low=math.hypot(max(x0-x,0,x-x1),max(y0-y,0,y-y1))
            high=max(math.hypot(x-i,y-jj) for i in [x0,x1] for jj in [y0,y1])
            within=x0<=x<=x1 and y0<=y<=y1
            np.testing.assert_allclose([low,high],[row['metrics'][k]['min_distance'],row['metrics'][k]['max_distance']],rtol=1e-10,atol=1e-8)
            assert bool(within)==row['metrics'][k]['inside'];inside[k]+=int(within);bounds[k]=(low,high,within)
        if box:
            valid+=1
            if len(bounds)<2:rel='not_available'
            elif bounds['B'][1]<bounds['A'][0]:rel='robust_closer'
            elif bounds['B'][0]>bounds['A'][1]:rel='robust_farther'
            else:rel='unresolved'
            assert rel==row['B_relation'];comparisons[rel]+=1
            if bounds['A'][2] and not bounds['B'][2]:lost.append([row['clip_id'],t,h,j])
            if not bounds['A'][2] and bounds['B'][2]:gained.append([row['clip_id'],t,h,j])
            if bounds['A'][2]:status='original_point_review_candidate'
            elif bounds['B'][2] and rel=='robust_closer':status='offset_corrected_point_review_candidate'
            else:status='visible_but_coordinate_match_unresolved'
        else:unknown+=1;assert row['B_relation']=='withheld';status='visibility_or_identity_insufficient'
        catalog.append(dict(clip_id=row['clip_id'],label=row['label'],frame=t,side=row['side'],joint=j,region=box,status=status,approved_for_training=False,reason=row['reason']))
    assert (valid,unknown)==(70,38)
    assert dict(inside)=={k:r['summary']['all']['conditions'][k]['inside'] for k in 'ABC'}
    assert all(comparisons[k]==r['summary']['all']['B_vs_A'][k] for k in comparisons)
    for c in r['clips']:
        rr=[x for x in r['rows'] if x['clip_id']==c['clip_id']]
        for part,ids in [('wrist',[0]),('thumb',[4]),('index',[8]),('tips',[4,8]),('all',[0,4,8])]:
            group=[x for x in rr if x['joint'] in ids];good=[x for x in group if x['region']];s=c['summary'][part]
            assert s['requested']==len(group) and s['marked']==len(good)
            for k in 'ABC':
                met=[x['metrics'][k] for x in good if x['metrics'][k] is not None]
                assert s['conditions'][k]['inside']==sum(x['inside'] for x in met)
                for metric in ['min_distance','max_distance']:
                    np.testing.assert_allclose(s['conditions'][k]['median_'+metric],np.median([x[metric] for x in met]))
        assert not c['training_target_approved'] and not c['all21points_approved']
    S.write('REFERENCE_CATALOG.json',dict(status='REVIEW_CANDIDATES_ONLY_NOT_TRAINING_LABELS',points=catalog,counts=dict(Counter(x['status'] for x in catalog)),approved_training_points=0,warning='Same-Codex approximate regions; candidate categories are descriptive audit triage, not a validated automatic correction rule. Candidate coordinates remain in separate prior experiment arrays.'))
    S.write('VERIFIED.json',dict(status='PASS_SOURCE_AND_RECTANGLE_ARITHMETIC',at=S.now(),videos=6,frames=18,points=108,visible_regions=valid,withheld=unknown,inside=dict(inside),B_relations=dict(comparisons),new_inside=gained,new_outside=lost,protected_files=len(f['protected']),pins=pins,annotation_sha=S.sha(U/'ANNOTATIONS.json'),result_sha=S.sha(U/'RESULT.json'),catalog_sha=S.sha(U/'REFERENCE_CATALOG.json'),verification='Same Codex, separate arithmetic implementation. Not independent human annotations or true coordinate accuracy.',operating_change=False,training=0))
    print({'status':'VERIFIED','inside':dict(inside),'new_inside':len(gained),'new_outside':len(lost),'catalog':dict(Counter(x['status'] for x in catalog))},flush=True)
if __name__=='__main__':run()
