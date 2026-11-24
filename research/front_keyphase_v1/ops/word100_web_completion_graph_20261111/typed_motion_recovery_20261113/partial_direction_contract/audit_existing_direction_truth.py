"""All existing axis2 FIT directional/hold targets under the palm contract."""
from audit import *
from palm_projection_core import geometry

def main():
    assert not(U/'FIT_DIRECTION_TARGET_AUDIT.json').exists()
    meta=read(HERE/'INPUTS_COMPLETE.json')
    rows=[dict(r,index=i)for i,r in enumerate(meta['rows'])if r['split']=='train'and r['axis']==2]
    assert all(r['person']in range(1,8)for r in rows)
    ready={r['clip_id']:r for r in read(OPS/'word100_shared_motion_learning_20261109/READY.json')['rows']if r['role']=='fit'and r['person']in range(1,8)}
    crops={r['clip_id']:r for p in[old.R/'CROP_INPUTS_COMPLETE.json',old.D/'CROP_INPUTS_COMPLETE.json']for r in read(p)['rows']if r['clip_id']in ready}
    sys.path.insert(0,str(OPS/'word100_observation_guard_20261108'))
    guard=old.load('_fit_palm_direction_guard',OPS/'word100_observation_guard_20261108/guard.py');ad=old.load('_fit_palm_direction_ad',OPS/'word100_rtmw_shared_input_20261108/adapter.py')
    cfg=dict(at=stamp(),scope='All existing M/H FIT axis2 directional and hold targets, people01..07; no probe arrays opened; no partial directional labels added',
        measurement='Pinned palm root+MCP58, actual same interval and stored reverse flag; preserve every target and denominator',
        pins={str(p):sha(p)for p in[Path(__file__),HERE/'INPUTS_COMPLETE.json',U/'palm_projection_core.py']},no_training=True,no_model_inference=True)
    assert not(U/'FIT_DIRECTION_TARGET_CONTRACT.json').exists();write(U/'FIT_DIRECTION_TARGET_CONTRACT.json',cfg)
    cache={};pins={};out=[]
    for r in rows:
        key=r['clip_id'],r['view']
        if key not in cache:
            sr=ready[r['clip_id']];assert sr['person']==r['person']
            for p,h in[(sr['path'],sr['sha256']),(sr['channel'],sr['channel_sha256'])]:assert sha(p)==h;pins[p]=h
            zz=np.load(sr['path']);ch=np.load(sr['channel']);tt=zz['timestamp_us']
            if r['view']=='actual_crop':
                cr=crops[r['clip_id']];assert sha(cr['path'])==cr['sha256'];pins[cr['path']]=cr['sha256'];cz=np.load(cr['path']);assert np.array_equal(tt,cz['timestamp_us']);cl=guard.quarantine(cz['xy_px'],cz['score'],tt,640,480);x=ad.convert(cl['xy_px'],cz['score'],tt,cz['frame_id'],640,480)['tcn280']
            else:x=ch['full']if r['view']=='full'else ch['feedback']if'feedback'in ch.files else ch['upper_mask']
            cache[key]=(x,tt)
        x,tt=cache[key];g,ok,info=geometry(x,tt,r['a'],r['b'],2,r['moving'],r['anchor'],r['reverse'])
        out.append(dict(index=r['index'],clip_id=r['clip_id'],person=r['person'],origin=r['origin'],view=r['view'],reverse=r['reverse'],target=r['target'],span=[r['a'],r['b']],eligible=bool(ok),coverage=info['coverage'],observed=info['observed'],delta=float(g[40])if ok else None,
            sign_conflict=bool((r['target']==2 and g[40]<=0)or(r['target']==0 and g[40]>=0))if ok and r['target']!=1 else None,
            anchor_length_range=info.get('anchor_length_range')))
    groups=defaultdict(list)
    for r in out:groups[r['origin'],r['target']].append(r)
    summary=[]
    for k,rs in sorted(groups.items()):
        good=[r for r in rs if r['eligible']]
        summary.append(dict(origin=k[0],target=k[1],rows=len(rs),eligible=len(good),unavailable=len(rs)-len(good),delta=numbers([r['delta']for r in good]),sign_conflicts=sum(r['sign_conflict']is True for r in rs)))
    for p,h in {**pins,**cfg['pins']}.items():assert sha(p)==h
    write(U/'FIT_DIRECTION_TARGET_AUDIT.json',dict(at=stamp(),rows=out,summary=summary,source_people=sorted(set(r['person']for r in out)),unavailable=sum(not r['eligible']for r in out),directional_conflicts=sum(r['sign_conflict']is True for r in out),pins=pins,contract_sha256=sha(U/'FIT_DIRECTION_TARGET_CONTRACT.json'),new_labels=False,model_inference=False,scope='FIT point-contract consistency, not independent accuracy'))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
