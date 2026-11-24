"""Source-episode membership labels. No framewise label interpolation.

This is a new isolated task: does a window contain observed evidence belonging
to the queried guide action? Presence is not completion or sign correctness.
"""
from pathlib import Path
import json,hashlib,collections
import numpy as np
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
FAMILIES=[2,3,4,5,7]
MAP={0:0,1:1,2:2,3:3,4:4,5:4,6:5,7:5,8:6,9:7}

def features(z,a,b):
    # B: unchanged time-window summary, C: start/middle/end of this window.
    # These thirds are input pooling bins, never invented phase targets.
    points=np.arange(a,b+1);v=z[:,228:456]
    bins=np.array_split(points,3)
    if any(len(g)==0 for g in bins):bins=[points[:1],points,points[-1:]]
    return z[a:b+1].mean(0).astype(np.float32),np.concatenate([v[g].mean(0) for g in bins]).astype(np.float32)

def main():
    assert not (W/'INPUTS_COMPLETE.json').exists()
    paths=dict(annotations=O/'word100_autonomous_localization_20261111/FIT_LABELS.json',guide=O/'word100_event_guide_linkage_20261110/GUIDE_CONTRACTS.json',cache=U/'context_role_training/INPUTS_COMPLETE.json',raw=O/'word100_shared_motion_learning_20261109/READY.json')
    guides=read(paths['guide'])['words'];anns=read(paths['annotations'])['rows']
    source={r['clip_id']:r for r in read(paths['raw'])['rows'] if r['role']=='fit' and r['word_id'] in guides}
    cache={(r['clip_id'],r['view']):r for r in read(paths['cache'])['rows']}
    selected=[a for a in anns if a['clip_id'] in source and a['role']=='fit']
    assert len(selected)==15 and len({a['clip_id'] for a in selected})==12
    rows=[];deferred=[];provenance=[]
    for cid in sorted({a['clip_id'] for a in selected}):
        s=source[cid];word=s['word_id'];assert sha(s['path'])==s['sha256'];raw=np.load(s['path'],allow_pickle=False)
        assert np.array_equal(raw['frame_id'],np.arange(len(raw['frame_id'])))
        aa=[a for a in selected if a['clip_id']==cid];actions=[];holds=[]
        for ann in aa:
            assert ann['source_sha256']==s['source_sha256'];assert sha(ann['sheet'])==ann['sheet_sha256']
            clauses=[g for g in guides[word] if g['axis']==ann['axis'] and g.get('parent') is None]
            assert len(clauses)==1;clause=clauses[0];families={MAP[k] for k in clause['contexts']};assert len(families)==1
            family=next(iter(families));assert family in FAMILIES
            # Only existing RGB-reviewed episode boundaries are used. A whole
            # episode label does not label every frame or prove exact contact.
            for j,e in enumerate(ann['events']):
                a,b=e['onset'][0],e['offset'][1]
                if b<=a:deferred.append(dict(clip_id=cid,reason='no_temporal_extent'));continue
                base=dict(a=a,b=b,family=family,axis=ann['axis'],moving=ann['moving'],clause=clause['id'],event=j,kind='reviewed_event',complete_semantic_label=None)
                actions.append(base)
                for lo,hi in ann['holds']:
                    if lo<=b+1<=hi and hi>b:
                        actions.append(dict(base,b=hi,kind='reviewed_event_plus_adjacent_hold',hold=[lo,hi]))
            for lo,hi in ann['holds']:
                # Any possible movement overlap makes a hold-only teacher unsafe.
                if any(not(hi<e['onset'][0] or lo>e['offset'][1]) for e in ann['events']):
                    lo=max([lo]+[e['offset'][1]+1 for e in ann['events'] if e['onset'][0]<=hi])
                if hi>lo:holds.append(dict(a=lo,b=hi,family=family,axis=ann['axis'],moving=ann['moving'],clause=clause['id'],kind='reviewed_hold_only'))
        for view in ['full','upper']:
            c=cache[cid,view];assert sha(c['path'])==c['sha256'];z=np.load(c['path'],allow_pickle=False)['z'];assert len(z)==len(raw['frame_id'])
            for item in actions+holds:
                a,b=item['a'],item['b'];axis=item['axis'];k=axis*2+(item['moving']=='L')
                # Cached normalized q retains the observed flag unchanged.
                observed=z[:,292:392].reshape(-1,10,10)[:,k,8]>.5
                if not observed[a:b+1].all():deferred.append(dict(clip_id=cid,view=view,item=item,reason='observation_gap'));continue
                y=np.zeros(5,np.float32);mask=np.zeros(5,np.float32);family=item['family'];j=FAMILIES.index(family)
                y[j]=0 if item['kind']=='reviewed_hold_only' else 1;mask[j]=1;targets=[dict(family=family,value=int(y[j]),basis=item['kind'])]
                # Different-action negatives require a separate reviewed guide
                # episode in the SAME recording, with no interval overlap.
                if y[j]:
                    for other in actions:
                        if other['kind']!='reviewed_event' or other['family']==family:continue
                        if other['b']<a or other['a']>b:
                            oi=FAMILIES.index(other['family']);mask[oi]=1;y[oi]=0
                            if not any(t['family']==other['family'] for t in targets):targets.append(dict(family=other['family'],value=0,basis='distinct_reviewed_source_guide_episode',other_span=[other['a'],other['b']]))
                mean,local=features(z,a,b);p=W/'inputs'/f'{len(rows):04d}.npz';p.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(p,mean=mean,local=local,y=y,mask=mask)
                rows.append(dict(path=str(p),sha256=sha(p),clip_id=cid,source_sha256=s['source_sha256'],view=view,person=s['person'],split='probe' if s['person']==7 else 'train',span=[a,b],item=item,targets=targets,cache=c['path'],cache_sha256=c['sha256']))
        provenance.append(dict(clip_id=cid,raw_cache=s['path'],raw_sha256=s['sha256'],source_sha256=s['source_sha256'],guide_word=word,annotation_count=len(aa)))
    report=dict(rows=rows,deferred=deferred,source_clips=len(provenance),source_records=provenance,counts=dict(collections.Counter(f"{r['split']}/{t['family']}/{t['value']}" for r in rows for t in r['targets'])),families=FAMILIES,pins={str(p):sha(p) for p in [Path(__file__)]+list(paths.values())},scope='FIT source-episode membership only. Four words supply initial reviewed guide relations; common learned observation100 retained. No word ID, axis ID, expected order or source frame index enters features. No whole-sign pass, exact finger truth, or framewise action-boundary supervision.',limits=['REAL07 previously exposed to old encoders/heads; continuation diagnostic only','Episode family linked from reviewed guide and observed source movement, not independently human reviewed new labels','Source labels at uncertain endpoints do not prove every frame is in the movement','Distinct source episodes are association negatives, not proof their geometry cannot overlap','Half an alternating cycle is evidence of that source action only, not completed alternation or repetition count','A hold-only negative concerns the queried motion; the held hand may still be an important valid posture','Other common100 words are not relabeled into these five meanings by their name','No private footage or runtime writes'],training_started=False,web_changed=False)
    write(W/'INPUTS_COMPLETE.json',report)
    print('Membership samples',len(rows),'targets',sum(len(r['targets']) for r in rows),'source clips',len(provenance),'deferred',len(deferred));print(report['counts'])

if __name__=='__main__':main()
