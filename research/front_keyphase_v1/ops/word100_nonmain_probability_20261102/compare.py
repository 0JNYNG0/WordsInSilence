from contract import *

def main():
    assert not (U/'COMPARISON.json').exists();plan=read(U/'FROZEN.json')
    for p,h in plan['pins'].items():assert sha(p)==h
    rows=eval_rows();graphs=previous.graphs();reports=[];count=0
    assert len(rows)==88 and {r['person'] for r in rows}=={11,12,13}
    for spec in plan['models']:
        old=read(spec['report']);oldseq={s['key']:s for s in old['sequences']};cases=[];points=[]
        for r in rows:
            source=Path(spec['arrays'])/(r['key']+'.npz');a=np.load(source);content=a[spec['content_key']];activity=a['activity_probability']
            for key,value in [('current_probability',r['current']),('image_available',r['available']),('source_frame_id',r['mapping']),('timestamp',r['times'])]:assert np.array_equal(a[key],value)
            z=observe(content,activity,r,graphs[r['word_id']]);q=z['sequence'];q.update(key=r['key'],word=r['word_id'],kind=r['kind'])
            base=oldseq[r['key']]
            cases.append(dict(sequence=q,old_status=base['status'],source_array=str(source),eligible=z['eligible'].tolist(),nonmain_probability=z['nonmain_probability'].tolist()))
            for f in np.flatnonzero(r['episode']>=0):
                points.append(dict(key=r['key'],word=r['word_id'],kind=r['kind'],frame=int(f),source_frame=int(r['mapping'][f]),y=int(r['episode'][f]),
                    raw_pred=int(content[f].argmax()),eligible=bool(z['eligible'][f])))
            count+=1
        major=[p for p in points if p['kind'] in NORMAL and p['y']<8];ns=[c['sequence'] for c in cases if c['sequence']['kind'] in NORMAL]
        rev=[c['sequence'] for c in cases if c['sequence']['kind']=='actual_reverse_main' and c['sequence']['precedences']]
        metrics=dict(raw_correct=sum(p['y']==p['raw_pred'] for p in major),known_major=len(major),
            used_correct=sum(p['y']==p['raw_pred'] and p['eligible'] for p in major),
            normal_order=sum(s['status']==OK and bool(s['precedences']) for s in ns),
            clean_order=sum(s['status']==OK and bool(s['precedences']) and not(s['evaluation_action_errors'] or s['evaluation_content_errors'] or s['previous_reviewed_core_unconfirmed']) for s in ns),
            short_reverse=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in rev),reverse_false_pass=sum(s['status']==OK for s in rev),
            false_order=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in ns),
            subtype_uncertain_nonmain_frames=sum(len(s['nonmain_subtype_uncertain_frames']) for s in ns))
        report=dict(tag=spec['tag'],cases=cases,points=points,metrics=metrics,baseline=spec['baseline'],raw_probabilities_modified=False)
        exclusive(U/'comparisons'/f"{spec['tag']}.json",report);reports.append(dict(tag=spec['tag'],**metrics));print(spec['tag'],metrics,flush=True)
    assert count==440
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(U/'COMPARISON.json',dict(at=clock(),reports=reports,new_interpretation_cases=count,neural_inference=0,old_scores_changed=False,whole_goal_complete=False))
if __name__=='__main__':main()
