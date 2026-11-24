"""Two predeclared global crossovers of saved neural outputs. No retraining."""
from pathlib import Path
import sys
U=Path(__file__).resolve().parent;O=U.parent;N=O/'word100_nonmain_probability_20261102'
sys.path.insert(0,str(N))
from contract import eval_rows,previous,observe,read,sha,exclusive,clock,np,NORMAL,OK,FLOW,OUT
CONDITIONS={'new_content_old_role':('new','old'),'old_content_new_role':('old','new')}

def prepare():
    assert not (U/'FROZEN.json').exists()
    v=read(N/'VERIFIED.json');assert v['status']=='COMMON_CALCULATION_GATES_MET'
    sources={'old':dict(directory=str(FLOW/'predictions'),content_key='content_probability',baseline=v['conditions']['prior_flow']),
             'new':dict(directory=str(OUT/'predictions/new_adapt'),content_key='raw_action_probability',baseline=v['conditions']['new_adapt'])}
    pins={str(p):sha(p) for p in [U/'run.py',U/'verify.py',N/'contract.py',N/'VERIFIED.json',N/'comparisons/prior_flow.json',N/'comparisons/new_adapt.json']}
    for source in sources.values():
        files=sorted(Path(source['directory']).glob('*.npz'));assert len(files)==88
        for p in files:pins[str(p)]=sha(p)
    exclusive(U/'FROZEN.json',dict(at=clock(),sources=sources,conditions=CONDITIONS,primary='new_content_old_role',
        rationale='New primary has590raw correct versus589 but12correct-main exclusions versus7. Test new content with prior process role globally; reverse crossover separates causes. Choice made before crossover scores.',
        constant='Verified nonmain-union calculation, source graphs, thresholds0.8, native rows/time and uncertainty handling. No per-word/frame selection, retraining or old evaluation overwrite.',
        family_baseline=read(O/'word100_guided_event_order_20261102/FROZEN.json')['baseline_family_eligible'],
        gates=['raw_at_least589','used_at_least582','family_not_worse','normal_order_at_least19','clean_order_at_least5','short_reverse_at_least2',
            'normal_false_order0','reverse_false_pass0','missing_held','raw_role_time_exact'],
        pins=pins,whole_goal_complete=False))
    print('Frozen two whole-model-output crossovers; primary new content plus prior role; no neural inference')

def compare():
    assert not (U/'COMPARISON.json').exists();plan=read(U/'FROZEN.json');rows=eval_rows();graphs=previous.graphs();summaries=[]
    assert len(rows)==88 and {r['person'] for r in rows}=={11,12,13}
    for p,h in plan['pins'].items():assert sha(p)==h
    for tag,(whichcontent,whichrole) in plan['conditions'].items():
        cases=[];points=[]
        for r in rows:
            a={name:np.load(Path(s['directory'])/(r['key']+'.npz')) for name,s in plan['sources'].items()}
            for k in ['current_probability','image_available','source_frame_id','timestamp','frame_id']:assert np.array_equal(a['old'][k],a['new'][k])
            for k,arr in [('current_probability',r['current']),('image_available',r['available']),('source_frame_id',r['mapping']),('timestamp',r['times'])]:assert np.array_equal(a['old'][k],arr)
            content=a[whichcontent][plan['sources'][whichcontent]['content_key']];activity=a[whichrole]['activity_probability']
            z=observe(content,activity,r,graphs[r['word_id']]);q=z['sequence'];q.update(key=r['key'],word=r['word_id'],kind=r['kind'])
            cases.append(dict(sequence=q,eligible=z['eligible'].tolist(),role_probability=z['role_probability'].tolist()))
            for f in np.flatnonzero(r['episode']>=0):points.append(dict(key=r['key'],word=r['word_id'],kind=r['kind'],person=r['person'],frame=int(f),source_frame=int(r['mapping'][f]),
                y=int(r['episode'][f]),raw_pred=int(content[f].argmax()),eligible=bool(z['eligible'][f])))
        main=[p for p in points if p['kind'] in NORMAL and p['y']<8];ns=[c['sequence'] for c in cases if c['sequence']['kind'] in NORMAL]
        rev=[c['sequence'] for c in cases if c['sequence']['kind']=='actual_reverse_main' and c['sequence']['precedences']]
        m=dict(raw_correct=sum(p['y']==p['raw_pred'] for p in main),used_correct=sum(p['y']==p['raw_pred'] and p['eligible'] for p in main),
            family_used={str(g):sum(p['y']==g and p['raw_pred']==g and p['eligible'] for p in main) for g in range(8)},
            normal_order=sum(s['status']==OK and bool(s['precedences']) for s in ns),
            clean_order=sum(s['status']==OK and bool(s['precedences']) and not(s['evaluation_action_errors'] or s['evaluation_content_errors'] or s['previous_reviewed_core_unconfirmed']) for s in ns),
            short_reverse=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in rev),reverse_false_pass=sum(s['status']==OK for s in rev),false_order=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in ns))
        exclusive(U/'comparisons'/f'{tag}.json',dict(condition=tag,content_source=whichcontent,role_source=whichrole,cases=cases,points=points,metrics=m,whole_goal_complete=False))
        summaries.append(dict(condition=tag,**m));print(tag,m,flush=True)
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(U/'COMPARISON.json',dict(at=clock(),reports=summaries,comparisons=176,neural_inference=0,training=0,whole_goal_complete=False))

if __name__=='__main__':
    if sys.argv[1:] == ['prepare']:prepare()
    elif sys.argv[1:] == ['compare']:compare()
    else:raise ValueError('Choose prepare or compare')
