"""One new evaluation; retained observer/model scores are never recomputed."""
import time,traceback
from episode_model import *
import evaluate as previous
from sequence_full import observe as observe_states
NORMAL=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']
SUPPORTED='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'

def graphs():
    rows=read(D/'ACTION_TARGETS.json')['rows'];out={}
    for word in process.WORDS:
        rr=[r for r in rows if r['word_id']==word];groups=sorted({p['action_family'] for r in rr for p in r['points'] if p['action_family']<8});edges=[]
        for a in groups:
            for b in groups:
                if a==b:continue
                if all(max(p['frame'] for p in r['points'] if p['action_family']==a)<min(p['frame'] for p in r['points'] if p['action_family']==b) for r in rr):edges.append([a,b])
        out[word]=dict(groups=groups,precedences=edges,source='All9public FIT sparse event-family witnesses. No runtime forced decoding.',no_repetition_count_rule=True)
    return out

def observe(pr,r,graph):
    before=r['current'].copy();p=np.zeros((len(pr),10),np.float32);p[:,:8]=pr[:,:8];p[:,9]=pr[:,8]
    seq=observe_states(p,r['available'],graph,r['times'])
    top=pr.argmax(1);conf=pr.max(1);av=r['available'];foreign=av&(conf>=.8)&(top!=8)&~np.isin(top,graph['groups'])
    if foreign.any() and seq['status']==SUPPORTED:seq['status']='UNEXPECTED_ACTION_CONTENT_UNCONFIRMED'
    core=np.flatnonzero(r['y']==1)
    uncertain=[int(f) for f in core if not(av[f] and conf[f]>=.8 and top[f] in graph['groups'])]
    known=np.flatnonzero((r['episode']>=0)&(r['episode']<8))
    mistakes=[int(f) for f in known if top[f]!=r['episode'][f]]
    seq.update(unexpected_action_frames=np.flatnonzero(foreign).tolist(),previous_reviewed_core_unconfirmed=uncertain,
      evaluation_action_errors=mistakes,raw_observations_preserved=True,scope='Action-event content/order; physical shape and directed motion remain independent. Never full language correctness.')
    assert np.array_equal(before,r['current']);return seq

def behavior():
    graph={'groups':[0,1,2],'precedences':[[0,1],[1,2],[0,2]]}
    def call(ids,gaps=()):
        n=len(ids);p=np.eye(9)[ids];r=dict(current=np.zeros((n,10)),available=np.ones(n,bool),times=np.arange(n)/30,y=np.ones(n,int),episode=np.array(ids))
        r['available'][list(gaps)]=False;return observe(p,r,graph)
    assert call([0,0,1,1,2,2])['status']==SUPPORTED
    assert call([2,2,1,1,0,0])['status']=='PHYSICAL_ORDER_CONFLICT'
    assert call([0,0,1,1,0,2,2])['status']!=SUPPORTED
    assert call([0,0,2,2])['status']!=SUPPORTED
    assert call([0,0,1,1,2,2],[2])['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
    return dict(reverse=True,short_contrary=True,missing_family=True,gap_preserved=True)

def main():
    assert read(D/'TRAIN_PROGRESS.json')['status']=='COMPLETE' and not (D/'EVALUATION.json').exists()
    torch.set_num_threads(2);plan=read(D/'FROZEN.json');s=read(D/'NORMALIZATION.json');tar=read(D/'DEV_ACTION_TARGETS.json')
    targets={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in tar['rows']}
    rows=attach_targets(process.add_rgb(process.old.inputs(False)),targets);gg=graphs();assert len(rows)==88
    pins={r['input']:r['input_sha256'] for r in rows}
    for r in rows:pins.update(dict(r['extra_pins']));pins[r['base_path']]=r['base_sha256']
    for p in [D/'DEV_ACTION_TARGETS.json',D/'evaluate_episode.py',D/'FROZEN.json']:pins[str(p)]=sha(p)
    exclusive(D/'EVALUATION_PROTOCOL.json',dict(at=clock(),primary=plan['primary'],cases=88,graphs=gg,behavior=behavior(),pins=pins,old_reinference=0,
      metric_definition='Sparse event-family labels score this NEW task; do not compare percentages to old9posture/3role accuracy. Compare same-task readout/episode and old saved functional sequence results.',
      gain_audit='A newly supported normal sequence is not accepted if a previously reviewed core frame is rest/uncertain or a newly reviewed action witness is misassigned. This is evaluation auditing, not a runtime oracle/filter.'))
    reports=[]
    for kind in plan['conditions']:
        for seed in plan['seeds']:
            tag=f'{kind}_{seed}';m=EpisodeNet(kind=='episode');path=D/'runs'/tag/'model.pt';assert sha(path)==read(path.parent/'COMPLETE.json')['sha256'];m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));m.eval()
            points=[];seqs=[];feedback=[];seconds=[]
            for r in rows:
                x,im,l,y=batch([r],s);t=time.perf_counter()
                with torch.inference_mode():pr=m(x,im,l).softmax(-1)[0].numpy()
                seconds.append(time.perf_counter()-t);sq=dict(key=r['key'],word=r['word_id'],kind=r['kind'],**observe(pr,r,gg[r['word_id']]));seqs.append(sq)
                fb=previous.compose(r,sq);fb.update(key=r['key'],kind=r['kind'],word=r['word_id']);feedback.append(fb);pins[fb['saved_movement_source']]=fb['saved_movement_sha256']
                for f in np.flatnonzero(r['episode']>=0):points.append(dict(key=r['key'],clip_id=r['clip_id'],person=r['person'],word=r['word_id'],kind=r['kind'],frame=int(f),source_frame=int(r['mapping'][f]),y=int(r['episode'][f]),pred=int(pr[f].argmax()),probability=pr[f].tolist()))
                out=D/'predictions'/tag/f"{r['key']}.npz";out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
                np.savez_compressed(out,probability=pr,current_probability=r['current'],image_available=r['available'],frame_id=np.arange(len(pr)))
            norms=[p for p in points if p['kind'] in NORMAL];acts=[p for p in norms if p['y']<8]
            report=dict(condition=tag,points=points,sequences=seqs,feedback=feedback,normal_metrics=process.state.metrics(norms),normal_action_metrics=process.state.metrics(acts),head_median_seconds=float(np.median(seconds)))
            exclusive(D/'evaluations'/f'{tag}.json',report);reports.append({k:v for k,v in report.items() if k not in ['points','sequences','feedback']})
            print(tag,'normal action',report['normal_action_metrics']['correct'],'/',len(acts),'order',sum(q['kind'] in NORMAL and q['status']==SUPPORTED for q in seqs),flush=True)
    for p,h in pins.items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(D/'EVALUATION.json',dict(at=clock(),reports=reports,cases=88,comparisons=528,pins=pins,whole_goal_complete=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (D/'EVAL_FIRST_FAILURE.txt').exists():(D/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
