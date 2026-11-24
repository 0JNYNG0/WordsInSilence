"""New branch once; old comparisons reused, not re-scored or re-inferred."""
import time,traceback
from process_model import *
from role_sequence import observe,behavior
from train import evaluate_points
NORMAL=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']
R=O/'word100_learned_rgb_motion_20261102'

def movement(r):
    rr=read(R/'COMPARISON.json')['rows']
    cid=r['clip_id'] if r['kind']=='native' else (r['clip_id']+'_reverse' if r['kind']=='actual_reverse' else r['word_id']+'_'+r['kind'])
    match=[v for v in rr if v['condition']=='rgb_residual' and v['seed']==42 and v['case']==cid]
    if match:return match[0]['motions'],str(R/'COMPARISON.json')
    if r['kind']=='actual_reverse_main':
        s=next(v for v in read(G/'MAIN_ORDER_RESULTS.json')['rows'] if v['word']==r['word_id'])
        return read(s['result'])['motion_observations'],s['result']
    if r['kind'].startswith('pose_'):
        s=next(v for v in read(G/'SENSOR_GAP_RESULTS.json')['rows'] if v['word']==r['word_id'] and f"{v['kind']}_f{v['center']}"==r['kind'])
        return s['motions'],str(G/'SENSOR_GAP_RESULTS.json')
    raise ValueError(('Missing saved movement evidence',cid))

def compose(r,seq):
    motions,path=movement(r);base=read(r['base_path'])
    order=seq['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
    motion_confirmed=bool(motions) and all(v['status']=='DIRECTED_MOTION_OBSERVED' for v in motions)
    # Observable state precedence and required movement must both survive.
    status='SUPPORTED_REVIEWED_ORDER_AND_MOVEMENT' if order and motion_confirmed else 'PHYSICAL_COMPONENTS_PARTIAL_OR_UNCONFIRMED'
    if seq['image_gap_frames']:status='OBSERVATION_GAP_NOT_CERTIFIED'
    evidence=[]
    for s in base['stages']:
        evidence.append(dict(guide_id=s['guide_id'],original_stage_status=s['status'],frame_ids=s['frame_ids'],
          description=s['description'],source_phrases=s.get('source_phrases',[]),source_phrase_scope=s.get('source_phrase_scope'),
          start_seconds=s['start_seconds'],end_seconds=s['end_seconds'],features=s['features'],relations=s.get('relations',[]),
          movements=[v for v in motions if v['guide_id']==s['guide_id']],
          source='Existing sourced guide binding and one learned-prototype DTW. No new sentence-to-video truth is assumed.'))
    return dict(status=status,sequence_status=seq['status'],movement_confirmed=motion_confirmed,motions=motions,
      guide_time_evidence=evidence,saved_movement_source=path,saved_movement_sha256=sha(path),
      base_feature_comparison_preserved=True,base_feature_issues=[s['guide_id'] for s in base['stages'] if s['status']!='SUPPORTED_FEATURES_SIMILAR'],
      combined_supported=order and motion_confirmed and all(s['status']=='SUPPORTED_FEATURES_SIMILAR' for s in base['stages']),
      full_native_frame_range=seq['comparison_frame_range'],whole_sign_correctness=None,
      limitation='Supported measured components only; not exact contact, thumb, required repetition count or all linguistic correctness.')

def main():
    assert read(B/'PROGRESS.json')['status']=='COMPLETE';assert not (B/'EVALUATION.json').exists();torch.set_num_threads(2)
    plan=read(B/'FROZEN.json');s=read(B/'NORMALIZATION.json');rows=add_rgb(old.inputs(False));graphs=read(S/'LEARNED_STATE_GRAPH.json')['graphs']
    exclusive(B/'EVALUATION_PROTOCOL.json',dict(at=clock(),primary=plan['primary'],conditions=plan['conditions'],seeds=plan['seeds'],cases=88,code_sha256=sha(B/'evaluate.py'),behavior=behavior(),old_state_and_motion_reinference=0,gates=plan['gate']))
    pins={r['input']:r['input_sha256'] for r in rows}
    for r in rows:pins.update(dict(r['extra_pins']));pins[r['base_path']]=r['base_sha256']
    exclusive(B/'EVALUATION_INPUTS.json',dict(pins=pins,rows=[{k:v for k,v in r.items() if k not in ['x','y','z','m','current','available','times','rgb','physical']} for r in rows]))
    reports=[]
    for kind in plan['conditions']:
        for seed in plan['seeds']:
            tag=f'{kind}_{seed}';m=ProcessNet();p=B/'runs'/tag/'model.pt';assert sha(p)==read(p.parent/'COMPLETE.json')['sha256'];m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));m.eval()
            roles=[];physical=[];pairs=[];seqs=[];secs=[];feedback=[]
            for r in rows:
                x,im,l,w,y,py=batch([r],s);t=time.perf_counter()
                with torch.inference_mode():
                    pr,ph,h=m(x,im,l,w);pl,yy=m.pairs(h,[r]);pr=pr.softmax(-1)[0].numpy();ph=ph.softmax(-1)[0].numpy()
                secs.append(time.perf_counter()-t);roles.append(pr);physical.append(ph)
                if len(yy):pairs.extend([dict(key=r['key'],kind=r['kind'],y=int(v),pred=int(q>=.5),probability=float(q)) for v,q in zip(yy,pl.sigmoid())])
                sq=dict(key=r['key'],word=r['word_id'],kind=r['kind'],**observe(r['current'],pr,r['available'],graphs[r['word_id']],r['times']));seqs.append(sq)
                fb=compose(r,sq);fb.update(key=r['key'],kind=r['kind'],word=r['word_id']);feedback.append(fb);pins[fb['saved_movement_source']]=fb['saved_movement_sha256']
                d=B/'predictions'/tag/f"{r['key']}.npz";d.parent.mkdir(parents=True,exist_ok=True);assert not d.exists();np.savez_compressed(d,role_probability=pr,auxiliary_physical_probability=ph,current_probability=r['current'],image_available=r['available'],frame_id=np.arange(len(pr)))
            points=point_records(rows,roles);ap=point_records(rows,physical,'physical')
            metrics={k:state.metrics([p for p in points if p['kind']==k]) for k in sorted({r['kind'] for r in rows})}
            report=dict(condition=tag,metrics=metrics,points=points,auxiliary_physical_points=ap,physical_metrics=state.metrics(ap),pair_points=pairs,pair_metrics=state.metrics(pairs),sequences=seqs,feedback=feedback,head_median_seconds=float(np.median(secs)))
            exclusive(B/'evaluations'/f'{tag}.json',report);reports.append(dict(condition=tag,metrics=metrics,head_median_seconds=report['head_median_seconds']))
            print(tag,'native',metrics['native']['correct'],'/',metrics['native']['known'],'order',sum(v['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED' for v in seqs if v['kind'] in NORMAL),flush=True)
    for p,h in pins.items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(B/'EVALUATION.json',dict(at=clock(),reports=reports,pins=pins,cases=88,comparisons=528,whole_goal_complete=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (B/'EVAL_FIRST_FAILURE.txt').exists():(B/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
