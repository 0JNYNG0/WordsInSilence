import time,traceback
from transfer_data import *
from evaluate_witness import observe,behavior,NORMAL,SUPPORTED
import evaluate_episode as previous

def eval_rows():
    # Reuse saved features/current observations. No completed model is inferred again.
    rows=process.add_rgb(process.old.inputs(False));coherent={r['clip_id']:r for r in read(K/'INPUT_MANIFEST.json')['rows']}
    for i,r in enumerate(rows):
        if r['kind']=='native':
            saved=K/'predictions/point_ce_42'/f"{r['clip_id']}.npz"
            changed=new_row(coherent[r['clip_id']],np.load(saved)['current_probability']);r.update(changed)
            r.update(extra_pins=r['extra_pins']+[(str(saved),sha(saved))],input_contract='coherent_full_sensor')
        else:r['input_contract']='existing_live_control'
    targets={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in read(D/'DEV_ACTION_TARGETS.json')['rows']}
    episode.attach_targets(rows,targets)
    for r in rows:r['role']=np.where(r['episode']<0,-1,np.where(r['episode']<8,1,2))
    assert len(rows)==88 and {r['person'] for r in rows}=={11,12,13};return rows

def main():
    assert not (L/'EVALUATION.json').exists();plan=read(L/'TRAINING_FREEZE.json');assert read(L/'TRAIN_PROGRESS.json')['status']=='COMPLETE'
    torch.set_num_threads(2);rows=eval_rows();stats=read(D/'NORMALIZATION.json');gg=previous.graphs();pins={}
    for r in rows:
        pins[r['input']]=r['input_sha256'];pins.update(dict(r['extra_pins']));pins[r['base_path']]=r['base_sha256']
    exclusive(L/'EVALUATION_PROTOCOL.json',dict(at=clock(),cases=88,comparisons=176,pins=pins,behavior=behavior(),graphs=gg,
        scope='12coherent native plus76unchanged controls. Candidate models new inference once. No old model rescore. Native full-feedback not asserted using stale ROI guide scores.',whole_goal_complete=False))
    reports=[]
    for tag in plan['conditions']:
        path=DEST/'runs'/tag/'model.pt';assert sha(path)==read(path.parent/'COMPLETE.json')['sha256']
        m=WitnessNet();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));m.eval()
        points=[];raw=[];roles=[];sequences=[];seconds=[]
        for r in rows:
            x,im,ll,_,_=batch_j([r],stats);start=time.perf_counter()
            with torch.inference_mode():ac,act,_=m(x,im,ll);pr=ac.softmax(-1)[0].numpy();activity=act.softmax(-1)[0].numpy()
            seconds.append(time.perf_counter()-start);sq,effective,ro=observe(pr,activity,r,gg[r['word_id']]);sq.update(key=r['key'],kind=r['kind'],word=r['word_id']);sequences.append(sq)
            for f in np.flatnonzero(r['episode']>=0):
                base=dict(key=r['key'],clip_id=r['clip_id'],person=r['person'],word=r['word_id'],kind=r['kind'],frame=int(f),source_frame=int(r['mapping'][f]))
                points.append(dict(base,y=int(r['episode'][f]),pred=int(effective[f].argmax())))
                raw.append(dict(base,y=int(r['episode'][f]),pred=int(pr[f].argmax())))
                roles.append(dict(base,y=int(r['role'][f]),pred=int(ro[f].argmax())))
            dest=DEST/'predictions'/tag/f"{r['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,probability=effective,raw_action_probability=pr,role_probability=ro,activity_probability=activity,
                current_probability=r['current'],image_available=r['available'],frame_id=np.arange(len(pr)),source_frame_id=r['mapping'],timestamp=r['times'])
        norms=[p for p in points if p['kind'] in NORMAL and p['y']<8]
        report=dict(condition=tag,points=points,raw_points=raw,role_points=roles,sequences=sequences,
            normal_action_metrics=process.state.metrics(norms),raw_normal_action_metrics=process.state.metrics([p for p in raw if p['kind'] in NORMAL and p['y']<8]),
            native_metrics=process.state.metrics([p for p in norms if p['kind']=='native']),seated_metrics=process.state.metrics([p for p in norms if p['kind']!='native']),
            head_median_seconds=float(np.median(seconds)),whole_feedback_certified=False)
        exclusive(DEST/'evaluations'/f'{tag}.json',report);reports.append({k:v for k,v in report.items() if k not in ['points','raw_points','role_points','sequences']})
        print(tag,'major',report['normal_action_metrics']['correct'],'order',sum(s['kind'] in NORMAL and s['status']==SUPPORTED for s in sequences),flush=True)
    for p,h in pins.items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(L/'EVALUATION.json',dict(at=clock(),reports=reports,cases=88,comparisons=176,pins=pins,whole_goal_complete=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (L/'EVAL_FIRST_FAILURE.txt').exists():(L/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
