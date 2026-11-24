"""One fixed U/W public comparison. Evidence evaluation is outside the matcher."""
from pathlib import Path
import importlib.util, sys, time, traceback, json, numpy as np
from prepare_pair import U, read, write, load, sha
import motion_cost, natural_evaluator
spec=importlib.util.spec_from_file_location('readonly_previous_evaluators',U.parent/'word100_phase_correspondence_20261101/run.py')
old=importlib.util.module_from_spec(spec)
sys.path.append(str(U.parent/'word100_phase_correspondence_20261101'));spec.loader.exec_module(old)

def paired(q,r,meta=None,kind=None):
    before=old.input_digest(q,r)
    for a in (q,r):
        for v in a.values():v.setflags(write=False)
    out={};details={}
    for mode in ['unit','bounded']:
        t=time.perf_counter();cost=motion_cost.cost(q,r,mode);d=motion_cost.dtw(cost,return_details=True)
        d['seconds']=time.perf_counter()-t
        d['evidence'],qd,rd,qs,rs=old.evidence(q,r,d['path'])
        if meta is not None:
            d['truth']=old.truth_metrics(d['path'],meta,kind,len(q['x']),len(r['x']))
            if d['truth']:d['truth']['counterfactual_feature_edit']=kind not in ['identity','speed_075','speed_125','hold15','repeat2_same','repeat3_same','early_slow','late_slow']
        out[mode]=d;details[mode]=(qd,rd,qs,rs)
    assert before==old.input_digest(q,r)
    out['input_digests_before_after']=before
    out['sidecar_support']={axis:{side:{'total':len(h['valid']),'observed':int(h['valid'].sum()),'zero_observed':int(h['observed_zero'].sum())} for side,h in motion_cost.velocity(a)['hands'].items()} for axis,a in [('query',q),('reference',r)]}
    if kind in ['left_short_shape_change_5f','short_shape_change_5f']:
        frames=meta.get('changed_frames',[]);cols=slice(11,25) if kind.startswith('left') else slice(39,53)
        known=q['mask'][frames,cols];a,b=details['unit'],details['bounded']
        out['short_shape_guard']=dict(frames=frames,actual_observed_changed_cells=meta.get('actual_observed_changed_cells',0),
            supported_cells_unit=int((known&a[2][frames,cols]).sum()),supported_cells_bounded=int((known&b[2][frames,cols]).sum()),
            lost_supported_cells=int((known&a[2][frames,cols]&~b[2][frames,cols]).sum()),
            query_frame_peaks_unit=a[0][frames,cols].max(axis=1).tolist(),query_frame_peaks_bounded=b[0][frames,cols].max(axis=1).tolist())
    return out

def main():
    f=read(U/'FROZEN_PAIR.json');inp=read(U/'PAIR_INPUTS.json')
    for p,h in {**f['pins'],**inp['pins']}.items():assert sha(p)==h,('changed',p)
    assert not (U/'PAIR_MANIFEST.json').exists() and not (U/'PAIR_RUNNING.json').exists()
    write(U/'PAIR_RUNNING.json',dict(start=time.time(),no_pid_only_unlock=True))
    manifest={'frozen_sha256':sha(U/'FROZEN_PAIR.json'),'controls':[],'natural':[]}
    def save(name,d):
        p=U/'pair_results'/f'{name}.json';write(p,d);return dict(file=str(p),sha256=sha(p),id=name)
    try:
        for c in inp['controls']:
            d=paired(load(c['input']),load(c['reference']),c['metadata'],c['kind'])
            d.update(id=c['id'],kind=c['kind'],normal_control=c['normal_control'],input=c['input'],reference_input=c['reference'])
            manifest['controls'].append(save(c['id'],d))
            if len(manifest['controls'])%16==0:print('controls',len(manifest['controls']),flush=True)
        for p in inp['natural']:
            a,b=p['query'],p['reference'];d=paired(load(inp['inputs'][a]['original']),load(inp['inputs'][b]['original']))
            d.update(query=a,reference=b)
            for mode in ['unit','bounded']:d[mode]['natural_evaluation']=natural_evaluator.evaluate(a,b,d[mode]['path'])
            manifest['natural'].append(save(f'{a}_{b}',d));print('natural',a,b,flush=True)
        write(U/'PAIR_MANIFEST.json',manifest)
        (U/'PAIR_RUNNING.json').unlink()
        (U/'CURRENT_EXECUTION.json').write_text(json.dumps(dict(status='PAIR_COMPLETE_AWAITING_VERIFICATION',active_session=None,new_inference=0,new_training=0,new_paths=408),indent=2),encoding='utf-8')
        print('COMPLETE 192 controls / 12 natural / 408 new paths',flush=True)
    except Exception:
        p=U/'FIRST_PAIR_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
if __name__=='__main__':main()
