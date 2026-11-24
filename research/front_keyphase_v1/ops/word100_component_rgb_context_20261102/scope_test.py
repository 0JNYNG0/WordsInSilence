"""New annotation scopes for the frozen RGB observer; never refit old models."""
from pathlib import Path
import sys,json,hashlib,time,traceback
import numpy as np
import torch
T=Path(__file__).resolve().parent;O=T.parent
U=O/'word100_component_learning_20261102';R=O/'word100_component_relative_20261102'
P=O/'word100_learned_guide_feedback_20261102';M=O/'word100_learned_motion_order_20261102'
V=O/'word100_learned_body_relation_20261102';RGB=O/'word100_learned_rgb_motion_20261102'
OLD=O/'word100_multipeople_live_feedback_20261101';S=O/'word100_source_component_connector_20261102'
sys.path.insert(0,str(RGB));sys.path.insert(0,str(M))
from rgb_model import RGBMotionNet,rgb_input
from motion_model import public_arrays,normalize,predict

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
    with (T/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

TARGET='right_hand_traverses_left_arm_toward_wrist'
def records():return [r for r in read(S/'DIRECT_TARGETS.json')['rows'] if r['target']==TARGET and r['kind']=='sequence' and r['mask'] and r['y']==1]

def saved_native_lookup():
    # These old results remain unchanged, including results on other scopes.
    a=np.load(RGB/'runs/rgb_residual_42/development_predictions.npz',allow_pickle=False)
    ds=read(M/'DATASET.json')['rows'];lookup={}
    for k,i in enumerate(a['old_indices']):
        r=ds[int(i)]
        if r['kind'] not in ['native_directed','native_stable']:continue
        key=(r['clip_id'],r['frame_ids'][0],r['frame_ids'][-1])
        lookup[key]=dict(probabilities=a['p'][k].tolist(),source=str(RGB/'runs/rgb_residual_42/development_predictions.npz'),row=k)
    for row in read(RGB/'COMPARISON.json')['rows']:
        if row['condition']!='rgb_residual' or row['seed']!=42 or row['kind']!='native':continue
        for m in row['motions']:
            if m['target']!=0:continue
            key=(row['case'],m['frame_ids'][0],m['frame_ids'][-1])
            v=list(m['all_motion_probabilities'])+[m['continuity_probability']]
            if key in lookup:assert np.allclose(v,lookup[key]['probabilities'],atol=1e-5)
            else:lookup[key]=dict(probabilities=v,source=str(RGB/'COMPARISON.json'),case=row['case'])
    return lookup

def prepare():
    rs=records();assert len(rs)==12 and {r['person'] for r in rs}==set(range(1,10))|{11,12,13}
    assert all(r['mask'] and r['y']==1 for r in rs)
    public={r['clip_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows']};pins={}
    dependencies=[S/'DIRECT_TARGETS.json',P/'INPUT_MANIFEST.json',M/'DATASET.json',M/'NORMALIZATION.json',
      M/'motion_model.py',RGB/'rgb_model.py',RGB/'RGB_NORMALIZATION.json',RGB/'runs/rgb_residual_42/model.pt',
      RGB/'runs/rgb_residual_42/development_predictions.npz',RGB/'COMPARISON.json',
      U/'OLD_EQUIVALENT_BASELINE.json',U/'ORDERED_RESULT.json',R/'RELATIVE_RESULT.json',T/'scope_test.py',
      O/'word100_learned_motion_path_20261102/path_model.py',
      O/'word100_learned_motion_invariance_20261102/TRAINING_CONTRACT.json']
    for r in rs:
        p=public[r['clip_id']];assert p['person']==r['person'] and p['role']==r['role']
        assert sha(p['input'])==p['input_sha256']
        dependencies += [Path(p['input']),OLD/('bank_predictions' if r['role']=='fit' else 'baseline_probabilities')/(r['clip_id']+'.npz'),V/'runs/rgb_geometry_42/predictions'/(r['clip_id']+'.npz')]
    for p in dependencies:pins[str(p)]=sha(p)
    cache=saved_native_lookup()
    manifest=[]
    for r in rs:
        lo,hi=min(r['native_frames']),max(r['native_frames']);cached=cache.get((r['clip_id'],lo,hi))
        manifest.append(dict(record=r,extent=[lo,hi],action='reuse_saved_prediction' if cached else 'new_annotation_scope_inference'))
    save('SCOPE_FROZEN.json',dict(rows=manifest,pins=pins,model='existing rgb_residual_42 target0 unchanged',
      goal='Determine whether new geometry connector lost already learned RGB evidence or failures persist at identical reviewed source-action scopes',
      new_training=False,normalization_updated=False,thresholds_updated=False,new_rgb_extraction=False,
      timing='head-only cached native inputs; not end-to-end latency',
      controls='Existing6 exact native direct intervals reused verbatim; no window scan or best window selection',
      scope='12 native positive sequence windows, FIT9 diagnostic and repeatedly exposed DEV3. Not independent evaluation, negative-control or KSL-correctness proof.',
      unknown_scope='12 additional unknown sequence annotations remain unknown in the source dataset; not converted into negatives or included in this positive-scope diagnostic.',
      decision='If old frozen observer better, preserve same-definition knowledge and focus scope connection. If both fail, investigate retained evidence before deciding coherent RGB training. Never deploy from this positive-only check.',
      private_holdout_or_original_writes=False))
    print('Frozen12positive source windows; cached matches',sum(r['action']=='reuse_saved_prediction' for r in manifest))

def run():
    torch.set_num_threads(4);plan=read(T/'SCOPE_FROZEN.json');assert not (T/'SCOPE_RESULT.json').exists()
    for p,h in plan['pins'].items():assert sha(p)==h,p
    cache=saved_native_lookup();public={r['clip_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows']}
    ms=read(M/'NORMALIZATION.json');zs=read(RGB/'RGB_NORMALIZATION.json')
    model=RGBMotionNet('rgb_residual');model.load_state_dict(torch.load(RGB/'runs/rgb_residual_42/model.pt',map_location='cpu',weights_only=True));model.eval();model.requires_grad_(False)
    out=[]
    for row in plan['rows']:
        r=row['record'];lo,hi=row['extent'];p=public[r['clip_id']]
        a,z,m=public_arrays(p);ids=np.flatnonzero((a['frame_id']>=lo)&(a['frame_id']<=hi))
        assert np.array_equal(a['frame_id'][ids],np.arange(lo,hi+1))
        ts=a['timestamp_us'][ids];assert list(map(int,ts[[0,-1]]))==[min(r['native_timestamps_us']),max(r['native_timestamps_us'])]
        video=rgb_input(a,zs);seq=np.concatenate([normalize(z,m,ms),video],-1)[ids]
        assert seq.shape==(hi-lo+1,915) and np.isfinite(seq).all()
        key=(r['clip_id'],lo,hi);t=time.perf_counter()
        if key in cache:pp=cache[key]['probabilities'];origin=cache[key]
        else:pp=predict(model,[seq])[0].tolist();origin={'new_scope_inference':True}
        elapsed=time.perf_counter()-t;coverage=float(m[ids].mean());im=float((video[ids,-1]>.5).mean())
        usable=coverage>=.5 and im>=.8 and len(ids)>=3
        status='MOTION_OBSERVATION_INSUFFICIENT' if not usable else 'MOTION_CONTINUITY_UNCONFIRMED' if pp[5]<.8 else 'DIRECTED_MOTION_OBSERVED' if pp[0]>=.8 else 'EXPECTED_MOTION_UNCONFIRMED'
        comps={}
        for name,file in [('geometry_ordered',U/'ORDERED_RESULT.json'),('geometry_relative',R/'RELATIVE_RESULT.json')]:
            match=[x for x in read(file)['rows'] if x['variant']=='native' and x['record']==r]
            assert len(match)==1;comps[name]={k:match[0][k] for k in ['probability','class_prediction','status']}
        out.append(dict(record=r,extent=[lo,hi],source_times_us=list(map(int,ts[[0,-1]])),probabilities=pp,
          status=status,geometry_coverage=coverage,image_coverage=im,usable=usable,
          origin=origin,component_head_seconds=elapsed,comparators=comps,whole_sign_correctness=None))
    summary={}
    for role in ['fit','development']:
        rr=[r for r in out if r['record']['role']==role]
        summary[role]=dict(positive_windows=len(rr),rgb_class_positive=sum(r['probabilities'][0]>=.5 for r in rr),
          rgb_confirmed=sum(r['status']=='DIRECTED_MOTION_OBSERVED' for r in rr),
          geometry_ordered_positive=sum(r['comparators']['geometry_ordered']['class_prediction']==1 for r in rr),
          geometry_relative_positive=sum(r['comparators']['geometry_relative']['class_prediction']==1 for r in rr))
    for p,h in plan['pins'].items():assert sha(p)==h,p
    save('SCOPE_RESULT.json',dict(rows=out,summary=summary,new_model_inference_windows=sum('new_scope_inference' in r['origin'] for r in out),
       old_completed_results_unchanged=True,new_training=0,negative_evaluation_not_extended=True,whole_goal_complete=False))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='run':run()
        else:raise ValueError('prepare/run')
    except BaseException:
        p=T/'FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
