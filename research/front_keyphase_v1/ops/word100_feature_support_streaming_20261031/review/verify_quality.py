"""Independent saved-array arithmetic; no quality/learning/torch imports or inference."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import collections
import numpy as np

U=Path(__file__).resolve().parent.parent
OLD=U.parent/'word100_stage_feedback_cycle_20261031'
MOTION_NAMES=['hand_wrist_x','hand_wrist_y','pose_wrist_x','pose_wrist_y','elbow_x','elbow_y','palm_center_x','palm_center_y','elbow_angle','wrist_velocity_x','wrist_velocity_y']
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def average(values):
    values=list(values)
    return math.fsum(values)/len(values) if values else None

class Audit:
    def __init__(self):self.count=0;self.failures=[]
    def check(self,truth,name,detail=None):
        self.count+=1
        if not bool(truth):self.failures.append(dict(check=name,detail=detail))
    def equal(self,actual,expected,name,tol=1e-11):
        if isinstance(expected,dict):
            self.check(isinstance(actual,dict),name+'/dict')
            if not isinstance(actual,dict):return
            for k,v in expected.items():
                self.check(k in actual,name+'/'+k+'/present')
                if k in actual:self.equal(actual[k],v,name+'/'+k,tol)
        elif isinstance(expected,(list,tuple)):
            self.check(isinstance(actual,(list,tuple)) and len(actual)==len(expected),name+'/length')
            if isinstance(actual,(list,tuple)):
                for i,(av,ev) in enumerate(zip(actual,expected)):self.equal(av,ev,name+'/'+str(i),tol)
        elif isinstance(expected,float):
            self.check(isinstance(actual,(float,int)) and math.isfinite(actual) and math.isclose(actual,expected,abs_tol=tol,rel_tol=tol),name,dict(actual=actual,expected=expected))
        else:self.check(actual==expected,name,dict(actual=actual,expected=expected))

def metric(prob,target,mask):
    assert prob.shape==target.shape==mask.shape and mask.dtype==np.bool_
    assert np.isfinite(prob[mask]).all() and np.isin(target[mask],[0,1]).all()
    assert ((prob[mask]>=0)&(prob[mask]<=1)).all()
    features=[]
    for j in range(target.shape[1]):
        counts=dict(tp=0,tn=0,fp=0,fn=0)
        for i in np.flatnonzero(mask[:,j]):
            actual=int(target[i,j]);predicted=int(float(prob[i,j])>=.5)
            counts[('tp' if predicted else 'fn') if actual else ('fp' if predicted else 'tn')]+=1
        positive=counts['tp']+counts['fn'];negative=counts['tn']+counts['fp'];n=positive+negative
        recalls=[]
        if positive:recalls.append(counts['tp']/positive)
        if negative:recalls.append(counts['tn']/negative)
        features.append(dict(known=n,**counts,balanced_accuracy=average(recalls),accuracy=(counts['tp']+counts['tn'])/n if n else None))
    return dict(known=sum(f['known'] for f in features),targets=features,macro_balanced_accuracy=average(f['balanced_accuracy'] for f in features if f['balanced_accuracy'] is not None))

def main(stage):
    a=Audit();proto=read(U/'QUALITY_PROTOCOL.json');old_info=read(OLD/'LEARNING_INPUTS.json');ready=read(U/'QUALITY_READY.json')
    all_targets=old_info['all_targets'];targets=old_info['targets'];training_rows=[r for r in old_info['rows'] if r['role'] in ['fit','development']]
    fresh_selection=read(U/'fresh/SELECTION.json');fresh_inputs=read(U/'fresh/INPUTS.json')
    result=dict(reviewer='Codex rgb_labels_b, separate from quality implementation',model_inference_calls=0,torch_or_quality_imported=False,old_held32_numeric_files_read=False,stage=stage,code_sha256=sha(U/'quality.py'))
    protected=dict(proto['protected']);source_hashes={str(U/'QUALITY_PROTOCOL.json'):sha(U/'QUALITY_PROTOCOL.json'),str(U/'NORMALIZATION.json'):sha(U/'NORMALIZATION.json')}
    for p,h in proto['pins'].items():a.equal(sha(p),h,'protocol_pin/'+Path(p).name)
    for p,h in protected.items():a.equal(sha(p),h,'protected/'+Path(p).name)
    a.equal(proto['seeds'],[42,43,44],'protocol/seeds');a.equal(proto['epochs'],160,'protocol/epochs');a.equal(proto['selected_epoch'],160,'protocol/selected_epoch');a.equal(proto['threshold'],.5,'protocol/threshold')
    a.equal(proto['targets'],targets,'protocol/targets')
    fit_rows=[r for r in training_rows if r['role']=='fit'];fit_ids=[r['clip_id'] for r in fit_rows]
    a.equal([r['alias'] for r in fit_rows],proto['training_fit'],'split/fit_aliases')
    a.check(len(training_rows)==48 and len(fit_rows)==36 and len(set(fit_ids))==36,'split/48_fitdev36fit')
    a.equal(sorted({r['person'] for r in fit_rows}),[1,2,3],'split/fit_people')
    a.equal(sorted({r['person'] for r in training_rows if r['role']=='development'}),[4],'split/dev_people')
    a.equal(sorted({r['person'] for r in fresh_inputs['rows']}),[7,8],'split/fresh_people')
    a.check(len(fresh_inputs['rows'])==32 and len({r['clip_id'] for r in fresh_inputs['rows']})==32,'split/fresh32_unique')
    a.check(set(r['clip_id'] for r in old_info['rows']).isdisjoint(r['clip_id'] for r in fresh_inputs['rows']),'split/fresh_disjoint_all_prior80_metadata')
    selected_rows={r['alias']:r for r in fresh_selection['rows']}
    for r in fresh_inputs['rows']:
        a.equal({key:r[key] for key in ['alias','clip_id','person','word_id','role']},{key:selected_rows[r['alias']][key] for key in ['alias','clip_id','person','word_id','role']},'split/frozen_selection/'+r['alias'])
    fit_words={r['word_id'] for r in fit_rows}
    a.check(all((r['word_id'] in fit_words)==(r['role']=='fresh_known') for r in fresh_inputs['rows']),'split/fresh_known_vs_expression')
    a.equal(collections.Counter(r['role'] for r in fresh_inputs['rows']),{'fresh_known':24,'fresh_expression':8},'split/fresh_role_counts')

    # Independent input/label reconstruction, including the midpoint convention.
    train_arrays=[];train_labels=[];train_centers=[];values=[];masks=[]
    shape_counts=np.zeros((2,2),int);motion_counts=np.zeros((2,4),int)
    old_visual={r['alias']:r for r in read(OLD/'VISUAL_INDEX.json')['rows']}
    def load_rows(rows,fresh=False):
        visual={r['alias']:r for r in read((U/'fresh' if fresh else OLD)/'VISUAL_INDEX.json')['rows']}
        arrays=[];labels=[];centers=[];lengths=[]
        for r in rows:
            a.equal(sha(r['input']),r['input_sha256'],'input/'+r['alias']+'/sha')
            with np.load(r['input'],allow_pickle=False) as z:data={k:z[k] for k in ['x','mask','frame_id','timestamp_us']}
            target_path=U/'fresh/targets'/f"{r['alias']}.npz" if fresh else Path(r['targets_file'])
            if not fresh:a.equal(sha(target_path),r['targets_sha256'],'input/'+r['alias']+'/target_sha')
            with np.load(target_path,allow_pickle=False) as z:yy,mm=z['y'],z['target_mask']
            a.check(yy.shape==mm.shape==(len(data['x']),len(all_targets)) and mm.dtype==np.bool_,'input/'+r['alias']+'/target_shape')
            a.check(np.isin(yy[mm],[0,1]).all(),'input/'+r['alias']+'/binary_known')
            frame=visual[r['alias']]['frames'];clip_centers=[];clip_lengths=[]
            for first,last in [(0,3),(3,6),(6,9),(9,12)]:
                low,high=frame[first]['frame_id'],frame[last]['frame_id']
                ix=np.flatnonzero((data['frame_id']>=low)&(data['frame_id']<=high))
                a.check(len(ix)==high-low+1 and np.array_equal(data['frame_id'][ix],np.arange(low,high+1)),'interval/'+r['alias']+'/all_original_frames')
                label_time=(frame[first]['seconds']+frame[last]['seconds'])/2
                label_center=int(np.abs(data['timestamp_us']/1e6-label_time).argmin())
                tensor_time=(float(data['timestamp_us'][ix[0]]/1e6)+float(data['timestamp_us'][ix[-1]]/1e6))/2
                tensor_center=int(np.abs(data['timestamp_us']/1e6-tensor_time).argmin())
                a.equal(tensor_center,label_center,'interval/'+r['alias']+'/frozen_float_midpoint')
                clip_centers.append(tensor_center);clip_lengths.append(len(ix))
            motion_cols=[j for j,name in enumerate(all_targets) if '_motion_' in name]
            a.check(set(np.flatnonzero(mm[:,motion_cols].any(1))).issubset(clip_centers),'interval/'+r['alias']+'/all_known_centers_covered')
            arrays.append(data);labels.append((yy,mm));centers.append(clip_centers);lengths.extend(clip_lengths)
        return arrays,labels,centers,lengths
    train_arrays,train_labels,train_centers,train_lengths=load_rows(training_rows)
    for r,data,(yy,mm),centers in zip(training_rows,train_arrays,train_labels,train_centers):
        if r['role']!='fit':continue
        for side in ['R','L']:
            cols=[r['feature_names'].index(side+'_'+name) for name in MOTION_NAMES]
            values.append(data['x'][:,cols]);masks.append(data['mask'][:,cols])
            for j,name in enumerate(['index_only','open_four']):
                col=all_targets.index(side+'_'+name)
                for value in [0,1]:shape_counts[value,j]+=int(((yy[:,col]==value)&mm[:,col]).sum())
            for mid in centers:
                for j,name in enumerate(['up','down','left','right']):
                    col=all_targets.index(side+'_motion_'+name)
                    if mm[mid,col]:motion_counts[int(yy[mid,col]),j]+=1
    data_x=np.concatenate(values).astype(np.float64);data_mask=np.concatenate(masks);stats=read(U/'NORMALIZATION.json')
    expected_mean=[float(data_x[data_mask[:,j],j].mean()) if data_mask[:,j].any() else 0. for j in range(11)]
    expected_std=[max(float(data_x[data_mask[:,j],j].std()),.001) if data_mask[:,j].any() else 1. for j in range(11)]
    a.equal(stats['motion_mean'],expected_mean,'normalization/fit_bilateral_mean',tol=2e-5);a.equal(stats['motion_std'],expected_std,'normalization/fit_bilateral_std',tol=2e-5)
    a.equal(ready['shape_counts'],shape_counts.astype(float).tolist(),'counts/shape');a.equal(ready['motion_counts'],motion_counts.astype(float).tolist(),'counts/motion')
    code=(U/'quality.py').read_text(encoding='utf-8-sig');ast.parse(code)
    a.check('pack_padded_sequence' in code and 'nn.GRU(26,24' in code,'code/full_frame_packed_gru')
    a.check('mask[0,-2:]=False' in code,'code/first_velocity_masked')
    a.check("candidate_availability means model output exists, NOT actual observation confidence" in code,'code/availability_not_observed')
    result['input_audit']=dict(fit_clips=36,development_clips=12,fresh_selected_clips=32,fitdev_intervals=len(train_lengths),interval_frame_lengths=[min(train_lengths),max(train_lengths)],shape_counts=shape_counts.tolist(),motion_counts=motion_counts.tolist())

    training=read(U/'TRAINING.json');a.equal(training['status'],'COMPLETE','training/status');a.equal(training['backbone_updates'],0,'training/backbone_updates');a.equal(training['old_model_updates'],0,'training/old_model_updates')
    a.equal([r['seed'] for r in training['runs']],[42,43,44],'training/seeds')
    for seed in proto['seeds']:
        directory=U/'runs'/str(seed);run=read(directory/'RUN.json')
        a.equal(run['epochs'],160,'training/epochs');a.equal(run['selected_epoch'],160,'training/selected_epoch')
        a.equal(run['parameters'],9990,'training/parameters');a.equal(run['shape_parameters'],6146,'training/shape_parameters');a.equal(run['motion_parameters'],3844,'training/motion_parameters')
        a.equal(sha(directory/'heads.pt'),run['checkpoint_sha256'],'training/new_checkpoint/'+str(seed))
        a.equal(sha(directory/'EXPOSURES.jsonl'),run['exposure_sha256'],'training/exposure_hash/'+str(seed))
        records=[json.loads(line) for line in (directory/'EXPOSURES.jsonl').read_text(encoding='utf8').splitlines()]
        a.equal(len(records),160,'training/exposure_epochs/'+str(seed));a.equal(len(run['trace']),160,'training/trace_length')
        for epoch,record in enumerate(records,1):
            a.equal(record['epoch'],epoch,'training/epoch_order');a.equal(record['clip_ids'],fit_ids,'training/exact36_fit_ids')
            a.equal([record['shape_loss'],record['motion_loss']],run['trace'][epoch-1],'training/loss_trace')
            a.check(math.isfinite(record['shape_loss']) and math.isfinite(record['motion_loss']),'training/loss_finite')
        for condition in ['pose_capacity_matched','pose_plus_signrep']:
            previous=OLD/'runs'/f'{condition}_{seed}';run_old=read(previous/'RUN.json')
            a.equal(sha(previous/'head.pt'),run_old['checkpoint_sha256'],'training/old_checkpoint_unchanged/'+condition+'/'+str(seed))

    scores=read(U/(stage.upper()+'_SCORES.json'));freeze=read(U/(stage.upper()+'_EVALUATION_FREEZE.json'))
    for p,h in freeze['pins'].items():a.equal(sha(p),h,'evaluation_pin/'+Path(p).name)
    for seed in proto['seeds']:
        a.check(str(U/'runs'/str(seed)/'heads.pt') in freeze['pins'],'evaluation/new_checkpoint_pinned')
        for condition in ['pose_capacity_matched','pose_plus_signrep']:
            a.check(str(OLD/'runs'/f'{condition}_{seed}'/'head.pt') in freeze['pins'],'evaluation/old_checkpoint_pinned')
    a.check(str(OLD/'NORMALIZATION.json') in freeze['pins'] and str(U/'NORMALIZATION.json') in freeze['pins'],'evaluation/both_normalizations_pinned')
    rows=fresh_inputs['rows'] if stage=='fresh' else training_rows
    arrays,labels,centers,lengths=load_rows(rows,stage=='fresh') if stage=='fresh' else (train_arrays,train_labels,train_centers,train_lengths)
    a.equal(scores['input_aliases'],[r['alias'] for r in rows],'evaluation/input_order');a.equal(scores['targets'],targets,'evaluation/target_order')
    n,T,k=len(rows),max(len(d['x']) for d in arrays),len(targets)
    selected=[all_targets.index(f) for f in targets];y=np.zeros((n,T,k),np.float32);mask=np.zeros((n,T,k),bool);expected_available=np.zeros_like(mask)
    for i,(data,(yy,mm)) in enumerate(zip(arrays,labels)):
        frames=len(data['x']);y[i,:frames]=yy[:,selected];mask[i,:frames]=mm[:,selected]
        for j,name in enumerate(targets):
            if '_motion_' in name:expected_available[i,centers[i],j]=True
            else:expected_available[i,:frames,j]=True
    predictions={}
    for p,h in scores['prediction_pins'].items():a.equal(sha(p),h,'prediction_hash/'+Path(p).name)
    for seed in proto['seeds']:
        availability=np.load(U/'predictions'/f'{stage}_candidate_availability_{seed}.npy',allow_pickle=False)
        a.check(availability.dtype==np.bool_ and np.array_equal(availability,expected_available),'availability/exact_real_frames_and_centers/'+str(seed))
        for condition in proto['comparison']:
            path=U/'predictions'/f'{stage}_{condition}_{seed}.npy';prob=np.load(path,allow_pickle=False)
            a.check(prob.shape==y.shape,'prediction/shape');a.check(np.isfinite(prob[mask]).all(),'prediction/known_finite')
            if condition=='shared_shape_interval_motion':
                a.check(np.isfinite(prob[availability]).all(),'prediction/available_finite')
                for j,name in enumerate(targets):
                    if '_motion_' in name:a.check(np.isnan(prob[:,:,j][~availability[:,:,j]]).all(),'prediction/unavailable_motion_nan')
                a.check(not (mask&~availability).any(),'prediction/every_known_cell_available')
            else:a.check(np.isfinite(prob).all(),'prediction/baseline_all_finite')
            predictions[(condition,seed)]=prob
        unchanged=[targets.index(f) for f in ['R_face_overlap','hands_overlap']]
        a.check(np.array_equal(predictions[('shared_shape_interval_motion',seed)][:,:,unchanged],predictions[('frozen_previous_pose',seed)][:,:,unchanged]),'prediction/unchanged_relation_outputs/'+str(seed))
    measured=[]
    for condition in proto['comparison']:
        for seed in proto['seeds']:
            for role in sorted({r['role'] for r in rows}):
                ids=[i for i,r in enumerate(rows) if r['role']==role]
                metrics=metric(predictions[(condition,seed)][ids].reshape(-1,k),y[ids].reshape(-1,k),mask[ids].reshape(-1,k))
                expected=dict(condition=condition,seed=seed,role=role,clips=len(ids),**metrics)
                actual=[r for r in scores['rows'] if (r['condition'],r['seed'],r['role'])==(condition,seed,role)]
                a.check(len(actual)==1,'metric/unique_row')
                a.equal(actual[0],expected,'metric/'+condition+'/'+str(seed)+'/'+role)
                measured.append(expected)
    summaries=[]
    for condition in proto['comparison']:
        for role in sorted({r['role'] for r in rows}):
            rr=[r for r in measured if r['condition']==condition and r['role']==role];values=[r['macro_balanced_accuracy'] for r in rr]
            expected=dict(condition=condition,role=role,mean=average(values),range=[min(values),max(values)],target_means=[average(r['targets'][j]['balanced_accuracy'] for r in rr) if rr[0]['targets'][j]['known'] else None for j in range(k)])
            actual=[r for r in scores['summary'] if r['condition']==condition and r['role']==role]
            a.check(len(actual)==1,'summary/unique');a.equal(actual[0],expected,'summary/'+condition+'/'+role);summaries.append(expected)
    result['independent_summary']=summaries
    result['metadata_audit_pins']={str(p):sha(p) for p in [OLD/'VISUAL_INDEX.json',U/'fresh/INPUTS.json',U/'fresh/VISUAL_INDEX.json',U/'fresh/SELECTION.json']}
    result['limitations']=['Availability records model output existence, not measured hand observation; feedback still requires an independent measurement mask.','Known annotation-derived intervals are supplied; this is not autonomous stage segmentation.','Shared shape uses thumb angles as input even though target definitions ignore thumbs. Missing first-frame wrist disables displacement rather than seeking a later anchor.','Shape uses centered five-frame context and motion uses the whole interval, so this observer is not a strictly causal streaming output.','The candidate changes architecture, bilateral sharing, supervision granularity and parameter count together; a gain cannot be attributed uniquely to one change.','Old/fresh VISUAL_INDEX and fresh INPUTS metadata were not directly in the training protocol pins; this audit supplements current hashes and verifies selected IDs, frame spans and frozen label midpoint consistency.','Same-Codex physical labels and prior TCN TRAIN sources do not establish expert educational correctness.']
    for p,h in source_hashes.items():a.equal(sha(p),h,'unchanged_during_audit/'+Path(p).name)
    result.update(status='PASS' if not a.failures else 'FAIL',checks=a.count,failures=a.failures,new_checkpoints_verified=3,old_checkpoints_verified=6,exposure_clip_epochs=3*160*36,protected_files_verified=len(protected),verifier_sha256=sha(__file__))
    path=U/'review'/f'QUALITY_VERIFY_{stage.upper()}.json'
    with path.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],checks=a.count,failures=len(a.failures),output=str(path)),ensure_ascii=False))
    return int(bool(a.failures))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['development','fresh']);args=parser.parse_args();raise SystemExit(main(args.stage))
