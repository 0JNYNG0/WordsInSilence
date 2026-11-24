"""Matched public FIT views. Keep hidden shape truth out of extra supervision."""
from pathlib import Path
import sys,copy
L=Path(__file__).resolve().parent;O=L.parent
J=O/'word100_sequence_supervision_audit_20261102';K=O/'word100_coherent_sensor_input_20261102'
DEST=Path('W:/WordsInSilence_derived/word100_seated_fit_transfer_20261102')
sys.path.insert(0,str(J))
from sequence_model import *

def new_row(meta,current=None):
    assert sha(meta['input'])==meta['input_sha256'];a=dict(np.load(meta['input']));ids=a['frame_id'];av=a['image_available']
    z,mask=process.old.geometric(a['x'],a['mask'],av)
    if current is None:
        obs=process.old.observer()
        with torch.inference_mode():current=obs(torch.from_numpy(a['latent'][None]),torch.tensor([len(ids)]),torch.tensor([process.WORDS.index(meta['word_id'])])).softmax(-1)[0].numpy()
    return dict(meta,x=a['latent'],z=z,m=mask,rgb=a['rgb'],current=current,available=av,mapping=ids,times=a['timestamp_us']/1e6,
        y=process.old.target(meta['clip_id'],ids),physical=process.state.labels_for(meta['clip_id'],ids))

def role_targets(row):
    target=next(r for r in read(E/'CONTEXT_TARGETS.json')['rows'] if r['clip_id']==row['clip_id'])
    lookup={int(p['frame']):int(p['action_family']) for p in target['points'] if p['mask']}
    yy=np.asarray([lookup.get(int(f),-1) for f in row['mapping']]);return np.where(yy<0,-1,np.where(yy<8,1,np.where(yy==8,2,0))).astype('int64')

def load_pairs():
    original=fit_rows();manifest=read(L/'PAIR_MANIFEST.json');extras={}
    for condition in ['original_replay','seated_variation']:
        out=[]
        for item in manifest['conditions'][condition]:
            assert sha(item['cache'])==item['cache_sha256'];r=dict(item);r.update(dict(np.load(item['cache'])));out.append(r)
        extras[condition]=out
    assert len(original)==216 and all(len(v)==36 for v in extras.values())
    for a,b in zip(extras['original_replay'],extras['seated_variation']):
        assert a['clip_id']==b['clip_id']
        for key in ['episode','role','mapping','available']:assert np.array_equal(a[key],b[key]),key
    return original,extras

def main():
    assert not (L/'PAIR_MANIFEST.json').exists();torch.set_num_threads(2)
    meta=read(L/'INPUT_MANIFEST.json')['rows'];crop={r['clip_id']:r for r in read(L/'CROP_MANIFEST.json')['rows']}
    originals={r['clip_id']:r for r in fit_rows() if r['kind']=='original'};assert len(meta)==len(originals)==36
    targets=episode.target_map();conditions={'original_replay':[],'seated_variation':[]};audit=[]
    for mm in meta:
        assert mm['person'] in range(1,10) and not crop[mm['clip_id']]['cropped_action_evidence']
        b=new_row(mm);episode.attach_targets([b],targets);b['role']=role_targets(b)
        a=copy.deepcopy(originals[mm['clip_id']]);assert np.array_equal(a['mapping'],b['mapping'])
        # No additional hidden-hand rest/shape supervision. Process role is
        # source-confirmed nonperformance, not an assertion about hidden fingers.
        for r in [a,b]:r['episode']=np.where(r['episode']==8,-1,r['episode']);r['physical'][:]=-1
        for key in ['episode','role','available']:assert np.array_equal(a[key],b[key]),(mm['clip_id'],key)
        source=np.load(mm['input']);assert np.max(np.abs(source['timestamp_us']-np.asarray(crop[mm['clip_id']]['source_timestamps_us'])))<=2
        main=(b['episode']>=0)&(b['episode']<8)
        audit.append(dict(clip_id=mm['clip_id'],main_points=int(main.sum()),main_rest_conflicts=int(((b['current'][:,9]>=.8)&main).sum()),
            input_sha256=mm['input_sha256'],source_sha256=mm['source_sha256'],frames=len(b['x']),mask_valid_fraction=float(b['m'].mean()),
            no_visible_rest_shape_label=True,main_evidence_crop_risk_count=0))
        for tag,r in [('original_replay',a),('seated_variation',b)]:
            dest=DEST/'prepared'/tag/f"{mm['clip_id']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            arrays={k:r[k] for k in ['x','z','m','rgb','current','available','mapping','y','physical','episode','role']}
            np.savez_compressed(dest,**arrays)
            conditions[tag].append(dict(clip_id=mm['clip_id'],word_id=mm['word_id'],person=mm['person'],kind=tag,cache=str(dest),cache_sha256=sha(dest),input=r['input'],input_sha256=r['input_sha256']))
    assert sum(a['main_points'] for a in audit)==706
    exclusive(L/'PAIR_MANIFEST.json',dict(at=clock(),conditions=conditions,original_replay_rows=216,additional_rows_per_condition=36,matched_supervision=True,whole_goal_complete=False))
    exclusive(L/'INPUT_TARGET_AUDIT.json',dict(at=clock(),status='PASS_FOR_PAIRED_PUBLIC_FIT_EXPERIMENT',rows=audit,major_points=706,role_scope='source-reviewed main versus nonperforming only; no hidden finger truth',
        visual_scope='Codex reviewed9contact sheets of4words x6selectedframes. Important hands/arm configurations retained. Selected-frame camera-view audit, not all-frame finger annotation or independent KSL correctness.',
        native_time=True,retained_completed_inputs=19,resumed_inputs=17,independent_expert_truth=False,whole_goal_complete=False))
    print('pairs72 ready; same706extra major witnesses; rest conflict',sum(a['main_rest_conflicts'] for a in audit),flush=True)
if __name__=='__main__':main()
