"""Pure coordinate transformation of previously frozen public FIT source arrays.

No backbone/hand model forward; labels, folds, windows and controls are copied
exactly. Unexpected invalid labelled/window bases stop instead of changing the
comparison denominator or inventing orientation.
"""
from hand_core import *
from rotation_core import canonicalize_hand63,rotate_hand63,BASIS_EPSILON
import copy

def main():
    assert not(U/'FROZEN.json').exists() and not(U/'INPUTS_COMPLETE.json').exists()
    old=read(ORIGINAL/'INPUTS_COMPLETE.json');cfg=read(ORIGINAL/'FROZEN.json')
    baseline=read(ORIGINAL/'TRAINING_CONTRACT.json')
    assert sha(ORIGINAL/'train.py')==baseline['pins'][str(ORIGINAL/'train.py')]
    assert (U/'train.py').read_bytes()==(ORIGINAL/'train.py').read_bytes()
    for p,h in cfg['pins'].items():assert sha(p)==h,p
    old_labels=copy.deepcopy(old['labels']);rows=[];audit=[];cache={}
    # Freeze transforms and fixed conditions before any new model/control output.
    paths=[Path(__file__),U/'rotation_core.py',U/'hand_core.py',U/'train.py',U/'test_rotation_core.py',
           ORIGINAL/'INPUTS_COMPLETE.json',ORIGINAL/'FROZEN.json',ORIGINAL/'TRAINING_CONTRACT.json',
           ORIGINAL/'hand_core.py',ORIGINAL/'INDEPENDENT_SAVED_REVIEW.json']
    frozen=dict(copy.deepcopy(cfg))
    frozen.update(at=stamp(),scope='Isolated input-basis comparison only. Same original HandObserver, initialization seeds,600steps,6LOPO, losses, labels and thresholds. No allFITmodel or web replacement.',
        basis_transform='Framewise rigid rotation around wrist maps wrist->middleMCP9 to canonical upward y. No length rescaling, no second left reflection, no new semantic labels.',
        basis_validity='Original wrist and middleMCP9 available AND norm>1e-6 numerical threshold. Invalid basis means unknown; never reconstruct it from neighboring frames.',
        normalization='Canonicalized hand-local coordinates then same fold-training-person-only mean/std; availability mean0/std1. Left anatomical reflection was already applied in original preparedx.',
        previous_results='Original hand_transition_learning saved18fold outputs only; no rerun of its completed original evaluation.',
        limitation='REAL05failure informed development. This is not a new independent test. Image-plane invariance cannot remove out-of-plane projection or inaccurate landmarks.',
        approval='Root delegated pure helper/tests and preparation; training requires root review and explicit launch.',
        pins=dict(cfg['pins'],**{str(p):sha(p)for p in paths}))
    write(U/'FROZEN.json',frozen)
    for r in old['sequences']:
        assert sha(r['path'])==r['sha256'];z=np.load(r['path'],allow_pickle=False)
        x,valid,info=canonicalize_hand63(z['x'])
        assert np.array_equal(x[:,42:],z['x'][:,42:])
        # No current source has invalid basis; fail instead of changing labels.
        assert valid.all(),('Unexpected basis exclusion; root must review denominator before training',r['clip_id'],r['side'])
        assert np.isfinite(x).all()
        for angle in [-90,-60,-30,30,60,90]:
            y,yv,_=canonicalize_hand63(rotate_hand63(z['x'],angle))
            assert np.array_equal(valid,yv)
            assert np.allclose(x,y,atol=2e-6,rtol=1e-6),('Rigid rotation math failed',r['clip_id'],angle)
        p=U/'sources'/Path(r['path']).name;p.parent.mkdir(exist_ok=True)
        assert not p.exists()
        payload={k:z[k].copy()for k in z.files};payload['x']=x
        payload['basis_valid']=valid;payload['basis_length']=info['basis_length'];payload['original_basis_unit']=info['original_basis_unit']
        np.savez_compressed(p,**payload)
        rows.append(dict(r,path=str(p),sha256=sha(p),original_path=r['path'],original_sha256=r['sha256']))
        cache[r['clip_id'],r['side']]=valid
        audit.append(dict(clip_id=r['clip_id'],person=r['person'],side=r['side'],source_frames=len(x),basis_valid_frames=int(valid.sum()),
                          unavailable_basis=int((~valid).sum()),minimum_basis_length=float(info['basis_length'].min()),
                          label_cells_unchanged=int(z['mask'].sum()),original_mask_preserved=True))
    for r in old['windows']+old['controls']:
        a,b=r['span'];assert cache[r['clip_id'],r['side']][a:b+1].all()
    new=copy.deepcopy(old);new['sequences']=rows;new['at']=stamp();new['source_pins']=frozen['pins']
    new['input_basis']='wrist_to_MCP9_canonical_imageplane_rotation'
    new['basis_audit']=audit;new['parent_manifest']=str(ORIGINAL/'INPUTS_COMPLETE.json')
    new['parent_manifest_sha256']=sha(ORIGINAL/'INPUTS_COMPLETE.json')
    new['raw_frame_observation_audit_note']='Inherited RTMW measurements before canonical rotation, not newly measured shape truth.'
    assert new['labels']==old_labels and new['windows']==old['windows'] and new['controls']==old['controls'] and new['counts']==old['counts']
    write(U/'INPUTS_COMPLETE.json',new)
    write(U/'PREPARATION_VERIFICATION.json',dict(at=stamp(),pass_=True,
        source_hand_sequences=len(rows),source_frames=sum(r['source_frames']for r in audit),invalid_basis=sum(r['unavailable_basis']for r in audit),
        same_labels=True,same_folds=True,same_windows=True,same_controls=True,same_training_code_bytes=True,
        canonical_rotation_math_checked_all_source_frames_at_six_angles=True,
        architecture='Imported identical original HandObserver class; only dataset coordinate basis differs.',
        new_model_inference=False,training_executed=False,no_private=True,no_final8=True,
        pins={str(p):sha(p)for p in[U/'FROZEN.json',U/'INPUTS_COMPLETE.json',U/'train.py',U/'rotation_core.py',U/'hand_core.py']}))
    print('Prepared rotation-only comparison',new['counts'],flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:preserve_failure('PREP');raise
