from common import *
import adapter as A
from collections import Counter
import traceback,os

def freeze():
    pre=read(U/'PREFLIGHT.json');check(pre['pins']);vf=read(V/'FROZEN.json');check(vf['protected'])
    ci=read(V/'CLASSIFIER_INPUTS.json');comparisons=read(R/'COMPARE_RESULT.json')
    paths=[U/'PREFLIGHT.json',V/'FROZEN.json',V/'INPUTS_VERIFIED.json',V/'CLASSIFIER_INPUTS.json',V/'CLASSIFIER_RESULT.json',V/'CLASSIFIER_VERIFIED.json',
        R/'FROZEN.json',R/'COMPARE_RESULT.json',R/'VERIFIED.json',R/'adapter.py',O/'word100_finger_evidence_20261020/candidate.py',
        O/'word100_finger_evidence_20261020/verify.py',O/'word100_angle_evidence_20261021/candidate.py']+list(U.glob('*.py'))
    paths+=list(V.glob('*_SCORES.json'))
    for row in vf['rows']:
        paths+=[V/'outputs'/(row['clip_id']+'.npz'),V/'records'/(row['clip_id']+'.json'),Path(row['record'])]
        paths+=[Path(read(row['record'])['views']['masked']['channel_file'])]
    paths +=[V/'classifier_inputs'/(row['arrays']['native']+'.npz') for row in ci['rows']]
    paths +=[R/'comparisons'/(p['name']+'.npz') for p in comparisons['rows']]
    protected=vf['protected']
    write(U/'FROZEN.json',dict(at=now(),rows=vf['rows'],comparison_rows=comparisons['rows'],protected=protected,pins={str(p.resolve()):sha(p) for p in paths},
        implementation='Codex',verification='Separate Codex arithmetic; not independent human truth',
        contract=['No detector/model forward, training or re-alignment. Original classifier arrays and all saved scores reused.',
        'Use previous channel_view and independent local-angle evidence. Original raw arrays are never deleted or reassigned.',
        'Second prediction only from existing saved accepted retry/YOLO crop, or unique confirm_existing_hand full prediction. Unchecked old hands remain computable but unassessed.',
        'Identity state0 means not rechecked,1 saved confirmation,2 accepted current-arm source,3 existing_identity_conflict or unresolved_existing_overlap involving present side. This is uncertainty, not incorrectness. Hand comparison eligibility held at state3; independent elbow/pose wrist remain.',
        'Frozen reference domain remains unchanged. Observation absence cannot change it. Missing differences are NaN internally/null in receipt, never zero. No whole-action correctness or correction instruction.',
        'Public DEV300 observation receipts; public TRAIN126 existing comparison receipts only. No extra199/200/user/mentor access. No operational adoption.']))
    print('FROZEN',len(vf['rows']),len(comparisons['rows']),flush=True)

def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected'])
    write(U/'RUNNING.json',dict(at=now(),pid=os.getpid()))
    for directory in ('inputs','receipts','comparison_receipts'):(U/directory).mkdir()
    B=module('existing_bridge_adapter',R/'adapter.py');Q=module('shape_quality',O/'word100_finger_evidence_20261020/candidate.py')
    E=module('local_angles',O/'word100_angle_evidence_20261021/candidate.py')
    ci={r['clip_id']:r for r in read(V/'CLASSIFIER_INPUTS.json')['rows']}
    scores={p.stem.removesuffix('_SCORES'):{r['clip_id']:r for r in read(p)['predictions']['native']} for p in V.glob('*_SCORES.json')}
    counts=Counter();rows=[];pins={};groups={g:dict(computable=[0,0,0],eligible=[0,0,0]) for g in A.GROUPS}
    states=np.zeros((3,5),np.int64)
    for ordinal,row in enumerate(f['rows']):
        cid=row['clip_id'];p=load(V/'outputs'/(cid+'.npz'));rr=read(V/'records'/(cid+'.json'))['frames'];z=A.build(p,rr,B,Q,E)
        out=U/'inputs'/(cid+'.npz');np.savez_compressed(out,**z);pins[str(out)]=sha(out)
        classifier=V/'classifier_inputs'/(ci[cid]['arrays']['native']+'.npz');old=Path(read(row['record'])['views']['masked']['channel_file'])
        x=load(classifier)['x'];ox=load(old)['x_whole'];assert same(x,ox) and x.dtype==ox.dtype
        prediction={model:{k:r[cid][k] for k in ('prediction','top5','correct','top5_correct','status')} for model,r in scores.items()}
        receipt=dict(clip_id=cid,word_code=row['word_code'],signer_id=row['signer_id'],label_index=row['label_index'],
            classifier_input=dict(path=str(classifier),sha256=sha(classifier),source=str(old),bytes_unchanged=True),
            recognition=dict(saved_original_predictions=prediction,source=str(V),new_inference=False),
            feedback_observation=A.observation_receipt(z),raw_observation_file=str(out),geometry_is_ground_truth=False)
        rp=U/'receipts'/(cid+'.json');write(rp,receipt);pins[str(rp)]=sha(rp)
        n=len(z['frame_id']);counts.update(clips=1,frames=n,original_classifier_cells=x.size,
            identity_disagreement_hand_frames=int(z['identity_conflict'].sum()),wrist_only_yolo=int(z['position_only'].sum()))
        for k in range(3):
            states[k]+=np.bincount(z['angle_state'][k].ravel(),minlength=5)
            for g,(a,b) in A.GROUPS.items():
                groups[g]['computable'][k]+=int(z['available'][k,:,:,a:b].sum());groups[g]['eligible'][k]+=int(z['eligible'][k,:,:,a:b].sum())
        rows.append(dict(clip_id=cid,receipt=str(rp),output=str(out),frames=n))
        if (ordinal+1)%50==0:print('PREPARED',ordinal+1,flush=True)
    comparison_rows=[]
    for row in f['comparison_rows']:
        name=row['name'];p=load(R/'comparisons'/(name+'.npz'));q=A.comparison_receipt(p)
        q.update(name=name,query=row['query'],reference=row['reference'],source=str(R/'comparisons'/(name+'.npz')))
        rp=U/'comparison_receipts'/(name+'.json');write(rp,q);pins[str(rp)]=sha(rp)
        comparison_rows.append(dict(name=name,receipt=str(rp),query=row['query'],reference=row['reference']))
    check(f['pins']);check(f['protected'])
    write(U/'RESULT.json',dict(at=now(),status='IMPLEMENTED_PENDING_SEPARATE_VERIFICATION',counts=dict(counts),angle_state_counts=states.tolist(),
        groups=groups,rows=rows,comparison_rows=comparison_rows,comparison_source_groups=read(R/'COMPARE_RESULT.json')['groups'],
        output_pins=pins,training=0,new_classifier_forwards=0,new_detector_forwards=0,operating_changes=0,user_access=0))
    (U/'RUNNING.json').unlink();print(json.dumps(dict(counts)),flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        (U/('FAILURE_'+now().replace(':','_')+'.txt')).write_text(traceback.format_exc(),encoding='utf8');raise
