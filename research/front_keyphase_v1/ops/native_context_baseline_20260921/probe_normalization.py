"""Same annotation frames, full-window normalization; no inference boundary selection."""
import evaluate_full_context as base
from pilot.data import build_channels
from collections import Counter

P=base.HERE;read=base.read;sha=base.sha;np=base.np;torch=base.torch;json=base.json
plan=read(P/'plan.json');control_plan=read(P/'normalization_plan.json')
assert not (P/'normalization_result.json').exists()
for p,d in {**plan['source_pins'],**control_plan['pins']}.items():assert sha(p)==d,p
assert sha(plan['checkpoint']['path'])==plan['checkpoint']['sha256']
ck=torch.load(plan['checkpoint']['path'],map_location='cpu',weights_only=True)
model,classes=base.union.rebuild_from_payload(ck);model.eval().requires_grad_(False);torch.set_num_threads(2)
targets=set(base.PROJECT_18.values());target_ids=[i for i,c in enumerate(classes) if c in targets]
weight=ck['state_dict']['head.weight'].numpy().astype(float);bias=ck['state_dict']['head.bias'].numpy().astype(float)
captured=[];hook=model.head.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().numpy().copy()[0]))
rows=[];features=[];scores_list=[];max_affine=0.
try:
    with torch.inference_mode():
        for spec in plan['inputs']:
            assert sha(spec['npz'])==spec['npz_sha256']
            assert sha(spec['original_annotation_npz'])==spec['original_annotation_sha256']
            with np.load(spec['npz'],allow_pickle=False) as full, np.load(spec['original_annotation_npz'],allow_pickle=False) as old:
                selected=full['annotation_source_rows'];x=full['x'][selected].copy();x[0,-1]=0
                original=build_channels(old['xy_native'],old['coordinate_present'],old['timestamp_nominal_us'])['x']
                assert x.shape==original.shape==(spec['annotation_frames'],280)
                np.testing.assert_array_equal(full['frame_index'][selected],old['frame_index'])
            masks_identical=np.array_equal(x[:,186:279],original[:,186:279])
            dt_identical=np.array_equal(x[:,-1],original[:,-1]);assert dt_identical
            differences={}
            for name,lo,hi in [('body',0,102),('left_hand',102,144),('right_hand',144,186)]:
                changed=x[:,lo:hi]-original[:,lo:hi]
                nonzero=np.abs(x[:,lo:hi])>1e-8
                ratios=original[:,lo:hi][nonzero]/x[:,lo:hi][nonzero]
                differences[name]={'mean_abs':float(np.abs(changed).mean()),'max_abs':float(np.abs(changed).max()),
                                  'original_to_full_value_ratio_median':float(np.median(ratios)) if len(ratios) else None}
            bx,pad,_=base.collate([{'x':x,'frames':len(x),'label':0}])
            direct=model(torch.from_numpy(bx),torch.from_numpy(pad))[0].numpy()
            z=captured[-1].astype(float);scores=weight@z+bias
            max_affine=max(max_affine,float(np.max(np.abs(scores-direct))));assert max_affine<1e-4
            features.append(z);scores_list.append(scores)
            yi=classes.index(spec['reference_label']);evaluations={}
            scopes={'full93':list(range(93))}
            if spec['reference_label'] in targets:scopes['project18']=target_ids
            for scope,indices in scopes.items():
                ranking=sorted(indices,key=lambda i:-float(scores[i]));other=next(i for i in ranking if i!=yi)
                evaluations[scope]={'prediction':classes[ranking[0]],'correct':ranking[0]==yi,'true_rank':ranking.index(yi)+1,
                                    'true_margin':float(scores[yi]-scores[other]),
                                    'original_annotation_prediction':spec['annotation_prediction_'+scope]}
            rows.append({'clip_id':spec['clip_id'],'reference_label':spec['reference_label'],'frames':len(x),
                'masks_identical':masks_identical,'timestamps_identical':dt_identical,'coordinate_differences':differences,
                'input_sha256':base.hashlib.sha256(x.tobytes()).hexdigest(),'evaluations':evaluations})
finally:hook.remove()
assert len(rows)==len(captured)==74
for p,d in {**plan['source_pins'],**control_plan['pins']}.items():assert sha(p)==d,p
with (P/'normalization_features_scores.npz').open('xb') as f:
    np.savez_compressed(f,features=np.stack(features),scores=np.stack(scores_list),clip_ids=np.array([r['clip_id'] for r in rows]),classes=np.array(classes))
aggregate={}
for scope in ('full93','project18'):
    rr=[r for r in rows if scope in r['evaluations']]
    aggregate[scope]={'n':len(rr),'normalization_control_correct':sum(r['evaluations'][scope]['correct'] for r in rr),
        'original_annotation_correct':sum(r['evaluations'][scope]['original_annotation_prediction']==r['reference_label'] for r in rr),
        'prediction_changes':sum(r['evaluations'][scope]['prediction']!=r['evaluations'][scope]['original_annotation_prediction'] for r in rr)}
result={'status':'COMPLETE_PENDING_RECOUNT','rows':rows,'aggregate':aggregate,'new_encoder_forwards':74,'new_training':0,
    'mask_change_clips':sum(not r['masks_identical'] for r in rows),'max_affine_error':max_affine,
    'plan_sha256':sha(P/'normalization_plan.json'),'script_sha256':sha(__file__),'arrays_sha256':sha(P/'normalization_features_scores.npz'),
    'annotation_or_whole_baseline_recomputed':False,'user_or_reserved_records':0,'runtime_changes':0}
with (P/'normalization_result.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ['aggregate','new_encoder_forwards','mask_change_clips']},ensure_ascii=False))
