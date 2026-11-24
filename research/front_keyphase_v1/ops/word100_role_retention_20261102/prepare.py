from retention_model import *
import time,traceback

def main():
    assert not (U/'TEACHER_INPUTS.json').exists();torch.set_num_threads(2)
    rows,normal=fit_rows();stats=read(ex.D/'NORMALIZATION.json');assert len(rows)==252
    plan=dict(at=clock(),conditions=['new_labels_only','new_labels_with_retention'],primary='new_labels_with_retention',seed=42,epochs=80,batch_size=12,lr=.0005,weight_decay=.01,
        retention_weight=.1,retention_temperature=1,source_checkpoint=str(INITIAL),teacher_checkpoint=str(TEACHER),
        goal='Retain newly learned short-order discrimination while preventing previously correct main events from being excluded.',
        design='Both conditions start with the same new_adapt actor/role weights and zero-initialized learned-prototype flow bridge. Actor frozen; only role/flow layers train. Difference is0vs0.1soft KL on unlabelled FIT times.',
        soft_target_scope='Old role probabilities on public FIT01-09 only. Not independent truth. Excluded on all explicit role labels, unavailable images and artificial edit seams. Explicit new labels take precedence.',
        split='FIT01-09 only; development11-13 exposed; no H14/15/user/mentor access. No17/18media evaluation until independent-use audit.',
        evaluation='Same88DEVcases per new model once. Corrected nonmain-union decision fixed; raw action exactly new_adapt590/615. No threshold/seed/epoch/per-word selection.',
        gates=['raw_exact','used_at_least583_and_control','family_not_worse','normal_order_at_least19_and_control','clean_order_at_least5_and_control',
            'short_reverse_at_least2_and_control','normal_false_order0','reverse_false_pass0','missing_held'],
        previous_goal_turn='PROGRESS: completed four neural trainings352comparisons, corrected common probability contract440cases; then crossovers176cases localized gain and loss to role branch.',
        pins={str(p):sha(p) for p in [U/'retention_model.py',U/'prepare.py',INITIAL,TEACHER,fl.FLOW_UNIT/'FLOW_INPUTS.json',ex.D/'NORMALIZATION.json',ex.R/'ANNOTATION_OVERLAY.json',O/'word100_content_role_crossover_20261102/VERIFIED.json']},
        whole_goal_complete=False)
    exclusive(U/'PREPARATION_FREEZE.json',plan)
    # Zero input bridge must be behaviorally identical before any new updates.
    m=RetainedRoleNet().eval();baseline=ex.model_for('new_adapt');baseline.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=True));baseline.eval()
    batch=rows[:2];x,rgb,l,y,a=ex.batch_j(batch,stats);flow=fl.flow_batch(batch,normal)
    with torch.inference_mode():
        p,q=m(x,rgb,l,flow);pp,qq,_=baseline(x,rgb,l);torch.testing.assert_close(p,pp,rtol=0,atol=0);torch.testing.assert_close(q,qq,rtol=0,atol=0)
        p1,q1=m(x[:1,:l[0]],rgb[:1,:l[0]],l[:1],flow[:1,:l[0]]);torch.testing.assert_close(q[0,:l[0]],q1[0],rtol=1e-5,atol=1e-5)
    pp,qq=m(x,rgb,l,flow);ex.loss_fn(qq,a,torch.ones(2)).backward()
    assert all(p.grad is None for p in m.actor.parameters()) and m.flow_bridge.weight.grad is not None
    dummy=torch.randn(1,4,2,requires_grad=True);teacher=torch.softmax(torch.randn_like(dummy),-1);mask=torch.tensor([[True,False,True,False]])
    retention_loss(dummy,teacher,mask).backward();assert not dummy.grad[0,[1,3]].any()
    teacher_model=fl.FlowRoleNet();teacher_model.load_state_dict(torch.load(TEACHER,map_location='cpu',weights_only=True));teacher_model.eval()
    for p in teacher_model.parameters():p.requires_grad_(False)
    manifest=[];start=time.perf_counter()
    for i,r in enumerate(rows):
        x,rgb,l,_,_=ex.batch_j([r],stats);flow=fl.flow_batch([r],normal)
        with torch.inference_mode():_,logits=teacher_model(x,rgb,l,flow);prob=logits.softmax(-1)[0].numpy()
        mask=retention_mask(r);assert not mask[r['role']>=0].any() and not mask[~r['available']].any()
        dest=OUT/'teacher'/f"{r['flow_key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
        np.savez_compressed(dest,probability=prob,mask=mask,source_frame_id=r['mapping'],image_available=r['available'],role_label=r['role'])
        manifest.append(dict(index=i,clip_id=r['clip_id'],kind=r['kind'],person=r['person'],flow_key=r['flow_key'],teacher_path=str(dest),teacher_sha256=sha(dest),
            input_cache=r['cache'],input_sha256=r['cache_sha256'],soft_frames=int(mask.sum()),frames=len(mask)))
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(U/'TEACHER_INPUTS.json',dict(at=clock(),rows=manifest,clips=252,soft_frames=sum(r['soft_frames'] for r in manifest),seconds=time.perf_counter()-start,
        old_DEVevaluation_repeated=False,new_FIT_teacher_inference=252,preflight=dict(zero_bridge_exact=True,padding=True,actor_frozen=True,KL_excludes_labelled_and_missing=True),whole_goal_complete=False))
    print('Prepared252FIT soft teacher sequences; labelled/missing/seams excluded; training not yet run')
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (U/'PREP_FIRST_FAILURE.txt').exists():(U/'PREP_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
