from experiment import *
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow

def main():
    assert not (U/'FROZEN.json').exists();torch.set_num_threads(2)
    old,new,changed=data();stats=read(D/'NORMALIZATION.json')
    for rows in [old,new]:
        for r in rows:r['activity']=np.where(r['role']<0,-1,(r['role']==1).astype('int64'))
    rows=[r for r in new if r['kind']=='original'][:2]
    x,rgb,l,y,act=batch_j(rows,stats);m=model_for('new_adapt');m.eval()
    with torch.inference_mode():
        a,b,_=m(x,rgb,l);aa,bb,_=m(x[:1,:l[0]],rgb[:1,:l[0]],l[:1]);torch.testing.assert_close(a[0,:l[0]],aa[0],atol=1e-5,rtol=1e-5);torch.testing.assert_close(b[0,:l[0]],bb[0],atol=1e-5,rtol=1e-5)
        zz=x.clone();zz[:,:,64:74]=torch.rand_like(zz[:,:,64:74]);az,_,_=m(zz,rgb,l);torch.testing.assert_close(a,az)
        ir=rgb.clone();ir[:,:,-1]=0;ab,bc,_=m(x,ir,l);ir[:,:,:768]=123;ac,bd,_=m(x,ir,l);torch.testing.assert_close(ab,ac);torch.testing.assert_close(bc,bd)
    for tag in ['new_frozen','new_adapt']:
        mm=model_for(tag);a,b,_=mm(x,rgb,l);loss_fn(a,y,weights(new,'episode',9)).backward()
        assert (mm.actor.context.weight_ih_l0.grad is not None)==tag.endswith('adapt')
        assert (mm.actor.delta.weight.grad is not None)==tag.endswith('adapt')
        assert mm.actor.embed[0].weight.grad is None and mm.actor.rgb[0].weight.grad is None
    # Independently verify only the promised source labels changed.
    lookup={r['clip_id']:r for r in read(R/'ANNOTATION_OVERLAY.json')['rows']};checks=0
    for a,b in zip(old,new):
        for key in ['x','z','m','rgb','current','available','mapping']:assert np.array_equal(a[key],b[key]),key
        pts={p['frame']:p for p in lookup[a['clip_id']]['points']}
        for j in np.flatnonzero((a['episode']!=b['episode'])|(a['role']!=b['role'])):
            p=pts[int(a['mapping'][j])];assert b['episode'][j]==p['action_family'] and b['role'][j]==p['role'] and b['available'][j];checks+=1
        bad=~b['available']
        assert np.array_equal(a['episode'][bad],b['episode'][bad]) and np.array_equal(a['role'][bad],b['role'][bad])
    audit=dict(changed_rows=changed,source_points=172,source_corrections=1,mapped_changed_points=checks,
        unknown_family_main_points=sum(z['main_unknown'] for z in changed),old_labels_unchanged=True,new_annotations_on_unreviewed_seated_views=False,
        correction_seated_view='f62 face/chin retained in fixed upper-body crop, same source identity; correction applies to all available copies.',
        padding=True,missing_image_feature_mask=True,physical_target_blind=True,encoder_gradient_contract=True,behavior=behavior())
    files=[U/'experiment.py',U/'preflight.py',U/'train.py',U/'evaluate.py',U/'verify.py',R/'ANNOTATION_OVERLAY.json',R/'CORRECTION_RESOLUTION.json',
        L/'PAIR_MANIFEST.json',L/'transfer_data.py',L/'evaluate_transfer.py',J/'sequence_model.py',F/'model.py',D/'NORMALIZATION.json',D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',INITIAL,T/'runs/current_42/model.pt']
    pins={str(p):sha(p) for p in files}
    for r in old:pins[r['cache']]=r['cache_sha256']
    exclusive(U/'INPUTS_VERIFIED.json',audit)
    plan=dict(at=clock(),conditions=CONDITIONS,primary='new_adapt',seed=42,epochs=80,batch_size=12,lr=.0005,encoder_lr=.0001,weight_decay=.01,
        initial=str(INITIAL),rows=252,source_review='12 FIT native clips, recorded IDs02/04/08; source labels same Codex RGB review, not expert truth.',
        design='2x2: old vs new reviewed sparse labels; frozen vs copied event GRU+delta fine-tuning. All other training, initial weights, order and views fixed.',
        protected='C/T3F/P/SignRep input features, original independent physical/shape/body/motion observers, old demos and all source files remain unchanged.',
        trainable_adapt='Copied action context BiGRU and delta projection; existing action readout and activity head. Fixed image/embed encoder.',
        evaluation='88 exposed DEV11-13 cases each once per new model, no completed model reruns. H14/15/user/mentor excluded. Component timing only.',
        baseline_family_used=read(O/'word100_guided_event_order_20261102/FROZEN.json')['baseline_family_eligible'],
        gates=dict(raw_at_least589=True,used_at_least582=True,family_not_worse=True,normal_order_above6_and_all_controls=True,short_reverse_at_least2_and_controls=True,
            clean_order_at_least1_and_controls=True,normal_false_order0=True,reversed_false_pass0=True,rest_false_main0=True,missing_never_supported=True),
        qualification='Primary chosen before evaluation. Failed gates => no web adoption. No DEV threshold/epoch/seed selection. Success still requires whole feedback and latency testing.',
        pins=pins,whole_goal_complete=False)
    exclusive(U/'FROZEN.json',plan)
    sync_flow.U=U
    sync_flow.sync('공개 FIT 추가 과정 정답·시간학습 네 조건 준비',
        '공개FIT12영상40연속프레임판/317장면검토. 새172시점(주요108/준비복귀60/대기4)과기존파생정답1개수정을별도overlay로보존·검산. 기존정답/결과변경없음. 추가정답유무×행동시계열GRU미세조정유무4조건을결과전에동결. 입력252,원본관측/누락/짧은근거와C/T3F/웹보존. 새성능아직미확인.',
        'word100_event_temporal_finetune_20261102/FROZEN.json의4조건80epoch후새모델4x88한번,저장배열별도검산. 역할만반복학습하지않고관측행동시간부품의제한미세조정효과를추가정답효과와분리. old/new x frozen/adapt;primary new_adapt,DEV튜닝금지. current active없음,학습시작후session기록. 이전완료학습/평가재실행금지. 전체goal미완료.',
        'EVENT_TEMPORAL_FACTORIAL_PREFLIGHT_VERIFIED',None)
    print('Preflight verified',checks,'mapped label changes; fixed four-condition design')

if __name__=='__main__':main()
