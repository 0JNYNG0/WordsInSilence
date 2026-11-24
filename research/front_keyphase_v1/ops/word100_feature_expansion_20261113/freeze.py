from model_core import *
from collections import Counter

def main():
    assert not (U/'FROZEN.json').exists()
    ready=read(U/'DATA_READY.json');rr=rows_load();common=[r for r in rr if r['origin']=='common100'];extra=[r for r in rr if r['origin']=='selected_extra']
    # Historical binary thumb-folded does not distinguish thumb adduction from flexion.
    # Remove these old coarse thumb targets before fixing the fine three-state task.
    changed=[]
    for r in extra:
        if r['word_id'] not in ['WORD2633','WORD2635']:continue
        z={k:v for k,v in r.items() if isinstance(v,np.ndarray)};n=int((z['fy'][:,:,0]>=0).sum());z['fy'][:,:,0]=-1
        if n:
            np.savez_compressed(r['path'],**z);changed.append(dict(clip_id=r['clip_id'],masked_thumb_cells=n))
            m=next(m for m in ready['rows'] if m['clip_id']==r['clip_id']);m['sha256']=sha(r['path']);m['finger_known']=int((z['fy']>=0).sum());r['fy']=z['fy']
    write(U/'DATA_READY.json',ready)
    fit=[r for r in rr if r['role']=='fit'];dev=[r for r in rr if r['role']=='development']
    assert len([r for r in common if r['role']=='fit'])==899 and len([r for r in common if r['role']=='development'])==300
    assert not ({r['person'] for r in fit}&{r['person'] for r in dev})
    assert not ({r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev})
    assert len({r['clip_id'] for r in rr})==len(rr)
    for r in rr:
        assert np.isfinite(r['hand']).all() and np.isfinite(r['x_full']).all()
        assert (np.diff(r['timestamp_us'])>0).all() and (np.diff(r['frame_id'])>0).all()
        assert set(np.unique(r['fy'])).issubset({-1,0,1,2})
    panels=read(U/'annotation_review/ROOT_REVIEW_INPUTS.json')['panels']
    write(U/'ROOT_LABEL_REVIEW.json',dict(status='RESEARCH_ANNOTATIONS_ACCEPTED_WITH_LIMITS',reviewer='root Codex image review separate from annotating agent; not KSL human expert',reviewed_frames=24,panels=[p['panel'] for p in panels],finding='E/B/F long-finger states agree with visible contours in 24 selected FIT/DEV frames; edge-on/hidden fingers stay unknown. New thumb E only. B examples sparse, no thumb B/F generalization.',legacy_thumb_masked=changed,allow_training=True,allow_educational_error_claim=False))
    def stats(rows):
        return dict(clips=len(rows),words=len({r['word_id'] for r in rows}),finger_cells=int(sum((r['fy']>=0).sum() for r in rows)),states={str(i):int(sum((r['fy']==i).sum() for r in rows)) for i in [-1,0,1,2]},hands={s:int(sum((r['fy'][:,i]>=0).sum() for r in rows)) for i,s in enumerate(['L','R'])})
    coverage={name:stats([r for r in rr if r['origin']==origin and r['role']==role]) for name,origin,role in [('ordinary_fit','common100','fit'),('ordinary_development','common100','development'),('extra_fit','selected_extra','fit'),('extra_development','selected_extra','development')]}
    write(U/'COVERAGE.json',coverage)
    pins={str(p):sha(p) for p in [U/'DATA_READY.json',U/'model_core.py',U/'train.py',U/'ROOT_LABEL_REVIEW.json',U/'annotation_review/ORDINARY_LABELS.json',U/'annotation_review/EXTERNAL_LABELS.json',old.V/'FEEDBACK_NORMALIZATION.json',B/'core.py',J/'feedback_model.py']}
    pins.update({str(B/f'runs/shared100_{s}/model.pt'):sha(B/f'runs/shared100_{s}/model.pt') for s in [42,43,44]})
    protection=read(A/'FROZEN.json')['protection'];assert all(sha(p)==h for p,h in protection.items())
    cfg=dict(status='FROZEN_BEFORE_NEW_FINGER_SCORES',conditions=['A','B','C'],seeds=[42,43,44],epochs=12,warmup_epochs=2,steps_per_epoch=38,lr_base=.00003,lr_head=.001,retention_weight=10.,extra_words=ready['extra_words'],pins=pins,protected=protection,initialization='existing common100 research SharedNet per paired seed; not production checkpoint replacement',architecture='existing temporal conv+biGRU plus shared left/right finger head: 63 local ->32, temporal64->16, concat48->15 three-state logits; training-only 7 extra word logits',conditions_explained={'A':'common100 + new fine ordinary labels','B':'A + seven selected extra word classes','C':'B + direct extra finger labels'},same_batches_capacity_steps=True,extra_inputs='6 numeric words + 3 verified apple records; 4 ID05 corrupt source remains a counted source failure; no apple05',fine_targets='Sparse native RGB anchors only. E=0/B=1/F=2/U=-1. Historical thumb folded of2/4 masked. No automatic propagation or numeric-name-derived fine targets.',normalization='Existing common100 FIT-only stats; palm-local shape only; original body-relative direction unchanged. Extra full view has no historical crop.',eval=dict(primary='macroF1 three finger states on all known ordinary DEV cells; masked/unknown excluded from truth not credited. Per-hand/state/record/word and all seeds.',variants=['full','upper','speed_075','speed_125','short_missing_2','long_missing_12'],confidence=.85,gate='C macroF1 must exceed A by0.01 in every seed; per-hand accuracy and known-state F1 may not decline; original coarse shape/stage/100word retention must not regress versus paired saved common100. Full-path order/direction requires separate validation; no automatic promotion.',old_no_fine_head='No old per-finger accuracy exists; baseline compare is old coarse+stage vs A. A-to-B-to-C only for fine fingers.',data_status='Historical DEV11..13, no new independent subjects; physical-attribute truth from Codex review not human educational errors.'),training_no_private=True,production_changed=False)
    write(U/'FROZEN.json',cfg);write(U/'PRETRAIN_CHECK.json',dict(status='PASS',frozen_sha256=sha(U/'FROZEN.json'),inputs=len(rr),fit=len(fit),dev=len(dev),source_person_roles_disjoint=True,finite=True,coverage=coverage))
    event('frozen',inputs=len(rr),fit=len(fit),dev=len(dev),coverage=coverage)

if __name__=='__main__':main()
