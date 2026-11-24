"""Measure the existing quality head on all new phase-candidate intervals.

Diagnostic only: quality classes do not establish continuous observed movement.
No selector changes, no new trainable weights, no expected-order input.
"""
from graph import *
from prepare_linker import load
import numpy as np,torch
O=U.parent;R=O/'word100_candidate_role_learning_20261111';G=O/'word100_guide_requirement_learning_20261111';W=U/'candidate_quality_probe'

def main():
    assert not (W/'COMPLETE.json').exists()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    role=load('_quality_probe_role',R/'role_core.py');rc=load('_quality_probe_req',G/'requirement_core.py')
    model=role.RoleAdapter(42);model.load_state_dict(torch.load(R/'runs/42/model.pt',weights_only=True,map_location='cpu'));model.eval()
    segnet=role.frozen_model()
    source={ (s['clip_id'],s['kind']):s for s in read(R/'EVALUATION_INPUTS.json')['rows']}
    source.update({(s['clip_id'],'dynamic_complete_blocks'):s for s in read(U/'dynamic_blocks/INPUTS_COMPLETE.json')['rows']})
    rows=[r for r in read(U/'context_role_training/RESULTS.json') if r['condition']=='permuted_context' and r['seed']==42]
    write(W/'FROZEN.json',dict(at=stamp(),scope='All phase-supported moving candidate windows in exposed82 inputs. Diagnostics only, no quality-based selection/promotion.',
      classes=['hold_or_background','derived_partial','reviewed_complete_extent','multiple_events'],
      limitations='Training labels are event extent, not continuity, anatomical correctness, or sign completion. Family-static cards excluded from motion quality.',
      model=str(R/'runs/42/model.pt'),model_sha256=role.sha(R/'runs/42/model.pt'),input_sha256=role.sha(U/'context_role_training/RESULTS.json'),
      code_sha256=role.sha(Path(__file__)),thresholds_changed=False,private_access=False,web_changed=False))
    out=[]; reconstruction=[]
    for row in rows:
        cid,kind=row['clip_id'],row['kind'];s=source[cid,kind];assert role.sha(s['path'])==s['sha256'];z=np.load(s['path']);word=cid.split('_')[2]
        h=torch.tensor(z['z'][:,228:292][None]);fh=torch.tensor(z['fh'][None]);q=torch.tensor(role.old.relation_features(z['x'])[None])
        # Verify cached normalized relationship features and h/fh offsets.
        with torch.inference_mode():
            qn=(q-segnet.qm)/segnet.qs;qn=torch.cat([qn[...,:4]*q[...,8,None],qn[...,4:8]*q[...,9,None],q[...,8:]],-1)
        assert np.allclose(qn[0].numpy().reshape(len(qn[0]),100),z['z'][:,292:392],atol=1e-6)
        assert np.array_equal(fh[0].numpy(),z['z'][:,392:456])
        specs=[];meta=[];seen=set()
        for card in row['cards']:
            definition=next(c for c in rc.FAMILIES[word] if c['id']==card['id'])
            if definition['axis'] is None:continue
            # Existing annotations supervise R; L not silently claimed tested.
            k=definition['axis']*2
            for group in card['groups']:
                for candidate in group['candidates']:
                    a,b=candidate['span'];key=(card['id'],a,b,k)
                    if key in seen:continue
                    seen.add(key);specs.append((0,a,b,k));meta.append(dict(card=card['id'],span=[a,b],axis=k//2,moving='R',role_peak=candidate['score'],was_selected=group['span']==[a,b]))
        with torch.inference_mode():
            features=segnet.segment(h,q,fh,specs)
            prob=model.quality(features).softmax(-1).numpy() if len(specs) else np.zeros((0,4))
        for m,p in zip(meta,prob):m.update(quality=p.tolist(),quality_class=int(p.argmax()),quality_confidence=float(p.max()))
        out.append(dict(clip_id=cid,kind=kind,candidates=meta))
        reconstruction.append(dict(clip_id=cid,kind=kind,cached_h_q_fh_verified=True))
    write(W/'RESULTS.json',out);write(W/'COMPLETE.json',dict(rows=len(out),candidates=sum(len(r['candidates']) for r in out),input_reconstruction=reconstruction,changed_selection=False,web_changed=False))
    print('Quality diagnostic complete:',len(out),'inputs,',sum(len(r['candidates']) for r in out),'candidate intervals')
    for r in out:
        if r['kind']=='dynamic_complete_blocks' and '0739_REAL11' in r['clip_id']:print(r)

if __name__=='__main__':main()
