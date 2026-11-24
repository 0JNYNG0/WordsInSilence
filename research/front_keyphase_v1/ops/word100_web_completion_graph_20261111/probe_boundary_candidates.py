"""Recover existing boundary-head proposals from saved encodings.

This is an observation/proposal audit, not a new selector or model. No expected
temporal order, target span, source-edit index or evaluation truth is supplied.
"""
from graph import *
from prepare_linker import load
from candidate_motion_evidence import candidate_motion_evidence
import torch,numpy as np
O=U.parent;R=O/'word100_candidate_role_learning_20261111';G=O/'word100_guide_requirement_learning_20261111';W=U/'boundary_candidate_probe'

def main():
    assert not (W/'COMPLETE.json').exists()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    role=load('_boundary_probe_role',R/'role_core.py');rc=load('_boundary_probe_req',G/'requirement_core.py')
    net=role.frozen_model();qh=role.RoleAdapter(42);qh.load_state_dict(torch.load(R/'runs/42/model.pt',weights_only=True,map_location='cpu'));qh.eval()
    index=read(R/'EVALUATION_INPUTS.json')['rows']+[dict(s,kind='dynamic_complete_blocks') for s in read(U/'dynamic_blocks/INPUTS_COMPLETE.json')['rows']]
    phase={(r['clip_id'],r['kind']):r for r in read(U/'context_role_training/RESULTS.json') if r['condition']=='permuted_context' and r['seed']==42}
    write(W/'FROZEN.json',dict(at=stamp(),scope='Existing frozen boundary and quality heads only; reuses cached h/q/fh; preserves all phase/short child proposals separately',
      candidate_algorithm='Unchanged joint_core.candidate_spans; no legacy input added here; full original82-input manifest',
      meaning='complete field from candidate generator means boundary-pair proposal, NOT actual/semantic completion.',
      direction_order_targets_used_for_generation=False,source_index_hints=False,threshold_changes=False,private_access=False,web_changed=False,
      pins={str(p):role.sha(p) for p in [Path(__file__),U/'candidate_motion_evidence.py',role.J/'joint_core.py',role.J/'runs/42/model.pt',R/'runs/42/model.pt',R/'EVALUATION_INPUTS.json',U/'dynamic_blocks/INPUTS_COMPLETE.json']}))
    out=[]
    for s in index:
        assert role.sha(s['path'])==s['sha256'];z=np.load(s['path']);T=len(z['ts']);h=torch.tensor(z['z'][:,228:292][None]);qn=torch.tensor(z['z'][:,292:392].reshape(1,T,10,10));fh=torch.tensor(z['fh'][None]);q=role.old.relation_features(z['x'])
        ids=torch.arange(10);query=torch.cat([torch.nn.functional.one_hot(ids//2,5),torch.nn.functional.one_hot(ids%2,2)],-1).float()[None,None].expand(1,T,-1,-1)
        with torch.inference_mode():
            bp=net.boundary(torch.cat([h[:,:,None].expand(-1,-1,10,-1),qn,fh[:,:,None].expand(-1,-1,10,-1),query],-1)).sigmoid()[0].numpy()
        candidates=role.jc.candidate_spans(bp,q,[])
        specs=[(0,c['a'],c['b'],c['axis']*2+(c['moving']=='L')) for c in candidates]
        with torch.inference_mode():
            features=net.segment(h,torch.tensor(q[None]),fh,specs)
            quality=qh.quality(features).softmax(-1).numpy() if len(specs) else []
        for c,p in zip(candidates,quality):
            c['boundary_pair_proposal']=c.pop('complete');c.update(quality=p.tolist(),quality_class=int(p.argmax()),quality_confidence=float(p.max()),physical_complete=None,semantic_complete=None)
        saved=phase[s['clip_id'],s['kind']];word=s['clip_id'].split('_')[2];pm=[]
        for card in saved['cards']:
            spec=next(c for c in rc.FAMILIES[word] if c['id']==card['id'])
            if spec['axis'] is None:continue
            k=spec['axis']*2
            for group in card['groups']:
                for c in group['candidates']:
                    values=q[:,k,:8];valid=np.repeat(q[:,k,8:10]>.5,4,axis=1)
                    evidence=candidate_motion_evidence(values,valid,z['ts'],*c['span'])
                    pm.append(dict(card=card['id'],axis=spec['axis'],moving='R',span=c['span'],was_selected=group['span']==c['span'],motion=evidence))
        p=W/'probabilities'/f"{s['clip_id']}_{s['kind']}.npz";p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,boundary=bp)
        out.append(dict(clip_id=s['clip_id'],kind=s['kind'],boundary_probability=str(p),boundary_sha256=role.sha(p),candidates=candidates,preserved_phase_cards=saved['cards'],phase_motion_measurements=pm))
    write(W/'RESULTS.json',out);write(W/'COMPLETE.json',dict(rows=len(out),boundary_candidates=sum(len(r['candidates']) for r in out),phase_motion_candidates=sum(len(r['phase_motion_measurements']) for r in out),no_new_encoder_inference=True,no_training=True,no_runtime_change=True))
    print(read(W/'COMPLETE.json'))

if __name__=='__main__':main()
