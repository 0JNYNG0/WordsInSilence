"""Source-only KSL observation preview and frozen-path evidence accounting.

No training, model selection, camera, operating model update, or linguistic verdict.
The previously selected observer is shown even though final external results
do not justify deployment. This diagnostic cannot rescue that failed gate.
"""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import torch
from torch.nn import functional as F
import encode_public as enc
import train_observer as tr
import feedback_evidence_bridge as bridge

U=Path(__file__).resolve().parent;OPS=U.parent

def geometry_paths(rows):
    source=tr.read(OPS/'word100_shared_feature_bundle_20261027/FROZEN.json')
    records={r['clip_id']:r for r in source['rows']};cases=defaultdict(list)
    for i,r in enumerate(rows):cases[r['case_id']].append(i)
    vectors={}
    for case,indices in cases.items():
        r=rows[indices[0]];record=tr.read(Path(records[r['clip_id']]['record']));p=Path(record['views']['native']['observation'])
        with np.load(p,allow_pickle=False) as z:xy=z['xy_px'].copy();valid=z['usable'].copy()
        side={'L':0,'R':1}[r['side']];keys=[13+side,15+side,33+21*side,42+21*side]
        vv=[]
        for i in indices:
            f=rows[i]['frame_id'];q=xy[f];center=(q[11]+q[12])/2;scale=float(np.linalg.norm(q[11]-q[12]))
            if valid[f,keys+[11,12]].all() and scale>1e-6:vv.append(((q[keys]-center)/scale).flatten())
            else:vv.append(np.full(8,np.nan))
        vectors[case]=np.array(vv)
    def dtw(a,b):
        d=np.full((len(a)+1,len(b)+1),np.inf);d[0,0]=0;parent={}
        for i in range(1,len(a)+1):
            for j in range(1,len(b)+1):
                observed=np.isfinite(a[i-1]).all() and np.isfinite(b[j-1]).all()
                cost=float(np.linalg.norm(a[i-1]-b[j-1])) if observed else 1.
                choices=[(i-1,j-1),(i-1,j),(i,j-1)];q=min(choices,key=lambda q:d[q]);d[i,j]=d[q]+cost;parent[(i,j)]=q
        i,j=len(a),len(b);path=[]
        while i and j:path.append([i-1,j-1]);i,j=parent[(i,j)]
        return path[::-1]
    pairs=[]
    for q,indices in cases.items():
        first=rows[indices[0]]
        if first['person']!=3:continue
        for r,refindices in cases.items():
            ref=rows[refindices[0]]
            if ref['word_id']==first['word_id'] and ref['side']==first['side'] and ref['person'] in (1,2):
                pairs.append({'query':q,'reference':r,'query_indices':indices,'reference_indices':refindices,
                              'side':first['side'],'path':dtw(vectors[q],vectors[r])})
    return {'pairs':pairs,'method':'Fixed ordinary DTW over shoulder-normalized elbow,wrist,hand points from existing native observations; no learned probabilities used',
            'missing_geometry_cost':1.,'scope':'Short13-frame research windows only; no proof of improved phase alignment or complete sign equivalence',
            'required':'All13reference positions for all3physical features; technical denominator, not linguistic necessity',
            'identity_basis':'inherited existing source assignment, not new human certification'}

def main():
    assert not (U/'KSL_TRANSFER_PREDICTIONS.json').exists()
    torch.set_num_threads(4)
    c=tr.read(U/'EXPERIMENT.json');source=tr.read(U/'data/KSL_TRANSFER.json');rows=source['rows']
    selected=tr.read(U/'OBSERVER_SELECTION.json')['selected']
    # Preserve the pre-test choice. Do not choose a different observer from final results.
    assert selected['kind']=='observer' and selected['condition']=='RJ_F'
    paths=geometry_paths(rows)
    if (U/'KSL_REFERENCE_PATHS.json').exists():assert tr.read(U/'KSL_REFERENCE_PATHS.json')==paths
    else:tr.save(U/'KSL_REFERENCE_PATHS.json',paths)
    protocol={'selection_sha256':tr.sha(U/'OBSERVER_SELECTION.json'),
        'inputs_sha256':tr.sha(U/'data/KSL_TRANSFER.json'),'paths_sha256':tr.sha(U/'KSL_REFERENCE_PATHS.json'),
        'checkpoints':{str(U/'models'/f'observer_RJ_F_{s}.pt'):tr.sha(U/'models'/f'observer_RJ_F_{s}.pt') for s in c['seeds']},
        'ensemble':'equal1/3softmax; no weights tuned','use':'description only; no candidate selection or KSL training',
        'failed_external_gate_not_overridden':True,'independent_KSL_truth_available':False}
    if (U/'KSL_DIAGNOSTIC_PROTOCOL.json').exists():assert tr.read(U/'KSL_DIAGNOSTIC_PROTOCOL.json')==protocol
    else:tr.save(U/'KSL_DIAGNOSTIC_PROTOCOL.json',protocol)
    cache=U/'data/KSL_OBSERVER_PROBABILITIES.npz'
    assert not cache.exists(),'Saved predictions exist; finish from cache rather than rerun'
    name=tr.read(U/'ENCODER_SELECTION.json')['selected'];model,forward,_=enc.load_encoder(name)
    available=np.array([bool(r['image']) for r in rows]);rgb=np.zeros((len(rows),576),np.float32)
    ids=np.where(available)[0]
    with torch.inference_mode():
        for start in range(0,len(ids),32):
            ix=ids[start:start+32];x=torch.stack([enc.crop_tensor(rows[i]['image']) for i in ix]);rgb[ix]=F.normalize(forward(enc.normalize(x,name)),dim=-1).numpy()
    z=np.load(U/'data/KSL_TRANSFER_JOINTS.npz',allow_pickle=False);raw=z['raw_xyz'].astype(np.float32);seen=z['observed'].copy()
    rel=raw-raw[:,:1];scale=np.linalg.norm(rel[:,[5,9,13,17]],axis=-1).max(1);seen &= scale>1e-6
    rel/=np.maximum(scale[:,None,None],1e-6);rel[~seen]=0
    joint=np.concatenate([rel.reshape(len(rows),63),seen[:,None]],axis=1)
    by={(r['case_id'],r['frame_id']):i for i,r in enumerate(rows)}
    context=np.array([[by.get((r['case_id'],r['frame_id']+d),-1) for d in [-2,-1,0,1,2]] for r in rows]);valid=context>=0
    values=np.concatenate([rgb,joint],axis=1)[np.maximum(context,0)]*valid[:,:,None]
    times=np.zeros((len(rows),5,1),np.float32)
    for i,r in enumerate(rows):
        for k,j in enumerate(context[i]):
            if j>=0:times[i,k,0]=(rows[j]['pts_us']-r['pts_us'])/1e6
    x=torch.tensor(np.concatenate([values,valid[:,:,None],times],axis=-1),dtype=torch.float32)
    allprob=[]
    for seed in c['seeds']:
        p=U/'models'/f'observer_RJ_F_{seed}.pt';saved=torch.load(p,map_location='cpu',weights_only=True)
        m=tr.Observer(x.shape[-1],45,False);m.load_state_dict(saved['state_dict'],strict=True);m.eval()
        with torch.inference_mode():allprob.append(torch.softmax(m(x),dim=-1).numpy())
    probs=np.mean(allprob,axis=0)
    np.savez_compressed(cache,probabilities=probs,seed_probabilities=np.array(allprob))
    out=[];frames={};known=agree=accepted=0
    for i,r in enumerate(rows):
        hand=bridge.project_class_probabilities(probs[i].tolist(),c['classes'],hand_present=bool(available[i]),identity_resolved=not r['identity_uncertain'])
        hand['identity_basis']='inherited native assignment; not independent truth'
        top=np.argsort(-probs[i])[:3];rr={k:r[k] for k in ['case_id','word_id','side','person','frame_id','pts_us','image','full_image','draft_only']}
        rr['features']=hand['features'];rr['top3']=[{'class_name':c['classes'][j],'probability':float(probs[i,j])} for j in top]
        rr['hand_crop_present']=bool(available[i]);rr['crop_detector_present']=bool(seen[i]);rr['educational_verdict']=None
        out.append(rr);frames[i]={'time_s':r['pts_us']/1e6,'hands':{r['side']:hand}}
        for key in ['index_only','open_four']:
            if r['draft_only'][key] is not None:
                known+=1
                value=hand['features'][key]['state']
                accepted+=value is not None;agree+=value is not None and value==r['draft_only'][key]
    comparisons=[];controls=[]
    byquery=defaultdict(list)
    for pair in paths['pairs']:byquery[pair['query']].append(pair)
    for case,pairs in byquery.items():
        query=[frames[i] for i in pairs[0]['query_indices']];references=[]
        for pair in pairs:
            references.append({'reference_id':pair['reference'],'frames':[frames[i] for i in pair['reference_indices']],
                'path':pair['path'],'required':{side:{f:[side==pair['side']]*13 for f in bridge.FEATURES} for side in ['L','R']}})
        result=bridge.compare_references(query,references);comparisons.append({'query':case,'result':result})
        missing=json.loads(json.dumps(query))
        for frame in missing:
            for side in list(frame['hands']):frame['hands'][side]=bridge.observed_hand({},hand_present=False,identity_resolved=False,evidence_id='explicit_missing_contract_control')
        controls.append({'query':case,'result':bridge.compare_references(missing,references)})
    summary={'frames':len(rows),'windows':36,'visible_crops':int(available.sum()),'crop_redetections':int(seen.sum()),
             'draft_known_cells':known,'model_answered_known_draft_cells':accepted,'agreement_with_AI_draft':int(agree),
             'agreement_is_accuracy':False,'reference_pairs':len(paths['pairs']),'query_windows':len(byquery),
             'real_correctness_validation':False,'operating_candidate_enabled':False,
             'scope':'Selected pre-test candidate descriptive outputs. Existing AI drafts are neither training labels nor independent evaluation truth.'}
    tr.save(U/'KSL_TRANSFER_PREDICTIONS.json',{'summary':summary,'rows':out})
    tr.save(U/'KSL_BRIDGE_REPLAY.json',{'comparisons':comparisons,'all_missing_controls':controls,'paths_sha256':tr.sha(U/'KSL_REFERENCE_PATHS.json'),
         'normal_crossing_accuracy':'not established','educational_verdict_enabled':False,'production_changed':False})
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
