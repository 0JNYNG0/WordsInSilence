"""Independent saved-data audit. No project/model imports or neural execution."""
from pathlib import Path
import json, hashlib, math
from collections import Counter
import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    d = read(U/'DATASET_FROZEN.json')
    checks=[]
    def ck(name, ok): checks.append({'name':name,'passed':bool(ok)})
    review = {}
    for word in ['0739','1133','1567','2523']:
        path=U/('WORD'+word+'_REVIEW.json')
        for r in read(path)['rows']:
            for q in r['queries']: review[(r['clip_id'],q['target_index'])]=(r,q,path)
    mp=O/'word100_learned_motion_order_20261102/MOTION_TARGETS.json'
    positive={}
    for r in read(mp)['rows']:
        if r['kind']=='directed':
            for j in range(5):
                if r['mask'][j] and r['y'][j]==1: positive[(r['clip_id'],j)]=r
    ck('expected_review_and_positive_queries60',len(review)==len(positive)==60)
    ck('no_private_or_locked_source_paths',all(not any('REAL'+str(i)+'_' in p for i in [14,15,16]) for p in d['pins']))
    for p,h in d['pins'].items(): ck('pin:'+p,sha(p)==h)
    lookup={r['clip_id']:r for r in d['rows']}
    ck('source48_unique',len(d['rows'])==len(lookup)==48)
    ck('source36fit12dev',Counter(r['role'] for r in d['rows'])=={'fit':36,'development':12})
    fitsha={r['source_sha256'] for r in d['rows'] if r['role']=='fit'}
    devsha={r['source_sha256'] for r in d['rows'] if r['role']=='development'}
    ck('source_hash_unique_split',len(fitsha)==36 and len(devsha)==12 and not fitsha&devsha)
    keep=read(O/'word100_learned_motion_invariance_20261102/TRAINING_CONTRACT.json')['motion_value_indices']
    valid=np.zeros(146,np.float32);valid[keep]=1;valid[np.array(keep)+73]=1
    ck('frozen_geometry_selection',d['geometry_valid']==valid.tolist())
    axes={}; observed_by_clip={}; fragments_by_clip={}
    for r in d['rows']:
        cid=r['clip_id']; role='fit' if r['person'] in range(1,10) else 'development' if r['person'] in [11,12,13] else None
        ck(cid+':registered_role',role==r['role'] and cid==f"NIA_SL_{r['word_id']}_REAL{r['person']:02d}_F")
        ck(cid+':source_input_pin_binding',d['pins'][str(Path(r['source']).resolve())]==r['source_sha256'] and d['pins'][str(Path(r['input']).resolve())]==r['input_sha256'])
        with np.load(r['input'],allow_pickle=False) as a, np.load(r['shape_observation']['path'],allow_pickle=False) as s, np.load(r['relation_observation']['path'],allow_pickle=False) as rel:
            ids=a['frame_id'];ts=a['timestamp_us'];n=len(ids)
            ck(cid+':exact_native_axes',ids.tolist()==r['native_frame_ids']==list(range(n)) and ts.tolist()==r['timestamp_us'] and bool(np.all(np.diff(ts)>0)))
            ck(cid+':auxiliary_axes',all(np.array_equal(a[k],z[k]) for k in ['frame_id','timestamp_us'] for z in [s,rel]))
            ck(cid+':dimensions',a['x'].shape==(n,69) and a['mask'].shape==(n,69) and a['rgb'].shape==(n,768) and r['input_shape']==[n,915])
            x=a['x'];mask=a['mask'].astype(bool);av=a['image_available'].astype(bool) if 'image_available' in a else np.ones(n,bool)
            v=x[:,2:4]-x[:,4:6];delta=x[:,36:38]-x[:,4:6];norm=(v*v).sum(1)
            good=mask[:,[2,3,4,5,36,37]].all(1)&np.isfinite(norm)&(norm>.0025)
            along=(delta*v).sum(1)/np.maximum(norm,1e-6)
            side=(delta[:,0]*v[:,1]-delta[:,1]*v[:,0])/np.maximum(norm,1e-6)
            m=np.column_stack([mask[:,:61]&np.isfinite(x[:,:61]),good&np.isfinite(along),good&np.isfinite(side),np.isfinite(s['probabilities']),np.isfinite(rel['probability'])])&av[:,None]
            observed=(av&np.isfinite(a['rgb']).all(1))|(m[:,keep].any(1))
            edges=np.diff(np.r_[False,observed,False].astype(int))
            fragments=[[int(lo),int(hi)-1] for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
            anchor=np.zeros(n,bool)
            for lo,hi in fragments:
                if hi>lo:anchor[lo:hi+1]=True
            ck(cid+':actual_observation_fragments',observed.tolist()==r['evidence_available'] and fragments==r['observed_fragments'] and anchor.tolist()==r['motion_anchor'])
            axes[cid]=(ids.copy(),ts.copy());observed_by_clip[cid]=observed;fragments_by_clip[cid]=fragments
    ck('groups60_unique',len(d['groups'])==60 and len({(g['clip_id'],g['target_index']) for g in d['groups']})==60)
    actual_counts={}; overlaps=[]; new_positive_support=[]; per_query={}
    for g in d['groups']:
        key=(g['clip_id'],g['target_index']);row=lookup[g['clip_id']];p=positive[key];r,q,rpath=review[key];ids,ts=axes[g['clip_id']]
        ck(g['group_id']+':provenance',all(g[k]==row[k]==r[k]==p[k] for k in ['role','person','source_sha256']) and r['input_sha256']==p['input_sha256']==row['input_sha256'])
        ck(g['group_id']+':two_bags_equal_group_weight',len(g['bags'])==2 and g['group_weight']==1.0)
        ck(g['group_id']+':broad_no_loss',g['broad_consistency']['loss_weight']==0 and not g['broad_consistency']['train_eligible'] and g['broad_consistency']['indices']==[0,len(ids)-1])
        for b in g['bags']:
            lo,hi=b['indices'];fs=ids[lo:hi+1].tolist();t=ts[lo:hi+1].tolist();old=b['kind']=='weak_positive_bracket'
            expected_frames=p['frames'] if old else [r['native_frames'][0],r['native_frames'][-1]]
            expected_label=1 if old else q['presence'];witness=fs if old else q['witness_frames']
            ck(b['bag_id']+':exact_scope_label',b['frames']==expected_frames and b['label']==expected_label and b['native_frames']==fs==list(range(*[expected_frames[0],expected_frames[1]+1])) and b['timestamp_us']==t)
            ck(b['bag_id']+':witness_pts',b['witness_frames']==witness and b['witness_timestamp_us']==[int(ts[f]) for f in witness] and set(witness).issubset(fs))
            ck(b['bag_id']+':review_source',Path(b['evidence']['path']).resolve()==(mp if old else rpath).resolve() and b['evidence']['sha256']==sha(mp if old else rpath))
            complete=hi>lo and bool(observed_by_clip[g['clip_id']][lo:hi+1].all()) and any(a<=lo and hi<=z for a,z in fragments_by_clip[g['clip_id']])
            eligible=expected_label is not None and complete
            ck(b['bag_id']+':eligibility',b['label_eligible']==eligible and b['train_eligible']==(eligible and g['role']=='fit' and g['target_index'] in d['supported_targets']))
            ck(b['bag_id']+':semantics_unknown_preserved',b['whole_clause_truth'] is None and b['educational_error'] is None and (expected_label is not None or (not b['train_eligible'] and not b['label_eligible'])))
            s=b['composed_structural_support']
            ck(b['bag_id']+':nominal_support',s['nominal_native_frames']==fs and s['nominal_timestamp_us']==t and s['nominal_indices']==[lo,hi])
            tcn=sorted({i for a,z in fragments_by_clip[g['clip_id']] if z>a for t0 in range(max(lo,a),min(hi,z)+1) for i in range(max(a,t0-7),min(z,t0+7)+1)})
            ck(b['bag_id']+':tcn_support',s['tcn_input_indices']==tcn and s['tcn_native_frames']==ids[tcn].tolist())
            ck(b['bag_id']+':rgb_support_time_and_outside',s['rgb_native_frames']==ids[s['rgb_input_indices']].tolist() and s['rgb_timestamp_us']==ts[s['rgb_input_indices']].tolist() and s['rgb_outside_nominal_native_frames']==sorted(set(s['rgb_native_frames'])-set(fs)))
            if not old and expected_label==1:new_positive_support.append({'clip_id':g['clip_id'],'role':g['role'],'target_index':g['target_index'],'bag_frames':b['frames'],'witness_frames':witness,'witness_timestamp_us':b['witness_timestamp_us']})
        lo=max(p['frames'][0],r['native_frames'][0]);hi=min(p['frames'][1],r['native_frames'][-1])
        if lo<=hi:overlaps.append({'clip_id':g['clip_id'],'target_index':g['target_index'],'new_label':q['presence'],'overlap':[lo,hi],'negative_contains_entire_positive':q['presence']==0 and r['native_frames'][0]<=p['frames'][0] and p['frames'][1]<=r['native_frames'][-1]})
    for role in ['fit','development']:
        gs=[g for g in d['groups'] if g['role']==role];bags=[b for g in gs for b in g['bags']];local=[b for b in bags if b['kind']=='reviewed_local']
        actual_counts[role]={'sources':sum(r['role']==role for r in d['rows']),'groups':len(gs),'weak_positive_bags':sum(b['kind']=='weak_positive_bracket' for b in bags),'reviewed_local_bags':len(local),'local_labels':{str(k):sum(b['label']==k for b in local) for k in [0,1,None]},'all_bags':len(bags),'full_native_consistency_queries':len(gs)}
        ck(role+':denominators',actual_counts[role]==d['denominators'][role])
        for j in range(5):
            bs=[b for g in gs if g['target_index']==j for b in g['bags']]
            per_query[role+':'+str(j)]={'positive':sum(b['label']==1 for b in bs),'negative':sum(b['label']==0 for b in bs),'unknown':sum(b['label'] is None for b in bs),'eligible':sum(b['label_eligible'] for b in bs)}
    support=[]
    for j in range(5):
        ids={str(y):sorted({g['person'] for g in d['groups'] if g['role']=='fit' and g['target_index']==j and any(b['label']==y and b['label_eligible'] for b in g['bags'])}) for y in [0,1]}
        s={'target_index':j,'fit_ids_by_label':ids,'supported':all(len(x)>=3 for x in ids.values()),'minimum_distinct_fit_ids_per_class':3};support.append(s)
    ck('target_support_exact',support==d['target_support'] and [s['target_index'] for s in support if s['supported']]==d['supported_targets'])
    trains=[g['group_id'] for g in d['groups'] if g['role']=='fit' and any(b['train_eligible'] for b in g['bags'])]
    ck('training_groups_and_budget',trains==d['training_group_ids'] and len(trains)==45 and d['epochs']==80 and d['group_batch_size']==12 and d['training_steps']==80*math.ceil(len(trains)/12)==320)
    ck('no_fully_contained_logical_conflict',not any(x['negative_contains_entire_positive'] for x in overlaps))
    inp=read(U/'INPUT_CHECKS.json')
    ck('builder_checks_lineage',inp['status']=='PASS' and len(inp['checks'])==inp['checks_count'] and all(c['passed'] for c in inp['checks']) and inp['denominators']==actual_counts)
    failed=[c for c in checks if not c['passed']]
    report={'status':'PASS_DATA_CONTRACT_REVIEW' if not failed else 'BLOCKER_DATA_CONTRACT_REVIEW','reviewer':'Codex /root/retention_direction_review; independent saved-data arithmetic, not independent human labels','checks_count':len(checks),'failed_checks':failed,'denominators':actual_counts,'per_query':per_query,'supported_targets':d['supported_targets'],'training_groups':len(trains),'training_steps':d['training_steps'],'group_weight_scope':'All source-query groups have weight1; broad duplicate has loss0. Optimizer class weighting is covered by the separate code review, not certified by this data audit.','overlap_summary':{'pairs':len(overlaps),'all_single_endpoint':all(x['overlap'][0]==x['overlap'][1] for x in overlaps),'negative_contains_entire_positive':sum(x['negative_contains_entire_positive'] for x in overlaps),'interpretation':'Weak positive bags may share a held endpoint with a local negative. No dense positive endpoint label is created.'},'new_local_positive_time_support':new_positive_support,'limitations':['Stored native axes independently match input and shape/relation caches. Actual source presentation timestamps were decoded by the pinned builder; this audit does not claim a second media decode.','Only structural TCN support and saved RGB support time/outside consistency rechecked here; encoder interpolation value provenance is covered by the pinned preparation checks.','Same Codex direct physical labels, repeated DEV, no whole-clause completion or educational-error truth.','Single unknown FIT1567 rise05 remains excluded from loss. New1567 rise/fall local positives are nonexclusive weak bags, not dense1.'],'pins':{str(U/'DATASET_FROZEN.json'):sha(U/'DATASET_FROZEN.json'),str(U/'INPUT_CHECKS.json'):sha(U/'INPUT_CHECKS.json'),str(Path(__file__)):sha(__file__)},'all_dataset_pins_rehashed':len(d['pins']),'data_ready_for_one_frozen_training_run':not failed,'separate_code_review_required':True,'execution':{'model_imports':0,'model_loads':0,'model_forwards':0,'training':0,'media_visual_reviews':0,'label_changes':0,'root_document_changes':0}}
    with (U/'DATA_REVIEW.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':report['status'],'checks':len(checks),'failures':failed,'report_sha256':sha(U/'DATA_REVIEW.json')},ensure_ascii=False))
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
