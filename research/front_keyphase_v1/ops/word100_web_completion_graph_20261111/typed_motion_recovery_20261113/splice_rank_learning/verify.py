"""Read-only saved input verification. No model inference or target regeneration."""
from pathlib import Path
import collections, hashlib, json
import numpy as np
U = Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (U/'VERIFICATION.json').exists()
    cfg=read(U/'FROZEN.json'); meta=read(U/'INPUTS_COMPLETE.json')
    checks=collections.Counter()
    def require(ok, category):
        assert bool(ok), category
        checks[category]+=1
    require(meta['frozen_sha256']==sha(U/'FROZEN.json'),'frozen_contract')
    for p,h in cfg['pins'].items(): require(sha(p)==h,'old_and_code_pin_unchanged')
    sequence={}
    for r in meta['sequences']:
        require(sha(r['path'])==r['sha256'],'query_pin')
        z=np.load(r['path']); n=len(z['x']); m=z['baseline_index']; order=z['source_index']
        rr=cfg['inputs'][r['clip_id']]['source']; pts=np.load(rr['path'])['timestamp_us']
        require(np.array_equal(np.sort(order),np.arange(n)),'every_source_frame_once')
        require(np.array_equal(z['source_pts'],pts[order]),'source_pts_provenance')
        require(np.array_equal(m,n-1-order if r['reverse'] else order),'baseline_orientation')
        require(np.all(np.diff(z['timestamp_us'])>0) and z['timestamp_us'][0]==0,'monotone_synthetic_pts')
        require(np.array_equal(z['x'][:,279],np.r_[0,np.diff(z['timestamp_us']).astype(np.float32)/1e6]),'real_dt_channel')
        joins=np.flatnonzero(np.diff(m)!=1)+1
        require(np.array_equal(joins,z['join_right']) and len(joins)==3,'exact_provenance_joins')
        internal=np.diff(m)==1
        require(np.array_equal(np.diff(z['timestamp_us'])[internal],np.abs(np.diff(z['source_pts']))[internal]),'internal_pts_intervals')
        require(np.all(np.diff(z['timestamp_us'])[~internal]==r['join_interval_us']),'join_median_interval')
        require(np.isfinite(z['x']).all(),'query_finite')
        if r['view']!='actual_crop':
            ch=np.load(rr['channel']); key='full' if r['view']=='full' else 'feedback' if 'feedback'in ch.files else 'upper_mask'
            require(np.array_equal(z['x'][:,:279],ch[key][order,:279]),'unaltered_observation_channels')
        sequence[r['clip_id'],r['view'],r['reverse']]=z
    for (cid,view,rev),z in sequence.items():
        if not rev:
            zz=sequence[cid,view,True]
            require(np.array_equal(z['x'][::-1,:279],zz['x'][:,:279]),'full_reverse_coordinates')
            require(np.array_equal(z['timestamp_us'][-1]-z['timestamp_us'][::-1],zz['timestamp_us']),'full_reverse_intervals')
    totals=collections.Counter(); per_source=[]
    for r in meta['bags']:
        require(sha(r['target_debug'])==r['target_debug_sha256'],'target_pin')
        t=np.load(r['target_debug']); q=sequence[r['clip_id'],r['view'],r['reverse']]
        m=q['baseline_index']; original=np.load(r['source_sequence']); j=cfg['family_order'].index(r['family'])
        spans=t['spans']; a,b=spans.T
        require(np.array_equal(spans,np.array([(aa,bb)for aa in range(len(m)-3)for bb in range(aa+3,len(m))],np.int32)),'same_complete_enumerator')
        crossing=np.array([np.any(np.diff(m[aa:bb+1])!=1)for aa,bb in spans])
        require(np.array_equal(crossing,t['crossing']),'crossing_exact_all_candidates')
        require(np.array_equal(t['negative'],t['eligible']&crossing),'only_extent_crossing_negatives')
        require(np.array_equal(t['rank_mask'],t['positive']|t['negative']),'explicit_rank_mask')
        require(not np.any(t['partial_audit']&t['rank_mask']),'partial_outside_rank')
        require(np.array_equal(t['inherited_y'],original['y'][m,j])and np.array_equal(t['inherited_mask'],original['mask'][m,j]),'no_progress_label_rewrite')
        require(np.array_equal(t['inherited_bands'],original['bands'][m,j])and np.array_equal(t['inherited_valid'],original['valid'][m,j]),'old_bands_valid_unchanged')
        old=np.load(r['original_replay_bag']); old_positive={tuple(v)for v in old['spans'][old['y']]}
        expected=np.array([not cross and (int(m[aa]),int(m[bb]))in old_positive for (aa,bb),cross in zip(spans,crossing)])
        require(np.array_equal(t['positive'],expected),'all_and_only_inherited_old_full_positives')
        selected=t['selected']; pos=t['positive'][selected]; neg=~pos
        require(np.all(t['rank_mask'][selected]),'unknown_never_selected')
        require(int(pos.sum())==r['positive'] and int(neg.sum())==r['negative'],'selected_counts')
        partners=t['negative_partners']; ns=spans[selected[neg]]; ps=spans[partners]
        require(np.array_equal(ns[:,1]-ns[:,0],ps[:,1]-ps[:,0]),'negative_duration_partners')
        require(np.all(t['eligible'][partners]&~t['crossing'][partners]),'partner_contained_no_new_label')
        if r['skipped']:
            require(r['positive']==0 or r['negative']==0,'empty_positive_safe_hold')
            require('path'not in r,'held_has_no_training_bag')
        else:
            require(sha(r['path'])==r['sha256'],'bag_pin')
            bag=np.load(r['path'])
            require(np.array_equal(bag['spans'],spans[selected])and np.array_equal(bag['y'],pos),'bag_target_alignment')
            require(np.all(bag['rank_mask'])and np.array_equal(bag['crossing'],neg),'bag_rank_mask')
            require(bag['local_full'].shape==(len(selected),1160)and np.isfinite(bag['local_full']).all(),'finite_candidate_features')
        for k in ['positive','negative','partial_audit']:totals[k]+=int(t[k].sum())
        per_source.append({k:r[k]for k in ['clip_id','view','reverse','family','original_positive_count','positive','negative','skipped']})
    require(len(meta['replay'])==178,'all_original_fit_bags_replayed')
    require({r['person']for r in meta['replay']}==set(range(2,8)),'fit_people_only')
    require(set(r['family']for r in meta['replay'])=={2,3,4,5,7},'all_original_families')
    require(meta['no_optimizer_or_training']and not list(U.glob('runs/**/*model*')),'no_training')
    out=dict(status='SAVED_INPUT_CONTRACT_PASS_NO_TRAINING',self_verifier='Codex subagent; parent review pending',
        total_checks=sum(checks.values()),checks=dict(checks),full_grid_totals=dict(totals),
        source_family_positive_counts=per_source,held_bags=len(meta['held_bags']),
        claim='Only synthetic non-single-occurrence extent labels and retained original source targets. No accuracy or KSL correctness claim.')
    (U/'VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(out['status'],out['total_checks'],'checks; held',out['held_bags'])

if __name__=='__main__':main()
