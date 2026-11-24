"""Independent saved-array derivation of rate audit positives; no model calls."""
from pathlib import Path
import collections,json,hashlib
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not(U/'SAVED_VERIFICATION.json').exists()
    cfg=read(U/'FROZEN.json');audit=read(U/'INPUT_VALIDITY_AUDIT.json');checks=collections.Counter()
    def ck(ok,name):assert bool(ok),name;checks[name]+=1
    for p,h in cfg['pins'].items():ck(sha(p)==h,'frozen_pin')
    ck(sha(U/'FROZEN.json')==audit['frozen_sha256'],'frozen_manifest')
    key=lambda r:(r['clip_id'],r['view'],r['reverse'],r['family'])
    old={key(r):r for r in cfg['replay_original']};splice={key(r):r for r in cfg['replay_splice']}
    qs={r['path']:r for r in audit['sequences']};cache={}
    splice_base={(r['clip_id'],r['view'],r['reverse']):r for r in cfg['splice_sequences']}
    for path,r in qs.items():
        ck(sha(path)==r['sha256'],'raw_pack_pin');q=np.load(path);raw=np.load(r['raw_source']);ids=q['source_index']
        ck(np.array_equal(q['xy_px'],raw['xy_px'][ids])and np.array_equal(q['score'],raw['score'][ids]),'saved_raw_equal_original')
        ck(np.array_equal(q['source_frame_id'],raw['frame_id'][ids])and np.array_equal(q['source_timestamp_us'],raw['timestamp_us'][ids]),'saved_source_time_ids')
        ck(np.all(np.diff(q['timestamp_us'])>0)and np.array_equal(q['frame_id'],np.arange(len(ids))),'saved_query_clock')
        ck(np.all(np.diff(q['base_query_index'])>=0),'saved_monotone_base_map')
        ck(not any(k in q.files for k in ['tcn280','x','local_full','features','hidden']),'no_encoder_features')
        ck(r['unique_source_frames']==len(np.unique(ids))and r['omitted_source_frames']==r['source_frames']-len(np.unique(ids)),'saved_sampling_denominator')
        cache[path]={k:q[k]for k in ['baseline_index','source_block_id','base_query_index']}
    expected_summary=collections.defaultdict(lambda:dict(total=0,viable=0,held=0,positive=0))
    for r in audit['cases']:
        ck(sha(r['label_path'])==r['label_sha256'],'label_pin');d=np.load(r['label_path']);q=cache[r['raw_path']]
        sr=qs[r['raw_path']];source=np.load(sr['source_sequence']);j=cfg['family_order'].index(r['family'])
        m=q['baseline_index'];block=q['source_block_id'];valid=source['valid'][:,j].any(1)
        witness=np.flatnonzero((source['mask'][:,j]>0)&(source['y'][:,j]>0));missing=np.setdiff1d(witness,np.unique(m))
        ck(np.array_equal(missing,d['missing_witnesses']),'lost_witness_ids')
        parent=(old if r['base_kind']=='normal'else splice)[key(r)];p=np.load(parent['path'])
        originals=p['spans'][p['y']]
        if r['base_kind']=='splice':
            base=np.load(splice_base[r['clip_id'],r['view'],r['reverse']]['path'])
            originals=base['baseline_index'][originals]
        expected=set()
        # Enumerate inherited endpoint preimages, not implementation's full-grid formula.
        if not len(missing):
            for sa,sb in originals:
                for a in np.flatnonzero(m==sa):
                    for b in np.flatnonzero(m==sb):
                        if b<a+3 or block[a]!=block[b]:continue
                        observed=valid[m[a:b+1]]
                        if not(observed[0]and observed[-1]and observed.mean()>=.8):continue
                        if len(np.unique(m[a:b+1][observed]))<4:continue
                        expected.add((int(a),int(b)))
        actual={tuple(v)for v in d['spans'][d['positive']]}
        ck(actual==expected,'positive_preimage_exact')
        ck(len(actual)==r['positive'],'positive_count')
        ck(bool(not r['positive']or not r['negative'])==r['held'],'full_rank_hold_contract')
        ck(not np.any(d['positive']&d['negative']),'rank_labels_disjoint')
        if r['base_kind']=='normal':ck(not d['crossing'].any(),'rate_skips_never_join_negatives')
        else:ck(np.array_equal(d['negative'],d['eligible']&d['crossing']),'splice_only_join_negatives')
        ss=expected_summary[r['rate'],r['base_kind'],r['family']]
        ss['total']+=1;ss['held']+=int(r['held']);ss['viable']+=int(not r['held']);ss['positive']+=len(actual)
    for r in audit['summary']:
        ss=expected_summary[r['rate'],r['base_kind'],r['family']]
        ck((r['total_cases'],r['viable_cases'],r['held_cases'],r['positive_candidates'])==(ss['total'],ss['viable'],ss['held'],ss['positive']),'aggregate_exact')
    ck(len(qs)==144 and len(audit['cases'])==288,'all_cases_accounted')
    ck(not(U/'runs').exists(),'no_training_outputs')
    out=dict(status='SAVED_RATE_AUDIT_PASS_NOT_ENCODER_READY',checks=dict(checks),total_checks=sum(checks.values()),
        source_people=[2,3,4,5,6,7],no_models_loaded=True,no_encoder=True,no_training=True,
        limitation='Inherited source availability only. New guard/canonical channels may change eligibility; not independent human labels.',
        script_sha256=sha(__file__),audit_sha256=sha(U/'INPUT_VALIDITY_AUDIT.json'))
    (U/'SAVED_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print('Saved rate audit PASS',out['total_checks'])
if __name__=='__main__':main()
