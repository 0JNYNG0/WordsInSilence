"""Every research response is checked against saved arrays and original classifier records."""
import study as S
from adapter import PublicFeedbackReplay
from collections import Counter
import numpy as np, traceback

def main():
    f=S.read(S.U/'FROZEN.json');m=S.read(S.U/'MANIFEST.json');d=S.read(S.U/'DONE.json');S.check(m['pins']);S.check(f['protected']);prior={x['query_id']:x for x in f['recognition']};pairs={(x['query'],x['reference']):x for x in d['rows']};reference={r['clip_id']:r for r in f['train']};cache={};api=PublicFeedbackReplay();count=Counter();summary=Counter()
    for row in m['rows']:
        qid=row['query_id'];receipt=S.read(row['receipt']);assert S.sha(row['receipt'])==row['sha256'];assert receipt['predictions']==prior[qid]['predictions'] and receipt['practice_target']['word_code']==prior[qid]['word_code']
        refs=receipt['references'];assert len(refs) in (8,9) and {r['reference_id'] for r in refs}=={rid for q,rid in pairs if q==qid}
        for ref in refs:
            rid=ref['reference_id'];z=S.load(pairs[qid,rid]['output']);assert S.sha(ref['pair_path'])==ref['pair_sha256'] and ref['recorded_id']<=9
            if rid not in cache:cache[rid]=S.load(reference[rid]['output'])
            np.testing.assert_allclose(ref['reference_seconds'],cache[rid]['timestamp_us']/1e6,rtol=0,atol=0)
            for g in ref['groups']:
                h=0 if g['side']=='left' else 1;a,b=(0,15) if g['group']=='angle' else (15,23);mode=g['mode'];st=z[mode+'_status'][:,h,a:b];dom=z['domain'][:,h,a:b];valid=z[mode+'_ok'][:,h,a:b];delta=z[mode+'_delta'][:,h,a:b]
                assert g['domain_cells']==int(dom.sum()) and g['raw_compared_cells']==int(valid.sum())
                assert g['state_counts']=={str(k):int((st==k).sum()) for k in range(1,6)};assert g['supported_cells']==int((st>=4).sum())
                assert not g['within_band_is_correct'] and not g['out_of_band_is_error']
                assert len(g['timeline_counts'])==len(st)
                np.testing.assert_array_equal(np.array(g['timeline_counts']),np.stack([(st==k).sum(1) for k in range(1,6)],1))
                value=delta if mode=='point' else delta[...,:3];expected=float(value[np.isfinite(value)].mean()) if np.isfinite(value).any() else None
                assert g['raw_level_difference']==expected
                # Validate exact interval boundaries from every state, independently of producer intervals().
                for name,code in [('missing_intervals',1),('image_uncertain_intervals',2),('reference_shortage_intervals',3),('difference_intervals',5)]:
                    mask=(st==code).any(1);runs=[];active=None
                    for i,on in enumerate(mask):
                        if on and active is None:active=i
                        if active is not None and (not on or i==len(mask)-1):
                            end=i if on else i-1;t=ref['reference_seconds'];runs.append([round(t[active],3),round(t[end],3)]);active=None
                    assert runs==g[name]
                prefix=mode+'_'+g['group'];summary[prefix+'_domain']+=g['domain_cells'];summary[prefix+'_supported']+=g['supported_cells']
                for k,v in g['state_counts'].items():summary[prefix+'_state'+k]+=v
                count['verified_groups']+=1
            count['verified_references']+=1
        for model in ('C','T3F','E3'):
            response=api.get(qid,model);assert response['prediction']==prior[qid]['predictions'][model];assert response['practice_target']==receipt['practice_target'];assert response['references']==refs
            assert all(response[k] is None for k in ('whole_gesture_correct','validated_correction','skill_score'));assert not response['live_webcam_supported'] and not response['inference_performed'] and not response['training_performed'] and not response['personal_data'];count['responses']+=1
        if count['responses']%150==0:print('VERIFIED_REPLAY_RESPONSES',count['responses'],900,flush=True)
    assert count['responses']==900 and count['verified_references']==2697 and dict(summary)==S.read(S.U/'SUMMARY.json')['counts']
    for query,model in [('USER_PRIVATE_RECORDING','C'),(m['rows'][0]['query_id'],'UNFROZEN_MODEL')]:
        try:api.get(query,model)
        except ValueError:count['unapproved_input_rejected']+=1
        else:raise AssertionError('Unapproved request accepted')
    S.check(f['protected']);S.write(S.U/'RESPONSES_VERIFIED.json',dict(at=S.now(),status='PASS900_REPLAY_RESPONSES',counts=dict(count),pins={str(p):S.sha(p) for p in [S.U/'MANIFEST.json',S.U/'SUMMARY.json',S.U/'adapter.py',__file__]},method='Same Codex separate response/source-array/interval arithmetic. Not a live webcam or independent human evaluation.'))
    print('REPLAY_VERIFIED',dict(count),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=S.U/'FIRST_ERROR_verify_responses.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
