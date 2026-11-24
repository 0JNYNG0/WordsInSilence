"""Independent scalar arithmetic over stored logits; no model inference/training."""
from pathlib import Path
import json,math,hashlib,statistics as st
import numpy as np
U=Path(__file__).resolve().parent;B=U.parent/'word100_body_boundary_20261017'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'INDEPENDENT.json').exists()
    a=read(U/'ANALYSIS.json');cor=read(U/'ORDER_CLOCK_CORRECTION.json');vectors=0;max_error=0
    for row in a['predictions']+cor['predictions']+a['saved_DEV']:
        scores=[row[m] for m in ('C','T3F')] if 'C' in row else [row]
        for s in scores:
            z=s['logits'];assert len(z)==100 and all(math.isfinite(x) for x in z)
            order=sorted(range(100),key=lambda i:(-z[i],i));p=[math.exp(x-max(z)) for x in z];den=math.fsum(p)
            for key,idx in [('target',row['label_index']),('other',row['other_index'])]:
                assert s[key+'_rank']==order.index(idx)+1
                err=abs(s[key+'_score']-p[idx]/den*100);max_error=max(max_error,err);assert err<1e-9
            assert abs(s['target_minus_other_logit']-(z[row['label_index']]-z[row['other_index']]))<1e-9
            vectors+=1
    assert vectors==648
    for m in a['metadata']:
        r=read(B/'data/records'/f"{m['clip_id']}.json");assert r['role']=='train_candidate' and 1<=r['signer_id']<=9
        with np.load(r['views'][m['view']]['observation'],allow_pickle=False) as z:
            with np.load(r['views'][m['view']]['channel_file'],allow_pickle=False) as c:
                ids=c['whole_frame_id'];idx=[int(np.where(z['frame_id']==i)[0][0]) for i in ids]
                assert len(idx)==m['active_frames'];assert abs((int(c['whole_ts'][-1])-int(c['whole_ts'][0]))/1e6-m['active_seconds'])<1e-8
                for side,lo,hi in [('left',33,54),('right',54,75)]:
                    assert sum(any(z['usable'][i,lo:hi]) for i in idx)==m[side+'_any']
                    assert sum(all(z['usable'][i,lo:hi]) for i in idx)==m[side+'_all']
    phase_max=0
    with np.load(U/'hidden_summaries.npz',allow_pickle=False) as z:
        ww=z['pair_phase_weights'].reshape(3,32)
        for d in a['phase_contributions']:
            key=d['clip_id']+'_'+d['view'];h=z[key+'_h'].astype(float);w=z[key+'_weights'].astype(float)
            sign=1 if d['label']=='믿다' else -1
            g=[math.fsum(row)/h.shape[1] for row in h]
            delta=[]
            for k in range(3):
                den=math.fsum(w[:,k]);part=[math.fsum(float(w[t,k])*h[j,t] for t in range(h.shape[1]))/den for j in range(32)]
                delta.append(math.fsum(sign*float(ww[k,j])*(part[j]-g[j]) for j in range(32)))
            err=max(abs(x-y) for x,y in zip(delta,d['early_middle_late_delta']));phase_max=max(phase_max,err);assert err<1e-4
            assert abs(d['base_margin']+math.fsum(delta)-d['final_margin'])<1e-4
    for p,h in a['pins'].items():assert sha(p)==h,p
    for v in read(U/'VISUAL_EVIDENCE.json'):assert sha(v['video'])==v['source_sha256']==v['video_sha256']
    valid=[r for r in a['predictions'] if r['condition']!='reverse_thirds']+cor['predictions']
    assert len(valid)==252 and len({(r['clip_id'],r['view'],r['condition']) for r in valid})==252
    orig={(r['clip_id'],r['view']):r for r in valid if r['condition']=='original'}
    summaries=[]
    for view in ('sensor','masked'):
        for label in ('믿다','사육장'):
            for condition in ('original','prefix_half','suffix_half','reverse_thirds','double_sequence','dt_x2','dt_x0.5'):
                rows=[r for r in valid if r['view']==view and r['label']==label and r['condition']==condition]
                assert len(rows)==9
                s={'view':view,'label':label,'condition':condition,'n':9}
                for model in ('C','T3F'):
                    s[model]={'original_label_top1':sum(r[model]['target_rank']==1 for r in rows),
                       'target_score_min':min(r[model]['target_score'] for r in rows),
                       'target_score_median':st.median(r[model]['target_score'] for r in rows),
                       'target_score_max':max(r[model]['target_score'] for r in rows),
                       'pair_logit_margin_delta_median':st.median(r[model]['target_minus_other_logit']-orig[r['clip_id'],view][model]['target_minus_other_logit'] for r in rows)}
                summaries.append(s)
    result={'status':'PASS','method':'Separate Python math arithmetic; NumPy for stored arrays only, no torch/model inference.',
      'score_vectors':vectors,'stored_logits':vectors*100,'score_max_error':max_error,'phase_scalar_max_error':phase_max,
      'source_and_model_pins_unchanged':len(a['pins']),'metadata_recomputed':len(a['metadata']),
      'valid_diagnostic_pairs':len(valid),'first_reverse_condition_superseded':36,
      'saved_DEV_unique_clips':len(set(r['clip_id'] for r in a['saved_DEV'])),
      'saved_DEV_top1':sum(r['target_rank']==1 for r in a['saved_DEV']),'saved_DEV_predictions':len(a['saved_DEV']),
      'summaries':summaries,'independent_human_or_agent':False}
    (U/'INDEPENDENT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='summaries'},ensure_ascii=False))
if __name__=='__main__':main()
