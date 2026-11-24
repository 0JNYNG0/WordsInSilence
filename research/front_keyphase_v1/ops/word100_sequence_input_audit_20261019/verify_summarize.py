"""Separate saved-value recount; no model loading or inference."""
import json,hashlib,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
U=Path(__file__).resolve().parent;B=U.parent/'word100_body_boundary_20261017'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
scope=read(U/'SCOPE.json');done=read(U/'COMPLETE.json');assert done['scope_sha256']==sha(U/'SCOPE.json')
ctx=read(B/'training_CONTEXT.json');labels=ctx['labels'];rows=[];score_n=0;maxerr=0;views=defaultdict(list);word=defaultdict(list)
for cid,h in done['record_hashes'].items():
    path=U/'records'/f'{cid}.json';assert sha(path)==h;r=read(path);assert r['scope_sha256']==sha(U/'SCOPE.json');rows.append(r)
    rec=read(B/'data/records'/f'{cid}.json');y=r['label_index'];assert r['label']==labels[y]
    assert len(r['predictions'])==10 and len({(p['view'],p['condition']) for p in r['predictions']})==10
    for p in r['predictions']:
        for model in ('C','T3F'):
            s=p[model];z=np.array(s['logits'],dtype='float64');assert z.shape==(100,) and np.isfinite(z).all()
            order=sorted(range(100),key=lambda j:(-z[j],j));rank=order.index(y)+1
            probability=100/math.fsum(math.exp(float(t-z[y])) for t in z)
            margin=z[y]-max(v for j,v in enumerate(z) if j!=y)
            assert s['target_rank']==rank and s['top1_index']==order[0] and s['top1_label']==labels[order[0]]
            maxerr=max(maxerr,abs(probability-s['target_score']),abs(margin-s['target_margin']));assert maxerr<1e-9;score_n+=1
    for s in r['input_stats']:
        v=s['view']
        with np.load(rec['views'][v]['channel_file'],allow_pickle=False) as z,np.load(rec['views'][v]['observation'],allow_pickle=False) as obs:
            ids=list(map(int,obs['frame_id']));lookup={val:k for k,val in enumerate(ids)};ix=np.array([lookup[int(f)] for f in z['whole_frame_id']])
            assert np.all(np.diff(ix)==1) and np.array_equal(z['whole_ts'],obs['timestamp_us'][ix])
            assert np.array_equal(z['x_whole'][1:,279],np.diff(z['whole_ts']).astype(np.float32)/1e6)
            core=(obs['timestamp_us']>=round(s['annotation_seconds'][0]*1e6))&(obs['timestamp_us']<=round(s['annotation_seconds'][1]*1e6))
            ret=np.zeros(len(ids),bool);ret[ix]=True;support=np.zeros((len(ids),2),bool);local=support.copy()
            for hand in (0,1):
                support[ix,hand]=(z['x_whole'][:,195+21*hand:216+21*hand]>0).any(axis=1)
                local[ix,hand]=(z['x_whole'][:,237+21*hand:258+21*hand]>0).any(axis=1)
            assert s['core_frames']==int(core.sum()) and s['core_retained']==int((core&ret).sum())
            assert s['core_trimmed_frames']==int((core&~ret).sum()) and s['core_any_hand']==int(support[core].any(1).sum())
            for j,name in enumerate(('left','right')):
                assert s[f'core_{name}_any']==int(support[core,j].sum()) and s[f'core_{name}_local']==int(local[core,j].sum())
                assert s[f'core_observed_but_input_{name}_unavailable']==int((core&obs['usable'][:,33+21*j:54+21*j].any(1)&~support[:,j]).sum())
            expected=np.flatnonzero(core&~support.any(1));got=[]
            for run in s['no_hand_runs']:
                indices=list(range(run['start_index'],run['end_index']+1));got.extend(indices)
                assert run['first_frame']==ids[indices[0]] and run['last_frame']==ids[indices[-1]] and run['frames']==len(indices)
                assert abs(run['seconds']-(obs['timestamp_us'][indices[-1]]-obs['timestamp_us'][indices[0]])/1e6)<1e-9
            assert got==list(expected)
        views[v].append(s)
    word[r['word_code']].append(r)
assert len(rows)==899 and len(word)==100 and score_n==17980
tables=[];perword=[]
for view in ('masked','sensor'):
    for model in ('C','T3F'):
        for condition in scope['conditions']:
            samples=[]
            for r in rows:
                ps={p['condition']:p[model] for p in r['predictions'] if p['view']==view};samples.append((ps['original'],ps[condition]))
            base=[(a,b) for a,b in samples if a['target_rank']==1]
            tables.append({'view':view,'model':model,'condition':condition,'total':len(samples),
                'original_top1':len(base),'target_top1':sum(b['target_rank']==1 for a,b in samples),
                'top1_retained_of_original_correct':sum(b['target_rank']==1 for a,b in base),
                'retention_pct':100*sum(b['target_rank']==1 for a,b in base)/len(base),
                'median_score_delta':float(np.median([b['target_score']-a['target_score'] for a,b in samples])),
                'median_margin_delta':float(np.median([b['target_margin']-a['target_margin'] for a,b in samples]))})
for wc,rs in word.items():
    d={'word_code':wc,'label':rs[0]['label'],'clips':len(rs),'conditions':{}}
    for view in ('masked','sensor'):
        for model in ('C','T3F'):
            for cond in scope['conditions']:
                pairs=[]
                for r in rs:
                    ps={p['condition']:p[model] for p in r['predictions'] if p['view']==view};pairs.append((ps['original'],ps[cond]))
                base=[(a,b) for a,b in pairs if a['target_rank']==1]
                d['conditions'][f'{view}/{model}/{cond}']={'original_top1':len(base),'target_top1':sum(b['target_rank']==1 for a,b in pairs),
                    'retained':sum(b['target_rank']==1 for a,b in base),'margin_delta_median':float(np.median([b['target_margin']-a['target_margin'] for a,b in pairs]))}
    perword.append(d)
input_summary={}
for view,ss in views.items():
    total=sum(s['core_frames'] for s in ss);retained=sum(s['core_retained'] for s in ss);hands=sum(s['core_any_hand'] for s in ss)
    input_summary[view]={'clips':len(ss),'annotation_frames':total,'retained_frames':retained,'retained_pct':100*retained/total,
        'trimmed_frames':total-retained,'trimmed_clips':sum(s['core_trimmed_frames']>0 for s in ss),
        'no_internal_deletion':899,'any_hand_frames':hands,'any_hand_pct':100*hands/total,
        'both_hand_absence_frames':total-hands,'clips_any_both_hand_absence':sum(s['core_frames']>s['core_any_hand'] for s in ss),
        'clips_both_hand_gap_ge_0_3':sum(s['longest_no_hand_seconds']>=.3 for s in ss),
        'local_absence_despite_global_left':sum(s['core_left_any']-s['core_left_local'] for s in ss),
        'local_absence_despite_global_right':sum(s['core_right_any']-s['core_right_local'] for s in ss),
        'observed_left_lost_at_input':sum(s['core_observed_but_input_left_unavailable'] for s in ss),
        'observed_right_lost_at_input':sum(s['core_observed_but_input_right_unavailable'] for s in ss)}
for p,h in scope['pins'].items():assert sha(p)==h,p
write(U/'SUMMARY.json',{'scope':'899 TRAIN /100 words /single frozen seed; altered clips have no correctness labels','metrics':tables,'per_word':perword,'input':input_summary})
write(U/'INDEPENDENT.json',{'status':'PASS','method':'Separate script saved-logit arithmetic and NPZ/frame/mask recount; same Codex operator, no second agent or sign-language expert','clips':899,'words':100,'logit_vectors':score_n,'logit_values':score_n*100,'maximum_scalar_difference':maxerr,'input_views_checked':1798,'source_pins_unchanged':len(scope['pins']),'training_runs':0,'new_inference':0,'script_sha256':sha(__file__),'summary_sha256':sha(U/'SUMMARY.json')})
print(json.dumps({'metrics':tables,'input':input_summary},ensure_ascii=False,indent=2))
