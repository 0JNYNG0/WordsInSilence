"""Independent saved-score recount and frozen-weight/training trace audit. No training."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch

U=Path(__file__).resolve().parent;B=U.parents[1];R=B/'runs/rgb_tcn_development_pair_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def main():
    plan=read(U/'FROZEN_COMPARISON.json');rec=read(R/'FROZEN_MODELS.json');done=read(R/'EVALUATION_COMPLETE.json')
    assert done['scores_sha256']==sha(R/'EVALUATION_SCORES.npz') and done['models_sha256']==sha(R/'FROZEN_MODELS.json')
    assert all(sha(p)==h for p,h in read(R/'SOURCE_PINS.json').items())
    parent=torch.load(B/'runs/partial_encoder_pair_v1/checkpoint_control_final.pt',map_location='cpu',weights_only=True)
    arrays=load(R/'EVALUATION_SCORES.npz');meta=read(R/'EVALUATION_ROWS.json');rows=meta['rows'];classes=meta['classes'];assert classes==parent['classes']
    assert len(rows)==735 and arrays['y'].tolist()==[classes.index(r['label']) for r in rows]
    old=read(B/'runs/stgcn_localshape_v1/EVALUATION_ROWS.json');oldmap={(r['group'],r['id']):r for r in old['rows']}
    for j,r in enumerate(rows):
        oldrow=oldmap['ksl36' if r['group'].startswith('ksl') else r['group'],r['id']]
        assert np.array_equal(arrays['baseline'][j],oldrow['baseline_scores'])
    for k in ('A','B','baseline'):assert arrays[k].shape==(735,93) and np.isfinite(arrays[k]).all()
    traces={}
    for r in rec['models']:
        assert sha(r['path'])==r['sha256'];ck=torch.load(r['path'],map_location='cpu',weights_only=True)
        assert ck['steps']==660 and ck['config_sha256']==sha(U/'FROZEN_COMPARISON.json') and ck['parent_sha256']==plan['parent']
        st=ck['state_dict'];changed=[]
        for k,v in parent['state_dict'].items():
            if not torch.equal(v,st['base.'+k]):changed.append(k)
        assert set(changed)=={'head.weight','head.bias'}
        assert bool(st['rgb_projection.weight'].any())==(r['arm']=='B')
        loss=load(R/('losses_'+r['arm']+'.npz'))['losses'];assert loss.shape==(660,4) and np.isfinite(loss).all()
        assert np.allclose(loss[:,:3].sum(1),loss[:,3],atol=1e-6,rtol=1e-7)
        traces[r['arm']]={'changed_base_tensors':changed,'rgb_projection_nonzero':bool(st['rgb_projection.weight'].any()),'steps':len(loss),'first_total_loss':float(loss[0,3]),'last_total_loss':float(loss[-1,3])}
    # Both start at the same classifier with zero RGB residual on identical draws.
    assert abs(traces['A']['first_total_loss']-traces['B']['first_total_loss'])<1e-7
    schedule=load(R/'SCHEDULE.npz');assert sha(R/'SCHEDULE.npz')==rec['schedule_sha256']
    for k,n in [('replay',2346),('paired',131)]:assert schedule[k].shape==(660,16) and ((schedule[k]>=0)&(schedule[k]<n)).all()
    predictions={k:np.argmax(arrays[k],axis=1) for k in ('baseline','A','B')};y=arrays['y']
    def stats(indices):
        ii=np.array(indices,dtype=int);n=len(ii);out={'n':n}
        for k,p in predictions.items():
            hits=int((p[ii]==y[ii]).sum());out[k]={'correct':hits,'accuracy_percent':100*hits/n}
        for base in ('A','baseline'):
            oldok=predictions[base][ii]==y[ii];newok=predictions['B'][ii]==y[ii]
            out['B_vs_'+base]={'delta_pp':out['B']['accuracy_percent']-out[base]['accuracy_percent'],'improved_ids':[rows[j]['id'] for j in ii[~oldok&newok]],'regressed_ids':[rows[j]['id'] for j in ii[oldok&~newok]],'changed_predictions':int((predictions['B'][ii]!=predictions[base][ii]).sum())}
        return out
    groups={g:stats([i for i,r in enumerate(rows) if r['group']==g]) for g in ('annotation_dev','ksl_primary34','ksl_previous2')}
    groups.update({label:stats([i for i,r in enumerate(rows) if r['group']=='ksl_primary34' and r['label']==label]) for label in ('공부','먹다')})
    groups.update({'signer_'+str(s):stats([i for i,r in enumerate(rows) if r['group']=='annotation_dev' and r['signer_id']==s]) for s in (11,12,13)})
    assert groups['annotation_dev']['baseline']['correct']==691 and groups['ksl_primary34']['baseline']['correct']==22
    method=groups['ksl_primary34']['B']['correct']>groups['ksl_primary34']['A']['correct'] and groups['annotation_dev']['B']['correct']>=groups['annotation_dev']['A']['correct'] and all(groups[w]['B']['correct']>=groups[w]['A']['correct'] for w in ('공부','먹다'))
    project=groups['ksl_primary34']['B']['correct']>22 and groups['annotation_dev']['B']['correct']>=691 and all(groups[w]['B']['correct']>=groups[w]['baseline']['correct'] for w in ('공부','먹다'))
    appearance=load(R/'APPEARANCE_SCORES.npz');adone=read(R/'APPEARANCE_COMPLETE.json');assert sha(R/'APPEARANCE_SCORES.npz')==adone['score_sha256']
    indices=[next(i for i,r in enumerate(rows) if r['id']==sid) for sid in appearance['ids']];assert len(indices)==34
    assert np.array_equal(appearance['y'],y[indices]);ap=[];natural=predictions['B'][indices];hits=int((natural==y[indices]).sum())
    for name,sc in zip(appearance['variants'],appearance['scores']):
        assert sc.shape==(34,93) and np.isfinite(sc).all();pred=sc.argmax(1);correct=int((pred==y[indices]).sum());flips=int((pred!=natural).sum())
        ap.append({'variant':str(name),'correct':correct,'n':34,'accuracy_percent':100*correct/34,'top1_changes':flips,'agreement_percent':100*(34-flips)/34,'max_logit_change':float(np.max(np.abs(sc-arrays['B'][indices]))),'frozen_screen_pass':correct>=hits-1 and flips<=1})
    result={'status':'VERIFIED','groups':groups,'method_gate_pass':method,'project_gate_pass':project,'appearance':ap,'appearance_screen_pass':all(a['frozen_screen_pass'] for a in ap),'trace':traces,'source_hashes_unchanged':True,'automatic_adoption':False,'training_runs':2,'settings_search':0,'scope':'KSL repeated two-word development evaluation and699 supplied-coordinate regression only; no new-person/form accuracy proof'}
    with (U/'VERIFIED_RESULT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'groups':{g:{k:v for k,v in m.items() if k in ('n','baseline','A','B')} for g,m in groups.items()},'method':method,'project':project,'appearance':ap},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
