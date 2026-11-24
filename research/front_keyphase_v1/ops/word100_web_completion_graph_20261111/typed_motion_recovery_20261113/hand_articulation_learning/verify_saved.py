"""Independent arithmetic/storage audit. Never imports a model or calls forward.

Root launches only after A evaluation and new-input rotation controls finish.
torch.load is used to inspect saved tensors/buffers, never to instantiate a net.
First audit attempt is preserved. Historical initialization is distinguished
from independently inspectable final buffers and the pinned initialization code.
"""
from pathlib import Path
from collections import defaultdict
import datetime,hashlib,json,time,traceback
import numpy as np
import torch

A=Path(__file__).resolve().parent
R=A.parent/'hand_rotation_invariant_learning'
U=A.parent/'hand_transition_learning'
T=A.parent/'typed_shape_transition_review'
E=A/'saved_evaluation'
ATTRS=['four_open','four_closed','index_only','intermediate_bending']
CHECKS=0

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def ok(value,message):
    global CHECKS
    assert value,message
    CHECKS+=1
def pins(p):
    for file,h in read(p).get('pins',{}).items():ok(sha(file)==h,('pin',file))
def equal(a,b,context):
    if isinstance(b,dict):
        for key,v in b.items():equal(a[key],v,(context,key))
    elif isinstance(b,list):
        ok(len(a)==len(b),context)
        for av,bv in zip(a,b):equal(av,bv,context)
    elif isinstance(b,float):ok(a is not None and abs(float(a)-b)<=1e-10,(context,a,b))
    else:ok(a==b,(context,a,b))
def probability(logits):
    v=np.asarray(logits,np.float64);p=np.exp(v-v.max());return p/p.sum()
def fraction(a,b):return float(a/b) if b else None

def binmetric(rows):
    q=[r for r in rows if r['eligible']]
    counts=np.bincount([2*r['target']+r['prediction'] for r in q],minlength=4)
    tn,fp,fn,tp=map(int,counts)
    return dict(known=len(rows),eligible=len(q),unavailable=len(rows)-len(q),positive=tp+fn,negative=tn+fp,
        tp=tp,tn=tn,fp=fp,fn=fn,precision=fraction(tp,tp+fp),recall=fraction(tp,tp+fn),
        specificity=fraction(tn,tn+fp),accuracy=fraction(tp+tn,len(q)),
        balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None)

def transmetric(rows):
    q=[r for r in rows if r['eligible']];emit=[r for r in q if r['confidence']>=.65]
    matrix=np.zeros((3,3),int)
    for r in q:matrix[r['target'],r['prediction']]+=1
    correct=int(np.trace(matrix));ec=sum(r['target']==r['prediction'] for r in emit)
    return dict(total=len(rows),eligible=len(q),observation_unknown=len(rows)-len(q),confusion=matrix.tolist(),
        class_order=['opening','stable_handshape','closing'],argmax_correct=correct,argmax_accuracy=fraction(correct,len(q)),
        emitted=len(emit),coverage=fraction(len(emit),len(rows)),low_confidence=len(q)-len(emit),
        emitted_correct=ec,emitted_wrong=len(emit)-ec,emitted_accuracy=fraction(ec,len(emit)))

def groups(rows,fields,metric):
    d=defaultdict(list)
    for r in rows:d[tuple(r[f] for f in fields)].append(r)
    return {k:metric(v) for k,v in d.items()}
def checkgroups(stored,rows,fields,metric,context,field='metrics'):
    expected=groups(rows,fields,metric);ok(len(stored)==len(expected),(context,'groupcount'))
    for row in stored:equal(row[field],expected[tuple(row[f] for f in fields)],context)
def kind(w):
    if w['kind']=='synthetic_repeated_pose':return 'synthetic_constant'
    if w['kind']=='direct_stable_handshape':return 'real_stable_closed_arm_raise' if w['body_raise_stable'] else 'real_stable_other'
    return ('full' if w['kind']=='full_state_change' else 'partial')+('_reverse_opening' if w['reverse'] else '_closing')
def pairmetric(rows):
    q=[r for r in rows if r['eligible']]
    return dict(total=len(rows),eligible=len(q),original_correct=sum(r['old_correct'] for r in q),
        new_correct=sum(r['new_correct'] for r in q),repaired=sum(not r['old_correct'] and r['new_correct'] for r in q),
        regressed=sum(r['old_correct'] and not r['new_correct'] for r in q),
        unchanged_correct=sum(r['old_correct'] and r['new_correct'] for r in q),
        unchanged_wrong=sum(not r['old_correct'] and not r['new_correct'] for r in q))


def geometry(x):
    """Independent vector implementation of12descriptors+12masks."""
    p=x[:,:42].reshape(-1,21,2).astype(float);obs=x[:,42:]> .5
    values=[];masks=[]
    for start in [5,9,13,17]:
        j=np.arange(start,start+4);b=np.diff(p[:,j],axis=1);length=np.sqrt((b*b).sum(2))
        for k in [0,1]:
            valid=obs[:,j[k:k+3]].all(1)&(length[:,k]>1e-6)&(length[:,k+1]>1e-6)
            v=np.zeros(len(x));v[valid]=np.clip(-(b[valid,k]*b[valid,k+1]).sum(1)/(length[valid,k]*length[valid,k+1]),-1,1)
            values.append(v);masks.append(valid)
        valid=obs[:,j].all(1)&(length>1e-6).all(1);v=np.zeros(len(x))
        v[valid]=np.linalg.norm(b[valid].sum(1),axis=1)/length[valid].sum(1)
        values.append(np.clip(v,0,1));masks.append(valid)
    return np.c_[np.asarray(values).T,np.asarray(masks).T].astype(np.float32)


def main():
    output=A/'INDEPENDENT_SAVED_REVIEW.json';started=A/'INDEPENDENT_AUDIT_STARTED.json'
    ok(not output.exists() and not started.exists(),'Preserve first audit; investigate any existing failure explicitly')
    for p in [A/'TRAINING_COMPLETE.json',E/'COMPLETE.json',E/'PAIRED_COMPARISON_R.json',E/'PAIRED_COMPARISON_U.json',A/'rotation_controls/COMPLETE.json']:
        ok(p.exists(),('Waiting for root completed outputs',str(p)))
    write(started,dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),
        mode='Saved storage/tensor/arithmetic only; no model import, instantiate, forward or retraining'))
    for p in [A/'FROZEN.json',A/'TRAINING_CONTRACT.json',E/'FROZEN.json',A/'INTEGRATION_DECISION_FROZEN.json',A/'rotation_controls/FROZEN.json']:
        pins(p)
    cfg=read(A/'FROZEN.json');meta=read(A/'INPUTS_COMPLETE.json');rm=read(R/'INPUTS_COMPLETE.json')
    items=read(A/'EVALUATION_ITEMS.json');done=read(A/'TRAINING_COMPLETE.json');review=read(T/'REVIEW_COMPLETE.json')
    ok(items==read(R/'EVALUATION_ITEMS.json')==read(U/'EVALUATION_ITEMS.json'),'All three fixed item/label sets equal')
    for k in ['labels','windows','controls','counts']:ok(meta[k]==rm[k],('Manifest unchanged',k))
    ok(len(done['runs'])==18,'18models');ok({(r['heldperson'],r['seed']) for r in done['runs']}=={(p,s) for p in range(2,8) for s in [42,43,44]},'sixfolds3seeds')
    cache={};seqmeta={}
    for r in meta['sequences']:
        ok(sha(r['path'])==r['sha256'] and sha(r['parent_path'])==r['parent_sha256'],('Input hash',r['clip_id'],r['side']))
        z=dict(np.load(r['path']));parent=dict(np.load(r['parent_path']));cache[r['clip_id'],r['side']]=z;seqmeta[r['clip_id'],r['side']]=r
        ok(np.array_equal(z['x'][:,:63],parent['x']),'Preserved63')
        for k in ['y','mask','available','frames','basis_valid']:ok(np.array_equal(z[k],parent[k]),('Preserved source',k))
        ok(np.allclose(z['x'][:,63:],geometry(parent['x']),atol=2e-6,rtol=1e-6),'Independent projected geometry')
    dense={(r['clip_id'],v['frame']) for r in review['rows'] for v in r['frame_records']}
    flex={(r['clip_id'],v['frame']) for r in review['rows'] for v in r['frame_records'] if v['hand_state']=='flexing'}
    unknown={(r['clip_id'],v['frame']) for r in review['rows'] for v in r['frame_records'] if v['hand_state']=='unknown'}
    ok(len(flex)==13 and len(unknown)==8,'13positive/8unknown')
    for cid,f in unknown:ok(not cache[cid,'R']['mask'][f].any(),'Unknown not negative')
    labelmap=defaultdict(list)
    for r in items['labels']:labelmap[r['clip_id'],r['frame'],r['side']].append(r)
    ar=[];wr=[];run_audit=[]
    for run in done['runs']:
        held,seed=run['heldperson'],run['seed'];folder=Path(run['model']).parent
        ok(read(folder/'COMPLETE.json')==run,'runrecord identity')
        ok(sha(run['model'])==run['model_sha256'] and sha(run['predictions'])==run['predictions_sha256'],'model/predhash')
        parent=read(R/'folds'/str(held)/str(seed)/'COMPLETE.json')
        ok(run['split']==parent['split'],'Parent split exact')
        trainseq=[r for r in meta['sequences'] if r['person']!=held]
        trainpeople=sorted({r['person'] for r in trainseq});ok(held not in trainpeople and trainpeople==run['split']['trainpeople'],'Noheldperson inFIT')
        ok(not set(run['split']['train_source_hashes'])&set(run['split']['held_source_hashes']),'Disjoint sourcehashes')
        state=torch.load(run['model'],map_location='cpu',weights_only=True)
        ps=torch.load(parent['model'],map_location='cpu',weights_only=True)
        ok(set(state)==set(ps)|{'articulation_mean','articulation_std','articulation_residual.weight'},'Only intended model tensors')
        ok(all(torch.isfinite(v).all().item() for v in state.values()),'Finite modeltensors')
        for key in ['mean','std']:ok(torch.equal(state[key],ps[key]),'Inherited63norm unchanged')
        fit=np.concatenate([cache[r['clip_id'],r['side']]['x'][:,63:] for r in trainseq])
        mean=np.zeros(24,np.float32);std=np.ones(24,np.float32);counts=[]
        for j in range(12):
            v=fit[fit[:,12+j]>.5,j];ok(len(v)>0,'Feature has training support');mean[j]=v.mean();std[j]=max(float(v.std()),.05);counts.append(len(v))
        ok(np.array_equal(state['articulation_mean'].numpy(),mean) and np.array_equal(state['articulation_std'].numpy(),std),'Newstats onlytrainpeople')
        init=run['initialization'];ok(init['old_tensors_copied_exactly'] and init['residual_initially_zero'],'Recordedguarded initialization')
        ok(init['parent_sha256']==parent['model_sha256']==cfg['parent_models'][str(held)][str(seed)],'Matchingparentcheckpointhash')
        ok(init['available_training_counts']==counts and init['trainpeople']==trainpeople,'Recordedtrainingstats/sourcepeople')
        equal(init['articulation_mean'],mean.tolist(),'Meanrecord');equal(init['articulation_std'],std.tolist(),'Stdrecord')
        ok(run['log'][-1]['step']==300 and cfg['steps']==300 and cfg['lr']==.0003,'Fixedfinal300/lr')
        p=np.load(run['predictions']);fi=[i for i,r in enumerate(items['frames']) if r['person']==held];wi=[i for i,r in enumerate(items['windows']) if r['person']==held]
        ok(p['frame_ids'].tolist()==fi==run['split']['test_frame_ids'],'Frameindices heldonly')
        ok(p['window_ids'].tolist()==wi==run['split']['test_window_ids'],'Windowindices heldonly')
        ok(p['frame_logits'].shape==(len(fi),4) and p['window_logits'].shape==(len(wi),3),'Logitshapes')
        ok(np.isfinite(p['frame_logits']).all() and np.isfinite(p['window_logits']).all(),'Finite savedlogits')
        for fid,logits in zip(fi,p['frame_logits']):
            fr=items['frames'][fid];cid,f,side=fr['clip_id'],fr['source_frame'],fr['side']
            for target in labelmap[cid,f,side]:
                j=target['attribute'];ar.append(dict(seed=seed,person=held,clip_id=cid,frame=f,side=side,attribute=j,
                    attribute_name=ATTRS[j],target=int(target['target']),prediction=int(logits[j]>=0),logit=float(logits[j]),eligible=bool(target['available']),
                    dense_new=(cid,f) in dense and side=='R',new_intermediate_positive=(cid,f) in flex and side=='R' and j==3))
        for wid,logits in zip(wi,p['window_logits']):
            w=items['windows'][wid];z=cache[w['clip_id'],w['side']];a,b=w['span'];pp=probability(logits);n=b-a+1
            eligible=bool(z['available'][w['anchor']]) if w['kind']=='synthetic_repeated_pose' else bool(z['available'][a] and z['available'][b])
            ok(z['available'][a:b+1].all(),'All prepared windowframes available')
            wr.append(dict(seed=seed,person=held,window_id=wid,clip_id=w['clip_id'],kind=kind(w),reverse=w['reverse'],target=w['target'],
                prediction=int(np.argmax(logits)),confidence=float(pp.max()),eligible=eligible,frames=n,
                duration_group='4..7 frames' if n<=7 else '8..15 frames' if n<=15 else '16..24 frames'))
        run_audit.append(dict(heldperson=held,seed=seed,modelhash_verified=True,split_verified=True,normalization_verified=True,
            parent_copy='Historical initializer assertion + pinnedcode + recordedparenthash; finalnorm buffers independentlyequal. Finallearnedweights are expected to differ after300updates.'))
    ak=lambda r:(r['seed'],r['clip_id'],r['frame'],r['side'],r['attribute'])
    saved={ak(r):r for r in read(E/'ATTRIBUTE_ROWS.json')};ok(len(ar)==len(saved)==2598,'2598maskedcells')
    for r in ar:equal(saved[ak(r)],r,'Frameprediction reconstruction')
    wz=dict(np.load(E/'WINDOW_DECISIONS.npz'));wm={(int(s),int(w)):i for i,(s,w) in enumerate(zip(wz['seed'],wz['window_id']))}
    ok(len(wr)==len(wm)==23316,'23316dependentwindows')
    for r in wr:
        i=wm[r['seed'],r['window_id']]
        for k in wz:equal(wz[k][i].item(),r[k],('Windowreconstruction',k))
    result=read(E/'RESULTS.json')
    for name,fields,rows in [('by_seed',['seed'],ar),('by_seed_attribute',['seed','attribute_name'],ar),('by_seed_person_attribute',['seed','person','attribute_name'],ar),
        ('dense_new_by_seed_attribute',['seed','attribute_name'],[r for r in ar if r['dense_new']]),
        ('intermediate_by_seed',['seed'],[r for r in ar if r['attribute']==3]),('new13_positive_by_seed',['seed'],[r for r in ar if r['new_intermediate_positive']])]:
        checkgroups(result['attributes'][name],rows,fields,binmetric,name)
    for name,fields in [('by_seed',['seed']),('by_seed_kind',['seed','kind']),('by_seed_person_kind',['seed','person','kind']),('by_seed_kind_duration',['seed','kind','duration_group'])]:
        checkgroups(result['windows'][name],wr,fields,transmetric,name)
    cr=read(E/'CANONICAL_SOURCE_ROWS.json');byw={(r['seed'],r['window_id']):r for r in wr}
    representative={}
    for source in review['rows']:
        rr=source['frame_records'];raised=[f['frame'] for f in rr if f['body_motion']=='forearm_raising' and f['hand_state']=='closed']
        opened=[f['frame'] for f in rr if f['hand_state']=='open'];closed=[f['frame'] for f in rr if f['hand_state']=='closed' and f['frame']<min(raised)]
        representative[source['clip_id'],'canonical_full_closing']=[min(opened),max(closed)]
        representative[source['clip_id'],'canonical_reversed_opening']=[min(opened),max(closed)]
        representative[source['clip_id'],'canonical_closed_arm_raise']=[min(raised),max(raised)]
    for r in cr:
        if not r['available']:ok(not r['passed'],'Unavailable notpass');continue
        ok(r['source_frames']==representative[r['clip_id'],r['kind']],'Independent label-only canonicalrange')
        d=byw[r['seed'],r['window_id']]
        ok(r['prediction']==d['prediction'] and r['confidence']==d['confidence'],'Canonical savedwindow lookup')
        ok(r['passed']==(d['eligible'] and d['confidence']>=.65 and d['prediction']==r['target']),'Canonical gate')
    ok(len(cr)==54 and result['canonical_gate']['passed_cases']==sum(r['passed'] for r in cr),'Canonicaldenominator')
    pairaudit=[];regressions=[]
    for old,tag in [(R,'R'),(U,'U')]:
        pa=read(E/f'PAIRED_COMPARISON_{tag}.json');oa={ak(r):r for r in read(old/'saved_evaluation/ATTRIBUTE_ROWS.json')}
        apar=[]
        for r in ar:
            q=oa[ak(r)];ok(q['target']==r['target'] and q['eligible']==r['eligible'],'Pairedattr exacttargets')
            apar.append(dict(seed=r['seed'],person=r['person'],attribute=r['attribute_name'],side=r['side'],eligible=r['eligible'],old_correct=q['prediction']==r['target'],new_correct=r['prediction']==r['target']))
            if q['prediction']==r['target'] and r['prediction']!=r['target']:regressions.append(dict(comparison=tag,type='attribute',case=r,old_prediction=q['prediction']))
        for name,fields in [('attributes_by_seed',['seed']),('attributes_by_seed_attribute',['seed','attribute']),('attributes_by_seed_person_attribute',['seed','person','attribute'])]:
            checkgroups(pa[name],apar,fields,pairmetric,name,'counts')
        oz=dict(np.load(old/'saved_evaluation/WINDOW_DECISIONS.npz'));ow={(int(s),int(w)):i for i,(s,w) in enumerate(zip(oz['seed'],oz['window_id']))};wpar=[]
        for r in wr:
            j=ow[r['seed'],r['window_id']];w=items['windows'][r['window_id']]
            ok(oz['target'][j]==r['target'] and oz['eligible'][j]==r['eligible'],'Pairedwindow exacttargets')
            oldcorrect=bool(oz['prediction'][j]==r['target']);newcorrect=r['prediction']==r['target']
            wpar.append(dict(seed=r['seed'],person=r['person'],kind=w['kind']+('_closed_arm_raise' if w.get('body_raise_stable') else ''),
                reverse=r['reverse'],eligible=r['eligible'],old_correct=oldcorrect,new_correct=newcorrect))
            if oldcorrect and not newcorrect:regressions.append(dict(comparison=tag,type='window',case=r,source_span=w['span'],old_prediction=int(oz['prediction'][j])))
        for name,fields in [('windows_by_seed',['seed']),('windows_by_seed_kind',['seed','kind','reverse']),('windows_by_seed_person_kind',['seed','person','kind','reverse'])]:
            checkgroups(pa[name],wpar,fields,pairmetric,name,'counts')
        pairaudit.append(dict(parent=tag,attribute_cells=len(apar),windows=len(wpar),attributes=groups(apar,['seed'],pairmetric),windows_counts=pairmetric(wpar)))
    controls=read(A/'rotation_controls/CASE_RESULTS.json');rc=read(A/'rotation_controls/RESULTS.json');oldcontrols=read(R/'rotation_controls/CASE_RESULTS.json')
    ck=lambda r:(r['person'],r['seed'],r['clip_id'],r['kind'],r['angle'])
    expected={(*key,angle) for key in [(r['person'],r['seed'],r['clip_id'],r['kind']) for r in cr] for angle in [-90,-60,-30,30,60,90]}
    ok({ck(r) for r in controls}==expected and len(controls)==324,'All324rotationcases retained')
    for r in controls:
        if r['eligible']:
            p=probability(r['logits']);ok(np.allclose(p,r['probabilities'],atol=1e-12),'Rotation probabilities fromsavedlogits')
            ok(r['prediction']==int(p.argmax()) and abs(r['confidence']-p.max())<1e-12,'Rotation classification')
            ok(r['emitted']==(r['confidence']>=.65),'Rotation emission')
        else:ok(r['prediction'] is None and not r['emitted'],'Unknown rotation notprediction')
    rotcorrect=sum(r['eligible'] and r['prediction']==r['target'] for r in controls)
    rotwrong=sum(r['eligible'] and r['emitted'] and r['prediction']!=r['target'] for r in controls)
    ok(rc['total']['argmax_correct']==rotcorrect and rc['total']['confident_wrong']==rotwrong and rc['total']['total']==324,'Rotation totals')
    for group in rc['saved_parent_comparison']:
        name=group['parent'];condition='original' if name=='U' else 'canonical_rotation';om={ck(r):r for r in oldcontrols if r['condition']==condition};paired=[]
        for r in controls:
            q=om[ck(r)];ok(q['eligible']==r['eligible'] and q['target']==r['target'],'Rotatedsavedparents identity')
            paired.append(dict(eligible=r['eligible'],old_correct=q['prediction']==r['target'],new_correct=r['prediction']==r['target']))
        counts=pairmetric(paired);equal(group['counts'],dict(total=counts['total'],eligible=counts['eligible'],old_correct=counts['original_correct'],new_correct=counts['new_correct'],repaired=counts['repaired'],regressed=counts['regressed']),'Rotationpaired')
    byseed=[];minimum={'42':653,'43':672,'44':647}
    for seed in [42,43,44]:
        part=[r for r in wr if r['seed']==seed and r['kind'].startswith('partial')]
        full=[r for r in wr if r['seed']==seed and r['kind'].startswith('full')]
        static=[r for r in wr if r['seed']==seed and r['kind']=='synthetic_constant']
        short=[r for r in part if r['frames']<=7]
        focal=[r for r in part if r['clip_id']=='NIA_SL_WORD0739_REAL06_F' and items['windows'][r['window_id']]['span'][0]==74]
        byseed.append(dict(seed=seed,attribute=binmetric([r for r in ar if r['seed']==seed]),partial=transmetric(part),shortpartial=transmetric(short),
            full=transmetric(full),static=transmetric(static),source74=transmetric(focal),partial_minimum=minimum[str(seed)],
            partial_minimum_pass=transmetric(part)['argmax_correct']>=minimum[str(seed)]))
    # Convert tuple-key grouped audit to portable rows rather than alter scores.
    for q in pairaudit:q['attributes']=[dict(seed=k[0],counts=v) for k,v in sorted(q['attributes'].items())]
    write(A/'INDEPENDENT_REGRESSIONS.json',regressions)
    report=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='SAVED_STORAGE_AND_ARITHMETIC_VERIFIED',checks=CHECKS,
        model_split_norm_audit=run_audit,per_seed=byseed,canonical_pass=sum(r['passed'] for r in cr),canonical_total=54,
        rotation_correct=rotcorrect,rotation_confident_wrong=rotwrong,rotation_total=324,paired=pairaudit,
        regressions=str(A/'INDEPENDENT_REGRESSIONS.json'),no_model_forward=True,no_training=True,
        independence='Separate arithmetic implementation bysameCodexagent, not independent human/KSL labels. Historical initialization backed bypinnedguard/records, not retained pre-update model snapshot.',
        deployment='NoallFIT/webdecisionmade; compare against pre-result INTEGRATION_DECISION_FROZEN and rootreview. Otherpassedcomponents are not blocked automatically.',
        pins={str(p):sha(p) for p in [Path(__file__),A/'INTEGRATION_DECISION_FROZEN.json',A/'TRAINING_COMPLETE.json',E/'RESULTS.json',E/'PAIRED_COMPARISON_R.json',E/'PAIRED_COMPARISON_U.json',A/'rotation_controls/RESULTS.json',A/'INDEPENDENT_REGRESSIONS.json']})
    write(output,report);print('Independent saved audit checks',CHECKS,'canonical',report['canonical_pass'],'/54 rotation',rotcorrect,'/324',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        (A/f'INDEPENDENT_AUDIT_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
