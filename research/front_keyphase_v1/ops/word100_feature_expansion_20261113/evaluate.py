from model_core import *
import statistics,os,traceback
GROUPS={s:i for i,g in enumerate(read(old.S/'FROZEN.json')['semantic_stage_groups']) for s in g}
VARIANTS=['normal','upper','slow','fast','missing_2','missing_long','reversed_main']
FINE_VARIANTS=['normal','upper','slow','fast','fine_missing_2','fine_missing_12']
ALL_VARIANTS=VARIANTS+['fine_missing_2','fine_missing_12']

def transform(r,v):
    b=batch([r],'upper' if v=='upper' else 'full');x=b['x'][0].numpy().copy();hand=b['hand'][0].numpy().copy();n=len(x);ids=np.arange(n);gold=np.flatnonzero(r['stage_gold']>=0)
    if v=='reversed_main' and len(gold):
        a=min([int(gold[0])]+[p['a'] for p in r['pairs']]);z=max([int(gold[-1])]+[p['b'] for p in r['pairs']]);ids[a:z+1]=ids[a:z+1][::-1];x=x[ids];hand=hand[ids]
    if v in ['slow','fast']:
        ix=np.rint(np.linspace(0,n-1,round(n*(1.25 if v=='slow' else .75)))).astype(int);x=x[ix];hand=hand[ix];ids=ids[ix]
    if v in ['missing_2','missing_long']:
        # Same mask timing as stored shared100 comparisons (whole clip for long).
        mid=int(np.median(gold)) if len(gold) else n//2;gap=2 if v=='missing_2' else n;a=max(0,mid-gap//2);z=min(n,a+gap)
        x=old.mask_hand(old.mask_hand(x,a,z,0),a,z,1);hand[a:z]=0
    if v in ['fine_missing_2','fine_missing_12']:
        anchors=np.flatnonzero((r['fy']>=0).any((1,2)));mid=int(anchors[0]);gap=2 if v=='fine_missing_2' else 12
        a=max(0,mid-gap//2);z=min(n,a+gap)
        x=old.mask_hand(old.mask_hand(x,a,z,0),a,z,1);hand[a:z]=0
    return x,hand,ids

def summarize(records):
    yy=np.array([r['target'] for r in records],int);pp=np.array([r['pred'] for r in records],int)
    cm=np.zeros((3,3),int)
    for a,b in zip(yy,pp):cm[a,b]+=1
    f1=[]
    for k in range(3):
        den=cm[k,:].sum()+cm[:,k].sum();f1.append(float(2*cm[k,k]/den) if den else None)
    accepted=[r for r in records if r['accepted']]
    return dict(total=len(records),correct=int((yy==pp).sum()),accuracy=float((yy==pp).mean()) if len(yy) else None,macro_f1=float(np.mean([x for x in f1 if x is not None])) if any(x is not None for x in f1) else None,state_f1=f1,confusion=cm.tolist(),accepted=len(accepted),accepted_correct=sum(r['pred']==r['target'] for r in accepted),withheld=len(records)-len(accepted))

def old_summary(rows):
    out={}
    for task in sorted({r['task'] for r in rows}):
        rs=[r for r in rows if r['task']==task];ac=[r for r in rs if r['eligible'] and r['confident']]
        out[task]=dict(n=len(rs),raw_correct=sum(r['pred']==r['target'] for r in rs),accepted=len(ac),accepted_correct=sum(r['pred']==r['target'] for r in ac))
    return out

def main():
    cfg=read(U/'FROZEN.json');assert (U/'TRAINING_COMPLETE.json').exists()
    protocol_path=U/'EVAL_FROZEN.json'
    if not protocol_path.exists():
        pins={str(U/'evaluate.py'):sha(U/'evaluate.py')}
        for seed in cfg['seeds']:
            p=B/'scores'/f'shared100_{seed}.json';pins[str(p)]=sha(p)
            for condition in cfg['conditions']:
                p=U/'runs'/f'{condition}_{seed}'/'model.pt';pins[str(p)]=sha(p)
        write(protocol_path,dict(status='FROZEN_BEFORE_EVALUATION',pins=pins,fine_variants=FINE_VARIANTS,legacy_variants=VARIANTS,fine_denominator=389,mask_center='first direct finger anchor; both hands2/12 frames',legacy_mask='exact saved baseline whole hand stream for missing_long',latency='36 different cached ordinary clips; no causal comparison with older single-clip repeated benchmark'))
    for p,h in read(protocol_path)['pins'].items():assert sha(p)==h,p
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rr=rows_load('development');dev=[r for r in rr if r['origin']=='common100'];labelled=[r for r in dev if (r['fy']>=0).any()];feedback=[r for r in dev if r['word']>=0]
    assert {r['clip_id'] for r in feedback}.issubset({r['clip_id'] for r in labelled}), 'Legacy feedback clips omitted from evaluation loop'
    extra_by={}
    for r in read(B/'EXTRA_EVAL_LABELS.json')['rows']:extra_by.setdefault(r['clip_id'],[]).append(r)
    out={}
    for condition in cfg['conditions']:
      for seed in cfg['seeds']:
        name=f'{condition}_{seed}';sp=U/'scores'/f'{name}.json'
        if sp.exists():out[name]=read(sp);continue
        write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATING',pid=os.getpid(),model=name))
        net=FineNet(len(cfg['extra_words']));net.load_state_dict(torch.load(U/'runs'/name/'model.pt',map_location='cpu',weights_only=True));net.eval();fingers=[];records=[];words=[];latency=[]
        with torch.inference_mode():
          for r in dev:
            b=batch([r]);o=net(b['x'],b['lengths'],b['word'],b['hand']);words.append(dict(clip_id=r['clip_id'],target=r['class_index'],pred=int(o['word_logits'].argmax(-1))))
            if r['clip_id'] in extra_by:
                p=o['attributes'][0].sigmoid().numpy();av=old.available(r['x_full'])
                for e in extra_by[r['clip_id']]:
                    k=int(np.flatnonzero(r['frame_id']==e['frame'])[0])
                    for j,t in enumerate(e['targets']):
                        if t is None:continue
                        z=float(p[k,j]);records.append(dict(task='expanded_shape',variant='normal',clip_id=r['clip_id'],frame=e['frame'],feature=j,target=t,pred=int(z>=.5),probability=z,eligible=bool(av[k,j]),confident=z<=.35 or z>=.65))
          for variant in ALL_VARIANTS:
            for r in labelled:
              if variant=='reversed_main' and r['word']<0:continue
              x,hand,ids=transform(r,variant)
              t=time.perf_counter();o=net(torch.from_numpy(x[None]),torch.tensor([len(x)]),torch.tensor([max(0,r['word'])]),torch.from_numpy(hand[None]));dt=time.perf_counter()-t
              if variant=='normal':latency.append(dt)
              fp=o['fingers'][0].softmax(-1).numpy();at=o['attributes'][0].sigmoid().numpy();st=o['stage'][0].softmax(-1).numpy();av=old.available(x)
              def ix(i):return int(np.argmin(abs(ids-int(i))))
              if variant in FINE_VARIANTS:
                for i,s,f in zip(*np.where(r['fy']>=0)):
                    k=ix(i);prob=fp[k,s,f];pred=int(prob.argmax());q=[0]+list(range(1+f*4,5+f*4));eligible=bool((hand[k,s,42:][q]>.5).all())
                    fingers.append(dict(variant=variant,clip_id=r['clip_id'],word=r['word_id'],person=r['person'],frame=int(r['frame_id'][i]),pts_us=int(r['timestamp_us'][i]),side='L' if s==0 else 'R',finger=int(f),target=int(r['fy'][i,s,f]),pred=pred,probability=prob.tolist(),eligible=eligible,accepted=bool(eligible and prob[pred]>=.85)))
              if r['word']>=0 and variant in VARIANTS:
                for task,mask,y in [('reviewed_shape',r['direct_ym'],r['y']),('original_shape_labels',r['old_ym'],r['old_y'])]:
                  for i,j in zip(*np.where(mask>0)):
                    k=ix(i);p=float(at[k,j]);records.append(dict(task=task,variant=variant,clip_id=r['clip_id'],frame=int(r['frame_id'][i]),feature=int(j),target=int(y[i,j]),pred=int(p>=.5),probability=p,eligible=bool(av[k,j]),confident=p<=.35 or p>=.65))
                for i in np.flatnonzero(r['stage_gold']>=0):
                    k=ix(i);pred=int(st[k].argmax());eligible=bool(x[k,187:189].min()>.5 and x[k,186:237].sum()>=5)
                    records.append(dict(task='semantic_stage',variant=variant,clip_id=r['clip_id'],frame=int(r['frame_id'][i]),target=GROUPS[int(r['stage_gold'][i])],pred=GROUPS[pred],eligible=eligible,confident=bool(st[k,pred]>=.65)))
                for pair in r['orders']:
                    a,z=sorted([ix(pair['a']),ix(pair['b'])]);sa=int(st[a].argmax());sz=int(st[z].argmax())
                    records.append(dict(task='normal_order',variant=variant,clip_id=r['clip_id'],target=int(variant!='reversed_main'),pred=int(sa<sz),eligible=bool(av[a:z+1].any()),confident=sa!=sz))
              if seed==42 and variant=='normal':
                dst=U/'predictions'/name/f"{r['clip_id']}.npz";dst.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(dst,fingers=fp,attributes=at,stage=st,frame_id=r['frame_id'],timestamp_us=r['timestamp_us'])
        normal=[r for r in fingers if r['variant']=='normal']
        result=dict(finger_by_variant={v:summarize([r for r in fingers if r['variant']==v]) for v in FINE_VARIANTS},finger_by_hand={s:summarize([r for r in normal if r['side']==s]) for s in ['L','R']},finger_by_record={str(p):summarize([r for r in normal if r['person']==p]) for p in [11,12,13]},finger_by_word={w:summarize([r for r in normal if r['word']==w]) for w in sorted({r['word'] for r in normal})},finger_by_finger={str(f):summarize([r for r in normal if r['finger']==f]) for f in range(5)},legacy={v:old_summary([r for r in records if r['variant']==v]) for v in VARIANTS},word100=dict(correct=sum(r['target']==r['pred'] for r in words),total=300),latency=dict(p50_seconds=statistics.median(latency),p95_seconds=float(np.percentile(latency,95)),clips=len(latency),scope='cached coordinates to whole candidate neural outputs, CPU4; excludes video extraction/network/UI'))
        write(U/'scores'/f'{name}_latency.json',dict(seconds=latency,clip_ids=[r['clip_id'] for r in labelled],scope=result['latency']['scope']))
        write(U/'scores'/f'{name}_fingers.json',fingers);write(U/'scores'/f'{name}_legacy.json',records);write(U/'scores'/f'{name}_words.json',words);write(sp,result);out[name]=result
        event('evaluated',model=name,finger=result['finger_by_variant']['normal'],word100=result['word100'])
    retention_rows=[]
    for condition in cfg['conditions']:
      for seed in cfg['seeds']:
        z=out[f'{condition}_{seed}'];base=read(B/'scores'/f'shared100_{seed}.json');fail=[]
        for v,ts in z['legacy'].items():
            assert set(ts)==set(base['by_variant'][v]),(condition,seed,v,'task mismatch')
            for task,q in ts.items():
                oldq=base['by_variant'][v].get(task)
                if oldq:
                    assert q['n']==oldq['n'],(condition,seed,v,task,q['n'],oldq['n'])
                    if q['raw_correct']<oldq['raw_correct']:fail.append(dict(variant=v,task=task,before=oldq['raw_correct'],after=q['raw_correct'],n=q['n']))
        retention_rows.append(dict(condition=condition,seed=seed,legacy_failures=fail,word100_before=base['aux_word']['correct'],word100_after=z['word100']['correct'],full_preservation=not fail and z['word100']['correct']>=base['aux_word']['correct']))
    gates=[]
    for seed in cfg['seeds']:
        a=out[f'A_{seed}'];c=out[f'C_{seed}'];base=read(B/'scores'/f'shared100_{seed}.json')
        improvement=c['finger_by_variant']['normal']['macro_f1']>=a['finger_by_variant']['normal']['macro_f1']+.01
        hands=all(c['finger_by_hand'][s]['accuracy']>=a['finger_by_hand'][s]['accuracy'] for s in ['L','R'])
        states=all((cv or 0)>=(av or 0) for av,cv in zip(a['finger_by_variant']['normal']['state_f1'],c['finger_by_variant']['normal']['state_f1']))
        failures=[]
        for v,ts in c['legacy'].items():
            for task,q in ts.items():
                b=base['by_variant'][v].get(task)
                if b and (q['n']!=b['n'] or q['raw_correct']<b['raw_correct']):failures.append(dict(variant=v,task=task,before=b['raw_correct'],after=q['raw_correct'],n=q['n'],baseline_n=b['n']))
        word_ok=c['word100']['correct']>=base['aux_word']['correct']
        gates.append(dict(seed=seed,consistent_fine_gain=improvement,hands_preserved=hands,states_preserved=states,legacy_failures=failures,word100_preserved=word_ok,pass_candidate=bool(improvement and hands and states and not failures and word_ok)))
    write(U/'EVALUATION_COMPLETE.json',dict(status='DEVELOPMENT_COMPARISON_COMPLETE',models=out,retention_all_conditions=retention_rows,gates=gates,all_pass=all(r['pass_candidate'] for r in gates),no_production_promotion=True,old_baseline_from_stored_scores=True))
    write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFICATION',pid=None))

if __name__=='__main__':
    try:main()
    except BaseException:
        (U/f'FAILURE_EVAL_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
