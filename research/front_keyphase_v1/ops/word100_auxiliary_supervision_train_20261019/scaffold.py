"""Create this isolated experiment from the audited prior driver, preserving originals."""
from pathlib import Path
U=Path(__file__).resolve().parent;F=U.parent/'word100_frozen_order_aux_train_20261019'
assert not (U/'common.py').exists()
s=(F/'common.py').read_text(encoding='utf8').replace("ARM='FG16'","ARM='FG16AUX01'\nF=O/'word100_frozen_order_aux_train_20261019';R=O/'word100_auxiliary_supervision_plan_20261019'")
(U/'common.py').write_text(s,encoding='utf8')
s=(F/'train.py').read_text(encoding='utf8')
s=s.replace('from collections import Counter','from collections import Counter\nfrom objective import objective, LAMBDA_AUX')
start=s.index('def prepare():');end=s.index('\ndef scores(',start)
s=s[:start]+'''def prepare():
 from preparation import prepare_experiment
 prepare_experiment(initialize,loadrows,batch,MODEL,TRAIN,NEW_NAMES)
'''+s[end:]
start=s.index('def scores(');end=s.index('\ndef savept(',start)
s=s[:start]+'''def scores(m,dev):
 before=statehash(m.state_dict());m.eval();out={};auxout={};baseout={}
 with torch.no_grad():
  for view in VIEWS:
   valid=[r for r in dev if r['arrays'][view] is not None]
   absent=[prediction(r,reason='frozen_input_unavailable') for r in dev if r['arrays'][view] is None]
   result=list(absent);ar=list(absent);br=list(absent)
   for j in range(0,len(valid),32):
    rows=valid[j:j+32];x,p,_,_=batch(rows,view=view)
    b,a,_,_,_=m.components(x,p);z=b+a
    assert torch.isfinite(z).all() and torch.isfinite(a).all()
    result.extend(prediction(r,q) for r,q in zip(rows,z.numpy()))
    ar.extend(prediction(r,q) for r,q in zip(rows,a.numpy()))
    br.extend(prediction(r,q) for r,q in zip(rows,b.numpy()))
   out[view]=sorted(result,key=lambda r:r['clip_id']);auxout[view]=sorted(ar,key=lambda r:r['clip_id']);baseout[view]=sorted(br,key=lambda r:r['clip_id'])
 assert statehash(m.state_dict())==before
 return out,auxout,baseout
'''+s[end:]
s=s.replace('trace=[];loss_sum=0.','trace=[];loss_sum=0.;aux_sum=0.;total_sum=0.;aux_correct=0')
start=s.index('   opt.zero_grad(set_to_none=True);loss=');end=s.index('  with torch.no_grad():assert torch.equal',start)
s=s[:start]+'''   opt.zero_grad(set_to_none=True);b,a,_,_,_=m.components(x,p)
   loss,final_ce,aux_ce=objective(b,a,y)
   loss.backward();assert all(q.grad is None for q in m.base.parameters())
   assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in params)
   grads={n:float(q.grad.norm()) for n,q in m.named_parameters() if q.requires_grad}
   prior=[q.detach().clone() for q in params]
   norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);opt.step()
   update=float(torch.sqrt(sum(((q.detach()-v)**2).sum() for q,v in zip(params,prior))))
   m.assert_frozen();assert statehash(m.base.state_dict())==frozen
   fc=float(final_ce.detach());ac=float(aux_ce.detach());lc=float(loss.detach())
   loss_sum+=fc*len(rr);aux_sum+=ac*len(rr);total_sum+=lc*len(rr)
   hits=int((a.detach().argmax(1)==y).sum());aux_correct+=hits
   trace.append({'step':step,'clips':[r['clip_id'] for r in rr],'choices':choices,'frozen_state_sha256':frozen,'frozen_modules_eval':True,'gradient_norm':float(norm),'optimizer_total_steps':(epoch-1)*29+step,
    'final_ce':fc,'aux_ce':ac,'total_loss':lc,'lambda_aux':LAMBDA_AUX,'aux_correct_preupdate':hits,'gradient_norm_by_tensor':grads,'update_norm':update,
    'mean_abs_aux_logits':float(a.detach().abs().mean()),'max_abs_aux_logits':float(a.detach().abs().max())})
'''+s[end:]
s=s.replace("pred=scores(m,dev);metrics=", "pred,auxpred,basepred=scores(m,dev);metrics=")
s=s.replace("  if last['best'] is None or ce<", "  auxmetrics={v:detailed_metrics(auxpred[v],context['missingness_groups'][v]) for v in VIEWS}\n  if last['best'] is None or ce<")
s=s.replace("'predictions':pred,'metrics':metrics}","'predictions':pred,'metrics':metrics,'aux_predictions':auxpred,'base_predictions':basepred,'aux_metrics':auxmetrics}")
s=s.replace("'train_ce':loss_sum/899,", "'train_ce':loss_sum/899,'train_aux_ce':aux_sum/899,'train_total_loss':total_sum/899,'train_aux_correct_online':aux_correct,'aux_sensor_top1':auxmetrics['sensor']['top1_correct'],'aux_masked_top1':auxmetrics['masked']['top1_correct'],'aux_sensor_ce':auxmetrics['sensor']['mean_ce'],'aux_masked_ce':auxmetrics['masked']['mean_ce'],")
s=s.replace("'feature_schema':'original_C_280_frozen_T3F_GRU16'", "'feature_schema':'original_C_280_frozen_T3F_GRU16','training_objective':'final_CE_plus_0.1_aux_CE','lambda_aux':LAMBDA_AUX")
s=s.replace("'metrics':best['metrics'],'predictions':best['predictions'],", "'metrics':best['metrics'],'predictions':best['predictions'],'aux_predictions':best['aux_predictions'],'base_predictions':best['base_predictions'],'aux_metrics':best['aux_metrics'],'lambda_aux':LAMBDA_AUX,")
# Actual execution evidence is written by the running process itself.
s=s.replace(" try:\n  for seed in SEEDS:"," write(U/'CURRENT_EXECUTION.json',{'pid':os.getpid(),'status':'RUNNING','running':True,'at':now(),'command':'train.py run','context_sha256':sha(U/'TRAIN_CONTEXT.json')},new=True)\n try:\n  for seed in SEEDS:")
s=s.replace("check(context['pins']);lock.unlink();", "check(context['pins']);current=read(U/'CURRENT_EXECUTION.json');current.update(status='COMPLETE',running=False,exit_code=0,completed_at=now());write(U/'CURRENT_EXECUTION.json',current);lock.unlink();")
(U/'train.py').write_text(s,encoding='utf8')
# The saved-score verifier is copied only into this new directory; all old evidence stays unchanged.
v=(F/'verify.py').read_text(encoding='utf8')
v=v.replace("models=['C','T3F','FG16']", "models=['C','T3F','FG16',ARM]")
v=v.replace("else T if model=='T3F' else U", "else T if model=='T3F' else F if model=='FG16' else U")
v=v.replace("if model=='FG16':traces[str(seed)]=trace(r,ctx)","if model==ARM:traces[str(seed)]=trace(r,ctx)")
v=v.replace("for baseline in ['C','T3F']:","for baseline in ['C','T3F','FG16']:")
v=v.replace("stored['FG16',seed]","stored[ARM,seed]")
v=v.replace("row['FG16_sensor_errors']", "row[ARM+'_sensor_errors']")
v=v.replace("gate(metrics['FG16'],", "gate(metrics[ARM],")
v=v.replace("'secondary_gate_vs_C':", "'primary_review_gate_vs_FG16':gate(metrics[ARM],metrics['FG16']),'secondary_gate_vs_C':")
v=v.replace("'saved_predictions_verified':5400", "'saved_predictions_verified':7200")
v=v.replace("GRU capacity and order both change; order-only causality not isolated.", "Versus FG16 only auxiliary objective changes; semantic order-only causality not established.")
v=v.replace(" assert result['predictions']==last['best']['predictions']", " assert result['lambda_aux']==best['lambda_aux']==0.1\n assert result['aux_predictions']==last['best']['aux_predictions'] and result['base_predictions']==last['best']['base_predictions']\n assert result['predictions']==last['best']['predictions']")
v=v.replace("  for step,b in enumerate(tr['batches'],1):", "  for field,tracefield in [('train_ce','final_ce'),('train_aux_ce','aux_ce'),('train_total_loss','total_loss')]:assert abs(h[field]-sum(b[tracefield]*len(b['clips']) for b in tr['batches'])/899)<1e-10\n  assert h['train_aux_correct_online']==sum(b['aux_correct_preupdate'] for b in tr['batches'])\n  for step,b in enumerate(tr['batches'],1):")
v=v.replace("   assert b['step']==step", "   assert b['lambda_aux']==0.1 and abs(b['total_loss']-(b['final_ce']+0.1*b['aux_ce']))<1e-6\n   assert 0<=b['aux_correct_preupdate']<=len(b['clips']) and b['update_norm']>=0\n   assert set(b['gradient_norm_by_tensor'])==set(ctx['trainable_parameter_names']) and all(math.isfinite(n) and n>=0 for n in b['gradient_norm_by_tensor'].values())\n   assert b['step']==step")
v=v.replace(" out={'status':", " from verify_aux import audit_aux\n aux_verification=audit_aux(ctx,stored,recount)\n out={'auxiliary_verification':aux_verification,'status':")
(U/'verify.py').write_text(v,encoding='utf8')
print('Created isolated common/train/verify from read-only prior sources.')
