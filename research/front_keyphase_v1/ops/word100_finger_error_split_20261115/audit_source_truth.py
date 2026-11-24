from split_runtime import *
import copy
changes=[dict(role='diagnostic',clip_id='NIA_SL_WORD2748_REAL10_F',hand='R',frame=61,finger=0,old_shape=1,new_shape=0,reason='Same-take F/L/R and neighboring58/61/64 show actual hooked index then reopening, not merely a straight index foreshortened. Initial frontal-only annotation was incorrect.',image=str(U/'review/multiview/NIA_SL_WORD2748_REAL10.jpg'))]
audit=dict(changes=changes,additional_checks=[dict(clip_id='NIA_SL_WORD2748_REAL16_F',frame=66,decision='retain flexed index label0',image=str(U/'review/multiview/NIA_SL_WORD2748_REAL16.jpg')),dict(clip_id='NIA_SL_WORD2551_REAL12_F',frame=35,decision='retain extended index label1; frontal overlay follows visible index. Genuine shape-decision error remains.',image=str(U/'review/multiview/historic_overlay.jpg'))],timing_caveat='All3streams same frame count/fps per take; neighboring visible hand bend/reopen events agree. No calibrated camera synchronization or 3D joint truth claimed.',procedure='Post-prediction source audit. Original frozen labels/results remain. Recount both baseline/candidate identically without retraining/reselection or changing promotion gates.',expert_review=False)
for stem in ['NIA_SL_WORD2748_REAL10','NIA_SL_WORD2748_REAL16','NIA_SL_WORD2551_REAL12']:
 rr=[r for r in read(U/'MULTIVIEW_INPUTS.json')['rows']if r['clip_id'].startswith(stem)];assert len({(r['frames'],r['fps'])for r in rr})==1
write(U/'SOURCE_TRUTH_AUDIT.json',audit)
d=read(U/'diagnostic_RESULTS.json');rr=copy.deepcopy(d['rows'])
for c in changes:
 r=next(r for r in rr if r['clip_id']==c['clip_id']and r['frame']==c['frame']and r['hand']==c['hand']);assert r['shape'][c['finger']]==c['old_shape'];r['shape'][c['finger']]=c['new_shape']
y=np.array([r['shape']for r in rr]);a=np.array([r['predictions']['available']for r in rr]);mask=(y>=0)&a;base=np.array([r['predictions']['base']for r in rr]);oldcorrect=(base==y)&mask;metrics={}
for mode in d['metrics']:
 p=np.array([r['predictions'][mode]for r in rr]);metrics[mode]=dict(known=int(mask.sum()),correct=int(((p==y)&mask).sum()),wrong=int(((p>=0)&(p!=y)&mask).sum()),unknown=int(((p<0)&mask).sum()),lost_previous_correct=int(((p!=y)&oldcorrect).sum()),new_correct=int(((p==y)&mask&~oldcorrect).sum()))
write(U/'diagnostic_SOURCE_AUDITED_RESULTS.json',dict(original=str(U/'diagnostic_RESULTS.json'),audit=str(U/'SOURCE_TRUTH_AUDIT.json'),metrics=metrics,rows=rr,not_an_independent_test=True,model_weights_unchanged=True))
# Controlled input-context ablation for the remaining historic error, diagnosis only.
torch.set_num_threads(2);record=read(J/'records/NIA_SL_WORD2551_REAL12_F.json');x,fid=load_channels(record['channel_path']);j=int(np.flatnonzero(fid==35)[0]);model=SplitObserver();real=model.all(x,'R');constant=model.all(np.repeat(x[j:j+1],15,axis=0),'R')
write(U/'CLEAN_CHAIN_DIAGNOSTIC.json',dict(clip_id='NIA_SL_WORD2551_REAL12_F',frame=35,hand='R',finger=0,truth=1,RGB_chain_quality='own chain visually consistent; detailed 3D angles not ground truth',real_states={k:int(real[k][j,0])for k in ['base','shape_only','compatible_shape','combined']},center_repeated_states={k:int(constant[k][7,0])for k in ['base','shape_only','compatible_shape','combined']},real_shape_probabilities=real['probabilities']['shape'][:,j,0],center_repeated_shape_probabilities=constant['probabilities']['shape'][:,7,0],scope='Mechanism probe only, repeated frame is synthetic and never training, model selection or efficacy evidence'))
print(json.dumps(dict(audited_diagnostic=metrics,probe=read(U/'CLEAN_CHAIN_DIAGNOSTIC.json'))))
