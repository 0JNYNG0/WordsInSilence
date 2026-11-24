"""Preserve first experiment. Fix fit/serving view mismatch for all four words."""
from common import *
import numpy as np,shutil
from prepare_feedback import coord_mask,normalize
def main():
 V=U/'feedback_viewfix';assert not V.exists(),'Do not replace an existing refinement';V.mkdir()
 write(V/'SCOPE.json',dict(date_kst='2026-11-08',reason='Frozen common input plan said alternate full/upper view. First feedback implementation used only its legacy upper view, and full-view diagnostic exposed degradation.',
  original_results_preserved=True,global_fix='Fit-only statistics over both views; deterministic full/upper alternation for all words; evaluate full actual-image bounds.',
  changed_hyperparameters=False,changes_selected_on_development='Input contract correction after full-view diagnostic; no seed, thresholds or epoch selection. DEV remains development, not independent test.',
  additional_conditions='same fresh/transfer three seeds, same120 final epochs',code_author_and_verifier='Codex, separate arithmetic checks'))
 f=read(U/'FROZEN.json');f['parent_frozen_sha256']=sha(U/'FROZEN.json');f['input']['training_view']='TCN and corrected feedback both alternate full/upper by clip SHA parity+epoch'
 f['feedback']['input']='280 shared pose channels; fit stats from both full and legacy upper views; runtime full actual camera bounds';write(V/'FROZEN.json',f)
 oldstats=read(U/'FEEDBACK_NORMALIZATION.json');ready=read(U/'FEEDBACK_READY.json');xs=[];base={}
 for r in ready['rows']:
  rec=read(U/'records'/f"{r['clip_id']}.json");a=np.load(rec['channel_path']);base[r['clip_id']]={'full':a['full'].copy(),'upper':a['feedback'].copy()}
  if r['role']=='fit':
   for x in base[r['clip_id']].values():xs.append(np.where(coord_mask(x),x[:,:186],np.nan))
 allx=np.concatenate(xs);mean=np.nan_to_num(np.nanmean(allx,0));std=np.nanstd(allx,0);std=np.where(np.isfinite(std)&(std>1e-4),std,1)
 stats=dict(oldstats,mean=mean.tolist(),std=std.tolist(),fit_only_normalization=True,normalization_views=['full','upper'])
 write(V/'FEEDBACK_NORMALIZATION.json',stats);newrows=[]
 for r in ready['rows']:
  z=np.load(r['path']);b=base[r['clip_id']];a={k:z[k].copy() for k in z.files};a.update(x=normalize(b['full'],stats),x_full=normalize(b['full'],stats),x_upper=normalize(b['upper'],stats),raw_x=b['full'])
  path=V/'feedback_inputs'/f"{r['clip_id']}.npz";path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**a)
  for k in ['y','ym','sy','sw','ci','frame_id','timestamp_us']:assert np.array_equal(a[k],z[k]),'Supervision changed'
  newrows.append(dict(r,path=str(path),sha256=sha(path)))
 ready.update(rows=newrows,frozen_sha256=sha(V/'FROZEN.json'),normalization_sha256=sha(V/'FEEDBACK_NORMALIZATION.json'));write(V/'FEEDBACK_READY.json',ready)
 write(V/'FEEDBACK_INPUT_VERIFICATION.json',dict(status='PASS',parent_raw_verification_sha256=sha(U/'FEEDBACK_INPUT_VERIFICATION.json'),supervision_unchanged=True,fit_statistics_only=True,
  note='Normalization and alternating views changed; raw physical coordinates already independently verified'))
 # New separate executable copy, never mutate the completed experiment.
 src=(U/'train_feedback.py').read_text(encoding='utf-8')
 old="model.train();order=np.random.default_rng(seed+epoch).permutation(len(fit));losses=[]"
 new="model.train();order=np.random.default_rng(seed+epoch).permutation(len(fit));losses=[]\n    for rr in fit:rr['x']=rr['x_upper'] if (epoch+int(hashlib.sha256(rr['clip_id'].encode()).hexdigest(),16))%2 else rr['x_full']"
 assert src.count(old)==1;src=src.replace(old,new)
 old="for r in rows:\n     b=collate([r]);start=time.perf_counter()";new="for r in rows:\n     r['x']=r['x_full']\n     b=collate([r]);start=time.perf_counter()";assert src.count(old)==1;src=src.replace(old,new)
 (V/'train_feedback.py').write_text(src,encoding='utf-8')
 for name in ['feedback_model.py','check_learning.py']:shutil.copyfile(U/name,V/name)
 shutil.copyfile(Path(__file__),V/'prepare_feedback.py') # exact provenance; preparation was executed in parent unit.
 (V/'common.py').write_text("from pathlib import Path\nimport importlib.util\np=Path(__file__).resolve().parent.parent/'common.py'\ns=importlib.util.spec_from_file_location('_joint_parent_common',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nglobals().update({k:v for k,v in vars(m).items() if not k.startswith('__')})\nU=Path(__file__).resolve().parent\n",encoding='utf-8')
 write(V/'PREPARED.json',dict(status='PASS',sources=48,original_target_arrays_unchanged=True,normalization_fit_people=list(range(1,10)),parent=str(U),new_path=str(V)))
 event('feedback_viewfix_prepared')
if __name__=='__main__':main()
