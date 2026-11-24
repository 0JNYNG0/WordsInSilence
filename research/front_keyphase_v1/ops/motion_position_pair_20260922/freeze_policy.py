"""Freeze the matched-capacity comparison before candidate learning."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pair_io import *

def main():
    prior=read(OLD/'comparison_policy.json');ready=read(MOTION/'TRAIN_INPUT_READY.json');oldtrain=read(L/'runs/partial_encoder_pair_v1/training.json')
    assert read(MOTION/'IMPLEMENTATION_AUDIT.json')['status']=='PASS'
    assert ready['module_sha256']==sha(MOTION/'generated_v1.py')
    cache=Path(ready['cache']);assert sha(cache)==ready['cache_sha256']
    pins={str(p):sha(p) for p in [MOTION/'generated_v1.py',MOTION/'TRAIN_INPUT_READY.json',MOTION/'IMPLEMENTATION_AUDIT.json',MOTION/'DESIGN_DECISION.md',MOTION/'CONTROL_REUSE_AUDIT.json',OLD/'comparison_policy.json',OLD/'local_io.py',L/'runs/partial_encoder_pair_v1/training.json',L/'runs/partial_encoder_pair_v1/evaluation.json',Path(prior['parent']['path'])]}
    pins[str(cache)]=ready['cache_sha256']
    p={'schema':'matched_position_vs_motion_input_v1','at_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
      'parent':prior['parent'],'cache':{'path':str(cache),'sha256':ready['cache_sha256']},'pins':pins,
      'arms':{'control':'position features:normalizedXY186+current observationmask93','treatment':'temporal relative coordinate rates186+validadjacentpairmask93'},
      'capacity':'Both add zero-initialized bias-free64x279 projection17856params plus same6045headparams; total23901trainable. Original encoder and all buffers fixed. Same original280-channel external input.',
      'scale':'Each uses class-balanced clip RMS fitted on originalbase2122 only. Motion scale reused; position scale computed once. No dev/user/reserved fitting. Values scaled, masks unscaled.',
      'settings':{'steps':660,'half_batch':16,'learning_rate':.001,'weight_decay':.001,'anchor_coefficient':.01,'output_coefficient':1.,'cpu_threads':2},
      'loss':'CEmean32 + .5*.01*sum squared change of23901trainableparams + .5*mean squared class-centered logit difference from fixedb8teacher on16base draws. Reuse exact existing660x16base/context schedules; no new sampling.',
      'data':'base2122 plus same-source context224;train01-09/dev11-13. User/KSL/reserved/official excluded. No new augmentation,angles,pooling,source clips or label-specific branches.',
      'historical_reference':oldtrain['models']['control'],'historical_reference_counts':{'native699':691,'native_macro_recall':.9861111111111112,'whole74':74,'mp_each_window7':7},
      'gates':{'native_count':'motion>=max(691,position count)','native_macro':'motion>position macrorecall+1e-12 AND motion>=historical .9861111111111112-1e-12 over81observedGTclasses',
        'native_each_signer':'motion>=position count for each11/12/13','native_target':'motion>=max(32,position) on33supportedtarget cases','apology_dev':'3/3 both scopes',
        'whole':'74/74full93 and30/30supportedproject18','public':'18intact/5approvedmissing/9apologytraining each retained bothscopes','mp':'7whole/7annotation each retained bothscopes'},
      'stopping':'One fixed pair only,660updates each,freeze both before public evaluation. No seed/step/coefficient/threshold tuning after scores. Preserve failures;no default promotion.',
      'claim_limits':'A derived motion feature+validity path comparison at equal parameter count; not proof of physical3D understanding,all18-target new-personweb performance,orcorrectexecution. New source diversity stillneeded for undercovered labels.',
      'roles':'Claude generic model extension/trainer;Codex local setup,execution,independent verification; no data/model/credentials sent externally.'}
    write(H/'comparison_policy.json',p);print(json.dumps({'status':'POLICY_FROZEN','sha256':sha(H/'comparison_policy.json'),'planned_training_runs':2}),flush=True)
if __name__=='__main__':main()
