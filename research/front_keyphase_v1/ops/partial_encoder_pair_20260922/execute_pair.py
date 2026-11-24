"""Execute the single frozen local comparison; freeze outputs before evaluation."""
import copy,time
from datetime import datetime
from zoneinfo import ZoneInfo
from local_io import *
from generated_v1 import run_pair

def main():
    p=policy();ia=read(H/'IMPLEMENTATION_AUDIT.json');ready=read(H/'TRAIN_PREFIX_READY.json')
    assert ia['status']==ready['status']=='PASS'
    assert sha(H/'generated_v1.py')==ia['module_sha256']==ready['module_sha256']
    assert sha(H/'train_prefixes.npz')==ready['cache_sha256']
    assert sha(H/'comparison_policy.json')==ready['policy_sha256']
    ck,m,classes=parent_model(p)
    with np.load(H/'train_prefixes.npz',allow_pickle=False) as z:
        assert z['classes'].tolist()==classes
        base=unpack(z['base_prefix'],z['base_offsets']);context=unpack(z['pair_context_prefix'],z['pair_context_offsets'])
        by=z['base_y'].copy();cy=z['pair_context_y'].copy();teacher=z['teacher_base_logits'].copy();bi=z['base_indices'].copy();ci=z['context_indices'].copy()
    assert bi.shape==ci.shape==(660,16)
    R.mkdir(exist_ok=False)
    pins={str(H/n):sha(H/n) for n in ['comparison_policy.json','generated_v1.py','IMPLEMENTATION_AUDIT.json','TRAIN_PREFIX_READY.json','train_prefixes.npz','local_io.py','execute_pair.py']}
    write(R/'preflight.json',{'status':'READY_FOR_ONE_PAIR','pins':pins,'parent':p['parent'],'train_clips':[len(base),len(context)],'train_signers':list(range(1,10)),'user_training_records':0})
    started=time.time()
    def progress(arm,done,total):print(json.dumps({'arm':arm,'steps':done,'of':total,'seconds':round(time.time()-started,1)}),flush=True)
    kwargs={k:p['settings'][k] for k in ['learning_rate','weight_decay','anchor_coefficient','output_coefficient']}
    result=run_pair(m,base,by,context,cy,teacher,bi,ci,progress=progress,**kwargs)
    specs={};arrays={'base_indices':bi,'context_indices':ci};changes={}
    for arm in ['control','treatment']:
        val=result[arm];state=val['state_dict'];allowed=set(val['trainable_names'])
        assert allowed=={n for n,_ in m.named_parameters() if n.startswith('head.') or (arm=='treatment' and n.startswith('blocks.3.'))}
        altered=[k for k in state if not torch.equal(state[k],ck['state_dict'][k])]
        assert altered and set(altered)<=allowed
        assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
        assert val['updates']==660 and np.isfinite(val['losses']).all()
        if arm=='treatment':assert any(n.startswith('blocks.3.') for n in altered)
        out=copy.deepcopy(ck);out['state_dict']=state;out['arm']=arm;out['step']=660
        out['produced_by']='Claude generic partial-encoder learner with local matched comparison'
        out['parent_checkpoint']=p['parent']['path'];out['parent_checkpoint_sha256']=p['parent']['sha256'];out['settings']=plain(result['settings'])
        out['partial_encoder_pair']={'policy_sha256':sha(H/'comparison_policy.json'),'trainable_names':list(val['trainable_names']),'augmentation_this_comparison':False,'all_buffers_fixed':True,'prefix_fixed':True,'registered_as_default':False}
        path=R/('checkpoint_'+arm+'_final.pt');torch.save(out,path)
        specs[arm]={'path':str(path),'sha256':sha(path),'changed_tensors':altered,'trainable_names':val['trainable_names']}
        arrays[arm+'_losses']=val['losses'];changes[arm]=len(altered)
    with (R/'training_arrays.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    for path,d in pins.items():assert sha(path)==d,path
    assert sha(p['parent']['path'])==p['parent']['sha256']
    result_record={'status':'TRAINING_COMPLETE_BOTH_CHECKPOINTS_FROZEN_BEFORE_DEVELOPMENT_EVALUATION','at_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
       'models':specs,'parent':p['parent'],'settings':result['settings'],'seconds':time.time()-started,'actual_training_runs':2,'optimizer_updates':1320,'retry_for_accuracy':0,
       'training_arrays_sha256':sha(R/'training_arrays.npz'),'policy_sha256':sha(H/'comparison_policy.json'),'user_arrays_read':0,'development_arrays_read':0,'reserved_arrays_read':0,'new_data':0}
    write(R/'training.json',result_record);print(json.dumps({'status':result_record['status'],'seconds':result_record['seconds'],'changed_tensors':changes}),flush=True)
if __name__=='__main__':main()
