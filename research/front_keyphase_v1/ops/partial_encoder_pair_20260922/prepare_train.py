"""Prepare only approved training temporal prefixes; verify new cache equivalence."""
import time
from local_io import *
from generated_v1 import extract_prefix,predict_prefix

def main():
    started=time.time();p=policy();audit=read(H/'IMPLEMENTATION_AUDIT.json');assert audit['status']=='PASS'
    assert audit['module_sha256']==sha(H/'generated_v1.py')
    assert not (H/'TRAIN_PREFIX_READY.json').exists()
    ck,m,classes=parent_model(p);xs,meta=train_inputs(p)
    arrays={};records={};maxerr=0.;prepared=0
    with np.load(p['paths']['old_features'],allow_pickle=False) as z:
        assert z['classes'].tolist()==classes
        for group in ['base','pair_context']:
            prefixes=[]
            for start in range(0,len(xs[group]),128):
                prefixes.extend(extract_prefix(m,xs[group][start:start+128]))
                if start%512==0:print('prefix',group,start,'/',len(xs[group]),flush=True)
            actual=predict_prefix(m,prefixes,batch_size=32)['pooled'];expected=z[group+'_x']
            err=float(np.max(np.abs(actual-expected)));maxerr=max(maxerr,err)
            assert np.allclose(actual,expected,atol=1e-4,rtol=1e-5),(group,err)
            w=ck['state_dict']['head.weight'].numpy().astype(float);b=ck['state_dict']['head.bias'].numpy().astype(float)
            assert np.array_equal((actual.astype(float)@w.T+b).argmax(1),(expected.astype(float)@w.T+b).argmax(1))
            packed,off=pack(prefixes);arrays[group+'_prefix']=packed;arrays[group+'_offsets']=off
            arrays[group+'_y']=z[group+'_y'].copy()
            assert arrays[group+'_y'].tolist()==[classes.index(r['label']) for r in meta[group]]
            records[group]={'clips':len(prefixes),'frames':len(packed),'max_prefix_to_prior_pooled_error':err,'same_parent_top1':True}
            prepared+=len(prefixes)
    with np.load(p['paths']['teacher_scores'],allow_pickle=False) as z:
        arrays['teacher_base_logits']=z['parent'].copy()
        assert z['ids'].tolist()==[r['id'] for r in meta['base']]
        assert z['classes'].tolist()==classes
    with np.load(p['paths']['schedule'],allow_pickle=False) as z:
        arrays['base_indices']=z['base_indices'].copy();arrays['context_indices']=z['pair_indices'].copy()
        assert arrays['base_indices'].shape==arrays['context_indices'].shape==(660,16)
    arrays['classes']=np.array(classes)
    with (H/'train_prefixes.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    result={'status':'PASS','policy_sha256':sha(H/'comparison_policy.json'),'module_sha256':sha(H/'generated_v1.py'),
            'cache_sha256':sha(H/'train_prefixes.npz'),'metadata_sha256':sha(p['paths']['metadata']),
            'records':records,'new_prefix_clips':prepared,'prefix_parent_comparison_clips':prepared,'seconds':time.time()-started,
            'training':0,'development_arrays_read':0,'user_arrays_read':0,'reserved_arrays_read':0,'new_extraction':0}
    write(H/'TRAIN_PREFIX_READY.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
