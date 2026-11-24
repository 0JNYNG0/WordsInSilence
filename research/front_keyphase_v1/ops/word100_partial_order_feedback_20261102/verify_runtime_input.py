"""No model forwards: exact existing-input compatibility for the new adapter."""
from order_runtime import *

def main():
    dest=Z/'INPUT_COMPATIBILITY_VERIFIED.json';assert not dest.exists()
    meta=read(O/'word100_coherent_sensor_input_20261102/INPUT_MANIFEST.json')['rows'][0]
    assert meta['person']==11 and meta['word_id']=='WORD0739'
    a=dict(np.load(meta['input'],allow_pickle=False)); base=read(meta['result'])
    validate_input(a,meta['word_id'],base)
    r=model.get_rows('dev')[0]; assert r['clip_id']==meta['clip_id'] and r['kind']=='native'
    saved=dict(np.load(model.OUT/'predictions/tied_readout/dev/dev_000.npz',allow_pickle=False))
    stats=read(model.D/'NORMALIZATION.json');x,rgb=normalized_input(a,saved['current'],stats)
    oldx,oldrgb,_,_,_=model.batch_j([r],stats)
    assert np.array_equal(x,oldx[0].numpy()) and np.array_equal(rgb,oldrgb[0].numpy())
    state=torch.load(model.PROTO,map_location='cpu',weights_only=True);wi=model.process.WORDS.index(meta['word_id']);ref=state['prototype'][wi,:int(state['prototype_lengths'][wi])].numpy()
    flow=read(model.FLOW_UNIT/'FLOW_INPUTS.json')
    raw,norm=flow_from_path(a,base['path'],ref,read(model.P/'GUIDE_SUPPORT.json')['guides'],meta['word_id'],flow)
    oldflow=model.flow_batch([r],flow)[0].numpy()
    assert np.array_equal(norm,oldflow)
    expected=np.load(flow['items']['dev_000']['path'])
    assert np.array_equal(raw,expected['features']) and np.array_equal(base['path'],expected['path'])
    # Only structural corruption tests; no alternate thresholds or scores.
    bad=dict(a);bad['timestamp_us']=np.asarray(a['timestamp_us'])[::-1]
    try: validate_input(bad,meta['word_id'],base)
    except AssertionError: pass
    else: raise AssertionError('nonmonotonic time accepted')
    out=dict(status='PASS',native_input=meta['input'],input_sha256=sha(meta['input']),
        features130_exact=True,rgb769_exact=True,flow6_exact=True,existing_DTW_path_reused=True,
        no_annotation_runtime_input=True,new_model_inference=0,new_training=0,
        scope='One preselected native input compatibility; saved results not re-scored',whole_goal_complete=False)
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out),flush=True)

if __name__=='__main__':main()
