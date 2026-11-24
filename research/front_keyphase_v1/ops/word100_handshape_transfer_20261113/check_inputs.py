from prepare import *

def main():
    cfg=read(U/'FROZEN.json');manifest=read(U/'INPUT_MANIFEST.json');z=np.load(U/'INPUTS.npz')
    assert not (U/'RESULTS.json').exists()
    checks=[]
    def check(name,passed):
        checks.append(dict(name=name,passed=bool(passed)));assert passed,name
    check('source_and_protected_hashes',all(sha(p)==h for p,h in dict(cfg['source_pins'],**cfg['protected']).items()))
    check('finite_observations',all(np.isfinite(z[k]).all() for k in z.files))
    check('unknown_masks_present',any(-1 in r['y'] for r in manifest['ordinary']) and any(-1 in r['finger_states'] for r in manifest['auxiliary']))
    check('sparse_target_only',all(r['origins'] for r in manifest['ordinary']))
    # Use a real public hand for invariance and complete-missing controls.
    cid=manifest['ordinary'][0]['clip_id'];rec=read(J/'records'/f'{cid}.json');raw=np.load(rec['raw_path']);chan=np.load(rec['channel_path']);r=manifest['ordinary'][0]
    i=int(np.flatnonzero(raw['frame_id']==r['frame'])[0]);sl=slice(112,133) if r['side']=='R' else slice(91,112);xy=raw['xy_px'][i,sl];mask=chan['point_available'][i,sl]
    base,ok=feature(xy,mask);check('base_available',ok)
    angle=.63;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],np.float32)
    for name,xx in [('shift',xy+[312.,-201.]),('scale',xy*1.7),('rotation',xy@rot),('reflection',xy*[-1,1])]:
        changed,good=feature(xx,mask);check('shape_invariant_'+name,good and np.max(np.abs(base-changed))<.0001)
    absent,good=feature(xy,np.zeros(21,bool));check('missing_is_not_invented',not good and not absent.any())
    check('numeric_dev_not_used',all(r['role']=='fit' and r['person']<=9 for r in manifest['auxiliary']))
    check('no_private_source',all(r['clip_id'].startswith('NIA_SL_WORD') for r in manifest['ordinary']+manifest['auxiliary']))
    write(U/'PRETRAIN_VERIFICATION.json',dict(status='PASS',checks=checks,reviewer='same Codex separate checks; not independent expert'))
    write(U/'CODE_PINS.json',{f:sha(U/f) for f in ['prepare.py','train.py','check_inputs.py']})
    print('PASS',len(checks),'checks; no model executed')
if __name__=='__main__':main()
