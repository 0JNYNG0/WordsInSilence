from common_action import *
def main():
 assert not (U/'CONTRACT_CHECK.json').exists()
 torch.set_num_threads(4);rr=rows();r=next(r for r in rr if r['role']=='fit' and r['pairs']);norm=read(U/'NORMALIZATION.json');checks=[]
 def ck(name,ok):
  assert ok,name;checks.append(name)
 for row in rr:
  for p in row['pairs']:ck('ordered reviewed indices '+row['clip_id'],row['frame_id'][p['a']]==p['frame_a'] and row['frame_id'][p['b']]==p['frame_b'])
 b=build_batch([r]);raw=physical(r['x_full']);p=r['pairs'][0]
 g,ok,_=motion.features(raw,r['timestamp_us'],p['a'],p['b'],motion.KINDS[p['axis']]);shift=raw.copy();shift[:,:102]=(shift[:,:102].reshape(-1,51,2)*1.7+np.array([2.4,-.8])).reshape(-1,102)
 g2,ok2,_=motion.features(shift,r['timestamp_us'],p['a'],p['b'],motion.KINDS[p['axis']]);ck('body-relative measurements translation/scale invariant',ok==ok2 and np.allclose(g,g2,atol=2e-5))
 for name in read(U/'EVALUATION_COMPLETE.json')['models']:
  n=CommonActionNet(norm['mean'],norm['std']);n.load_state_dict(torch.load(U/'runs'/name/'model.pt',map_location='cpu',weights_only=True));n.eval()
  with torch.inference_mode():
   a=n.base(b['x'],b['lengths'],torch.tensor([0]));c=n.base(b['x'],b['lengths'],torch.tensor([1]));ck(name+' target cannot alter observed shape',torch.equal(a['attributes'],c['attributes']));ck(name+' target cannot alter latent movement',torch.equal(a['latent'],c['latent']));ck(name+' target cannot alter direction',torch.equal(n.movement(a['latent'],b['geometry'],b['indices']),n.movement(c['latent'],b['geometry'],b['indices'])))
 write(U/'CONTRACT_CHECK.json',dict(status='PASS',checks=len(checks),details=checks,scope='Invariance and target-leakage contracts, not sign correctness. Only FIT example; no DEV changes.'))
 print('contract checks',len(checks))
if __name__=='__main__':main()
