from semantic_model import *

def main():
 cfg=read(U/'FROZEN.json');rows=load_rows();checks=[]
 def ok(name,value):
  assert value,name;checks.append(name)
 ok('48 public clips only',len(rows)==48)
 ok('disjoint participant split',all((r['role']=='fit')==(r['person'] in range(1,10)) for r in rows))
 ok('12 development clips',sum(r['role']=='development' for r in rows)==12)
 ok('all frozen inputs match',all(sha(p)==h for p,h in cfg['data_pins'].items()))
 ok('protected original assets match',all(sha(p)==h for p,h in cfg['source_pins'].items()))
 ok('annotation freeze matches',sha(U/'ANNOTATIONS.json')==cfg['annotations_sha256'])
 ok('no personal paths',all('local_runs' not in r['source_input'] and 'user' not in r['source_input'].lower() for r in rows))
 reviews={r['clip_id']:r for r in read(U/'REVIEW_INPUTS.json')['rows']}
 for r in rows:
  ok(r['clip_id']+' finite',all(np.isfinite(r[k]).all() for k in ['x_full','x_upper','y','ym']))
  if r['pairs']:
   frames=set(reviews[r['clip_id']]['frames']);idx=np.flatnonzero(r['stage_gold']>=0)
   ok(r['clip_id']+' exact RGB anchors',set(r['frame_id'][idx]).issubset(frames))
   rev=transform(r,reverse=True)
   ok(r['clip_id']+' reverse preserves shape count',np.array_equal(rev['ym'].sum(0),r['ym'].sum(0)))
   ok(r['clip_id']+' reverse negates all pair signs',all(a['target']==2-b['target'] for a,b in zip(rev['pairs'],r['pairs'])))
   ok(r['clip_id']+' reverse cannot align to normal prototype',np.all(rev['ci']==-1))
   ok(r['clip_id']+' order negative for reversed non-repeating stages',all(p['target']==0 for p in rev['orders']))
   missing=transform(r,gap=len(r['x'])*2)
   ok(r['clip_id']+' no evidence cannot give direction',all(not pair_available(missing['x'],p['a'],p['b'],p['axis']) for p in r['pairs']))
   ok(r['clip_id']+' no evidence cannot give handshape',not availability(missing['x']).any())
 torch.set_num_threads(4);torch.manual_seed(42);m=SemanticNet();m.base.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=True));m.eval()
 r=transform(rows[0]);b=collate([r])
 with torch.no_grad():
  a=m.base(b['x'],b['lengths'],b['word']);c=m.base(b['x'],b['lengths'],(b['word']+1)%4)
 ok('target word cannot alter observed shape',torch.equal(a['attributes'],c['attributes']))
 ok('target word cannot alter observation latent',torch.equal(a['latent'],c['latent']))
 d1,_=m.pair(a['latent'][0],10,20,0);d2,_=m.pair(a['latent'][0],10,20,1)
 ok('target word cannot alter direction',torch.equal(d1,d2))
 loss=batch_losses(m,[transform(r) for r in rows[:6]],torch.ones(7));total=sum(loss.values());total.backward()
 ok('finite backward to GRU',m.base.context.weight_ih_l0.grad is not None and torch.isfinite(m.base.context.weight_ih_l0.grad).all().item())
 ok('direction and order gradients',m.direction[0].weight.grad is not None and m.order[0].weight.grad is not None)
 for p in m.base.parameters():p.requires_grad_(False)
 unlabeled=[transform(r) for r in rows if not r['pairs']][:6]
 empty=sum(batch_losses(m,unlabeled,torch.ones(7)).values())
 ok('frozen empty-label batch has no gradient and must be skipped',not empty.requires_grad)
 write(U/'PRETRAIN_CHECK.json',dict(status='PASS',checks=checks,count=len(checks),frozen_sha256=sha(U/'FROZEN.json'),code_pins={str(U/f):sha(U/f) for f in ['semantic_model.py','train.py','common.py']},reviewer='Codex local code review + executable checks, not independent expert'))
 event('pretrain_checks_pass',count=len(checks))
if __name__=='__main__':main()
