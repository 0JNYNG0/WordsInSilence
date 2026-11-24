from completion_model import *
import traceback

def main():
 assert not (U/'CROSS_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 cfg=read(U/'FROZEN.json');sys.modules['motion_core']=mc;reader=load('_bilateral_public_convert',M/'evaluate.py')
 rows=read(R/'CROSS_EXPRESSION_SCORES.json');raws={r['clip_id']:r for r in read(A/'EXTRACTION_COMPLETE.json')['rows'] if r['person']==14};fitsha={r['source_sha256'] for r in read(U/'LABELS.json')['rows'] if r['role']=='fit'}
 assert not fitsha & {r['source_sha256'] for r in raws.values()}
 write(U/'CROSS_FROZEN.json',dict(pins={str(U/'cross_check.py'):sha(U/'cross_check.py')},videos=list(raws),variants=['normal','whole_reverse','speed075','speed125'],reuse_previous_proposals=True,exposure='Previously used non-focus six expressions. Separate from current fit but NOT fresh unseen evaluation.'))
 result=[]
 nets={}
 for seed in cfg['seeds']:
  net=CompletionNet(seed,cfg['norm']);net.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval();nets[seed]=net
 for clip,src in raws.items():
  assert sha(src['raw'])==src['raw_sha256']
  for kind in ['normal','whole_reverse','speed075','speed125']:
   x,ts=reader.convert(src['raw'],kind=='whole_reverse');ts=np.asarray(ts)/dict(speed075=.75,speed125=1.25).get(kind,1)
   for seed,net in nets.items():
    rr=[r for r in rows if r['clip_id']==clip and r['kind']==kind and r['seed']==seed];pp=infer_batch(net,x,ts,[r['event'] for r in rr])
    for r,p in zip(rr,pp):result.append(dict(clip_id=clip,kind=kind,seed=seed,event=r['event'],target=r['target'],before=r['after'],before_complete=bool(r['after']==2 and r['after_probability'][2]>=.65),after=p,known=r['known']))
 write(U/'CROSS_SCORES.json',result);write(U/'CROSS_COMPLETE.json',dict(videos=6,conditions=24,rows=len(result),fresh_independent_human=False));event('cross_checked',conditions=24)
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'CROSS_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
