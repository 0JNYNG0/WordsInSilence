from core import *
from collections import defaultdict

def main():
 assert not (W/'FROZEN_V2.json').exists()
 meta=read(W/'INPUTS_COMPLETE.json');rows=meta['rows'];d=np.load(W/'pairs.npz');train=d['train']&d['eligible'];supervision=d['mask'].copy();mixed=[];detail=[]
 for j in range(2):
  words=sorted({r['word_id'] for r in rows});supported=[]
  for word in words:
   m=train&(d['mask'][:,j]>0)&np.array([r['word_id']==word for r in rows]);classes=set(d['y'][m,j].tolist())
   if classes=={0.,1.}:supported.append(word)
  mixed.append(supported);supervision[:,j]*=np.array([r['word_id'] in supported for r in rows])
  detail.append(dict(head=j,mixed_train_words=supported,source_pairs=int(np.sum(supervision[:,j]*train)/2),reason='Only words with both outcomes supervise this head. Single-outcome words remain separately reported transfer controls, not training shortcuts.'))
 assert len(mixed[0])==3 and len(mixed[1])==1
 np.savez_compressed(W/'SUPERVISION.npz',mask=supervision)
 note=dict(at=stamp(),cause='Word-only parent classifier achieved95.34% balanced accuracy on unbalanced probe; source-label mapping valid, raw metric misleading.',repair='Within-word class-balanced supervision and within-word balanced primary reporting. Word-constant classifier then scores50% in each mixed word. Do not manufacture negatives from unrelated words.',heads=detail,atomic_limitation='Only one word supplies both atomic outcomes in TRAIN. Atomic head is exploratory; repeated-cycle generalization not established. It is not used for candidate linkage or semantic repetition feedback.',label_mapping_changed=False,first_audit=str(W/'PRETRAIN_VERIFICATION.json'),upstream_preserved=True,source_boundary_labels_added=False)
 write(W/'A_REWORK.json',note)
 cfg=read(W/'FROZEN.json');cfg['at']=stamp();cfg['supersedes_without_training']=str(W/'FROZEN.json');cfg['loss']='For each head use only TRAIN words containing both outcomes; equal weight per word, class, source clip and pair/view. Two heads equal. No word input.';cfg['supervision']=detail;cfg['primary_metric']='Macro per-word balanced accuracy for source-reviewed mixed-outcome words; separately show all original probe and untrained atomic words. Word-only baseline is50% per mixed word.';cfg['pins'].update({str(p):sha(p) for p in [W/'A_REWORK.json',W/'SUPERVISION.npz',Path(__file__),W/'train.py']});write(W/'FROZEN_V2.json',cfg)
 print('A repaired supervision:',detail,flush=True)

if __name__=='__main__':main()
