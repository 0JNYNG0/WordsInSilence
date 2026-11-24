"""New subject-heldout anchor/progression audit. No new fit, thresholds, or model selection."""
from core import *
from association import runs
def main():
 assert not (E/'HELDOUT.json').exists();torch.set_num_threads(4);records=[];partials=[];newpred=[]
 labels={r['clip_id']:r for r in read(E/'REVIEW_LABELS.json')['rows']}
 for fold in [[1,2,3],[4,5,6],[7,8,9]]:
  name='heldout_'+'_'.join(map(str,fold));h=ProgressHead();h.load_state_dict(torch.load(E/'runs'/name/'model.pt',map_location='cpu',weights_only=True));h.eval()
  with torch.inference_mode():
   for r in read(E/'INPUTS.json')['rows']:
    if r['person'] not in fold:continue
    z=np.load(r['path']);f=np.load(E/'features'/name/f"{r['clip_id']}_{r['kind']}.npz")['features'];p=h(torch.tensor(f)).softmax(-1).numpy();dest=E/'heldout_predictions'/f"{r['clip_id']}_{r['kind']}.npz";dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,prob=p);newpred.append(dict(path=str(dest),sha256=sha(dest)))
    if r['kind'] in ['full_normal','actual_crop']:
     seen=set()
     for a in labels[r['clip_id']]['anchors']:
      ids=np.flatnonzero(z['frame_id']==a['frame'])
      if not len(ids) or a['label']<0:continue
      i=int(ids[0]);pred=int(p[i].argmax());y=a['label'];records.append(dict(clip=r['clip_id'],person=r['person'],kind=r['kind'],frame=a['frame'],label=y,pred=pred,correct=pred==y,event_correct=(pred-1)//2==(y-1)//2 if y else pred==0,core_probability=float(p[i,y]) if y and y%2==0 else None))
    if r['kind'].startswith('partial_before_core'):
     c=int(r['kind'].split('core')[1]);positive=runs(p[:,c]>=.65,z['timestamp_us'],60);partials.append(dict(clip=r['clip_id'],person=r['person'],kind=r['kind'],core_class=c,wrong_core_claim=bool(positive),runs=positive,source_frames=z['frame_id'].tolist()))
 summary={}
 for kind in ['full_normal','actual_crop']:
  rr=[r for r in records if r['kind']==kind];core=[r for r in rr if r['label']>0 and r['label']%2==0];summary[kind]=dict(anchors=len(rr),exact=sum(r['correct'] for r in rr),event_identity=sum(r['event_correct'] for r in rr),core_anchors=len(core),core_above_threshold=sum(r['core_probability']>=.65 for r in core))
 write(E/'HELDOUT.json',dict(summary=summary,anchors=records,partial_controls=partials,partial_wrong_core=sum(r['wrong_core_claim'] for r in partials),partial_total=len(partials),scope='New progression head and geometry encoder both exclude each heldout FIT person group. This is not whole feedback accuracy or independent expert truth.',predictions=newpred))
 print(json.dumps(dict(summary=summary,partial_wrong_core=sum(r['wrong_core_claim'] for r in partials),partial_total=len(partials)),ensure_ascii=False))
if __name__=='__main__':main()
