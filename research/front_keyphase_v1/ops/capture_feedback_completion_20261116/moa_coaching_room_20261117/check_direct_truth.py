from probe_endpoint_roles import *
def main():
 torch.set_num_threads(2);a=ShortArrivalLesson();b=RoleLesson();rows=[]
 truth=read(O/'basic_demo_feedback_20261115/source_corrected/SOURCE_TRUTH.json')['rows']
 for r in read(G/'frontal_scope_recovery_20261117/INPUTS.json')['rows']:
  if r['word']!='WORD1290':continue
  anchors=[t for t in truth if t['clip_id']==r['id']]
  if not anchors:continue
  z=dict(np.load(r['raw']));qa=a.analyze(z);qb=b.analyze(z)
  for t in anchors:
   i=int(np.flatnonzero(z['frame_id']==t['frame'])[0]);old=qa['relation_states'][i];new=qb['relation_states'][i];label=t['state']
   rows.append(dict(id=r['id'],frame=t['frame'],truth=label,before=old,after=new,new_false_near=new==2 and old!=2 and label!=2,lost_correct=old==label and new!=label))
 q=dict(passed=not any(r['new_false_near']or r['lost_correct']for r in rows),rows=rows,before_correct=sum(r['before']==r['truth']for r in rows),after_correct=sum(r['after']==r['truth']for r in rows),private_data=False,scope='Existing direct source frame labels; development evidence, no physical contact ground truth')
 (D/'checks').mkdir(exist_ok=True);(D/'checks/DIRECT_TRUTH_CHECK.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8');print('TRUTH',q['passed'],len(rows),q['before_correct'],q['after_correct'])
if __name__=='__main__':main()
