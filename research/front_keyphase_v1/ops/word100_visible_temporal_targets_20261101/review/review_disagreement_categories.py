from pathlib import Path
import json
U=Path(__file__).resolve().parent.parent
def read(p):return json.loads(Path(p).read_text('utf8'))
def main():
 pp={r['sample_id']:r for f in ['LABELS_A.json','LABELS_B.json'] for r in read(U/'review'/f)['rows']};ss=read(U/'review/LABELS_SECONDARY.json')['rows']
 groups={k:dict(both_known=0,disagree=0,one_unknown=0,both_unknown=0) for k in ['shape_bit','motion_category','bilateral_category','roi_center','roi_window']};events=[]
 def add(key,a,b,sample,side):
  d=groups[key]
  if a is None or b is None:d['both_unknown' if a is None and b is None else 'one_unknown']+=1
  else:d['both_known']+=1;d['disagree']+=int(a!=b)
  if a!=b:events.append(dict(sample_id=sample,group=key,side=side,primary=a,secondary=b))
 for q in ss:
  p=pp[q['sample_id']]
  for h in 'RL':
   for k in range(3):add('shape_bit',p['center_shape'][h][k],q['center_shape'][h][k],q['sample_id'],h+str(k))
   add('motion_category',p['motion'][h],q['motion'][h],q['sample_id'],h)
   for k in ['roi_center','roi_window']:add(k,p[k][h],q[k][h],q['sample_id'],h)
  add('bilateral_category',p['center_relation'],q['center_relation'],q['sample_id'],'both')
 out=dict(status='DESCRIPTIVE_RAW_CATEGORY_AUDIT',groups=groups,events=events,gate='Original <=5% known binary cell disagreement gate unchanged; one-hot cells are correlated. These raw counts expose category-level uncertainty. Disputed target cells are withheld; no labels rewritten.',human_independent=False)
 with (U/'review/CATEGORY_AGREEMENT.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False)
 print(json.dumps(groups))
if __name__=='__main__':main()
