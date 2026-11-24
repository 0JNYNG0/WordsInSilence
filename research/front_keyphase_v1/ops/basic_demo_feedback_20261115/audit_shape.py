from core import *
def main():
 torch.set_num_threads(2);sys.path[:0]=[str(O/'word100_finger_repair_20261115'),str(O/'word100_clause_finger_diagnostics_20261114')]
 from finger_runtime import FingerObserver
 n=FingerObserver();rows=[]
 for r in read(U/'INPUTS.json')['rows']:
  ch=dict(np.load(r['channels']));z=dict(np.load(U/'cache'/f'{r["clip_id"]}.npz'));anchors=np.flatnonzero(z['y']>0)
  for hand in ['L','R']:
   p=n.infer(ch['tcn280'],hand);s=p['states'][anchors]
   rows.append(dict(clip_id=r['clip_id'],split=r['role'],hand=hand,anchors=anchors.tolist(),extended=int((s==1).sum()),flexed=int((s==0).sum()),unknown=int((s<0).sum()),states=s.tolist(),note='Source panels show coarse open hands; individual occluded digits not promoted to precise truth'))
 write(U/'SHAPE_READINESS.json',dict(rows=rows,scope='Existing observer output readiness; not an independent finger accuracy score',new_shape_training=False))
 print('SHAPE READINESS DONE',len(rows))
if __name__=='__main__':main()
