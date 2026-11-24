from common import *
sys.path.insert(0,str(G/'public_visibility_roles_20261116'))
from visibility import hand_evidence
def main():
 torch.set_num_threads(2);net=QualifiedFormLesson();src={r['id']:r for r in sources()};out=[]
 for cid,h,ids in [('NIA_SL_WORD1290_REAL13_L','L',range(48,65)),('NIA_SL_WORD1290_REAL11_R','R',range(62,73)),('NIA_SL_WORD1290_REAL11_F','R',range(48,72))]:
  z=dict(np.load(src[cid]['raw']));ch=channels(z);q=net.analyze(z);obs=net.shape.observations[h];vis=hand_evidence(ch,h)
  for i in ids:
   out.append(dict(id=cid,hand=h,frame=i,shape=q['shape_states'][h][i],form=q['hand_form_evidence'][h]['form'][i],plane=q['plane_states'][h][i],overlap=bool(vis['projected_overlap'][i]),missing=bool(vis['own_missing'][i]),conflict=bool(obs['projection_conflict'][i]),fingers=q['shape_fingers'][h][i]))
 write(E/'visibility_diagnosis.json',dict(rows=out));[print(r,flush=True)for r in out]
if __name__=='__main__':main()
