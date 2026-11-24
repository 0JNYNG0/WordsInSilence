from common_action import *
import shutil,collections
def main():
 assert not (U/'FROZEN.json').exists()
 (U/'before').mkdir(exist_ok=True)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']:
  shutil.copyfile(p,U/'before'/p.name)
 movements={r['clip_id']:r for r in read(M/'READY.json')['rows']};rr=[]
 for r in read(B/'READY.json')['rows']:
  if not r['has_labels']:continue
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);assert np.isfinite(z['x_full']).all()
  pairs=movements.get(r['clip_id'],{}).get('pairs',[])
  for p in pairs:assert 0<=p['a']<p['b']<len(z['frame_id']) and z['frame_id'][p['a']]==p['frame_a'] and z['frame_id'][p['b']]==p['frame_b']
  assert r['role'] in ['fit','development'];assert (r['person']<=9)==(r['role']=='fit')
  rr.append(dict(r,pairs=pairs))
 fit=[r for r in rr if r['role']=='fit'];dev=[r for r in rr if r['role']=='development']
 assert not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev}
 assert not {r['person'] for r in fit}&{r['person'] for r in dev}
 write(U/'READY.json',dict(rows=rr,scope='Reused reviewed public annotations; 12 expressions with attributes; movement supervision in four expressions. Historical DEV only, not fresh test.'))
 observed=[];targets=[]
 for r in fit:
  z=np.load(r['path'])
  for view in ['full','upper']:
   for p in r['pairs']:
    for rev in [False,True]:
     g,ok,_=motion.features(physical(z['x_'+view]),z['timestamp_us'],p['a'],p['b'],motion.KINDS[p['axis']],reverse=rev)
     if ok:observed.append(g);targets.append(2-p['target'] if rev else p['target'])
 data=np.asarray(observed);mean=data.mean(0);std=np.maximum(data.std(0),.05)
 write(U/'NORMALIZATION.json',dict(mean=mean.tolist(),std=std.tolist(),motion_class_counts=np.bincount(targets,minlength=3).tolist(),fit_only=True))
 # Raw visual verification sampled before predictions; inherited labels retain original reviewer/provenance.
 review=[dict(clip_id='NIA_SL_WORD'+w+'_REAL02_F',sheet=str(M/'review'/('NIA_SL_WORD'+w+'_REAL02_F.jpg')),observations=txt) for w,txt in [
 ('0739','f46-58 open long fingers before face; f64-70 closes and lowers; f70-83 forearm becomes upright; not all-thumb geometry truth.'),
 ('1133','f49-62 open right hand moves along left forearm toward wrist; f66-74 broadly holds; left hand stays closed; contact pressure not observable.'),
 ('2523','f53-55 hands form apex in projection; f55-62 separate; f64-73 broadly maintained. Final thumb/all-tip contact not verified. No convergence label invented.')]]
 for r in review:r['sha256']=sha(r['sheet'])
 write(U/'ANNOTATION_AUDIT.json',dict(status='PASS_WITH_SCOPE_LIMITS',fit_clips=len(fit),dev_clips=len(dev),fit_ids=sorted({r['person'] for r in fit}),dev_ids=sorted({r['person'] for r in dev}),attribute_expressions=sorted({r['word_id'] for r in rr}),fit_motion_pairs=sum(len(r['pairs']) for r in fit),dev_motion_pairs=sum(len(r['pairs']) for r in dev),reverified_rgb=review,reviewer='same Codex visual review, not independent human expert',inherited_sources=[str(B/'ANNOTATION_REVIEW.json'),str(M/'READY.json'),str(M/'REVIEW.json')],unknown_policy='No invisible thumb, exact touch, repetition rule or final all-tip convergence labels. Weak DTW stage labels retain weights; not promoted to direct truth.',private_used=False))
 inventory=[
 ['long-finger shape','7 existing attributes incl relative height','reviewed temporal labels','learned; current web','thumb and full named shapes not covered'],
 ['hand/body motion','5 relative relation kinds','reviewed interval direction/hold','old MotionNet experiment; explicit along-forearm rule in web','joint encoder/head comparison here'],
 ['bimanual relation','height and separation changes','reviewed height/interval labels','height learned; separation experiment','exact contact and per-hand tip convergence absent'],
 ['event order/completion','event progress/core','sparse public anchors; deletion controls','isolated completion guard verified','not expert correctness; keep separate from shape'],
 ['guide linkage','structured clause/component contract','source description vs observed variant','implemented for four expressions','no language-model comprehension claim']]
 write(U/'COMPONENT_INVENTORY.json',inventory)
 protected=[BASE,D/'consumer.py',D/'CARDS.json',J/'tcn_runs/finetune_C_20261013/model.pt']
 cfg=dict(authorization='User 2026-11-10 approved all four stages and retrospective history backfill.',conditions=['heads_only','joint_finetune'],primary='joint_finetune_42',seeds=[42,43,44],epochs=30,batch=6,lr_head=.002,lr_base=.0001,weight_decay=.001,loss_weights=dict(movement=1,attributes=.5,stage=.1,latent_preservation=1,attribute_preservation=.25),normalization=str(U/'NORMALIZATION.json'),confidence=.65,base_heads_frozen=True,base_finetune_parts=['input','temporal','context'],selection='Fixed final epochs; no DEV, ID10 or user tuning; no best seed selection.',tests=['reviewed handshape/height','reviewed motion full/upper/reversed','actual automatic windows shape-only/direction-only/combined','normal speed/crossing/omission/missing full path'],adoption_gate='No normal or original attribute/stage regression, no weaker important-error detection, missing retained; otherwise no default change.',motion_input='Body-normalized relative geometry + current latent endpoints/difference, no target/gloss/person/frame index. Target masks stage only.',limitations='Twelve expressions attribute labels; four motion expressions; historical DEV; inherited base saw FIT IDs. Not all100 semantic labels or expert educational truth.',protected={str(p):sha(p) for p in protected},source_pins={str(p):sha(p) for p in [B/'READY.json',M/'READY.json',U/'READY.json',U/'ANNOTATION_AUDIT.json',U/'NORMALIZATION.json']},code_pins={str(U/f):sha(U/f) for f in ['common_action.py','prepare.py']},private=False,default_web_change=False)
 write(U/'FROZEN.json',cfg)
 write(U/'CURRENT_EXECUTION.json',dict(status='ANNOTATIONS_VERIFIED_TRAINING_READY',pid=None))
 event('prepared',fit_clips=len(fit),dev_clips=len(dev),fit_motion_variants=len(data))
if __name__=='__main__':main()
