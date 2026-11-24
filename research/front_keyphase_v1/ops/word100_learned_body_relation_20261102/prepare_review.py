from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw,ImageFont
V=Path(__file__).resolve().parent;P=V.parent/'word100_learned_guide_feedback_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(n,x):
 with (V/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 rows=read(P/'INPUT_MANIFEST.json')['rows'];(V/'review').mkdir(exist_ok=True)
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
 for w in sorted({r['word_id'] for r in rows}):
  for people in [[1,2,3],[4,5,6],[7,8,9],[11,12,13]]:
   selected=[r for r in rows if r['word_id']==w and r['person'] in people]
   canvas=Image.new('RGB',(1440,960),'#ffffff');draw=ImageDraw.Draw(canvas)
   for y,r in enumerate(selected):
    for x,l in enumerate(r['labels']):
     im=Image.open(l['center_image']).convert('RGB');W,H=im.size
     # Analysis contact sheet only. No changing source pixels or training images.
     im=im.crop((int(W*.23),0,int(W*.77),int(H*.92)));im.thumbnail((475,288))
     canvas.paste(im,(x*480+(480-im.width)//2,y*320+29))
     draw.text((x*480+4,y*320+2),f"{w} ID{r['person']:02d} W{x} frame{l['center_frame']}",fill='black',font=font)
   canvas.save(V/'review'/f'{w}_{people[0]:02d}.jpg',quality=92)
 save('FROZEN_RELATION_PLAN.json',dict(
  reason='Actual-video removal retained all old supported similarities because positive bilateral/body relations were not evaluated.',
  targets=['R_hand_in_front_of_face','bilateral_upper_ends_converged','R_hand_at_L_arm'],
  definitions=['Right hand overlaps/closely approaches face region in front view; shoulder-side is negative; uncertain borders unknown.',
   'Upper finger ends of two hands visibly converge near one common point; not generic palm proximity, not proof of 3D touch.',
   'Right hand is visibly at/over the left forearm or elbow region; not proof of contact or correct travel.'],
  review='Saved public center RGB, model predictions hidden; same Codex visual physical labels, not independent expert truth.',
  fit_people=list(range(1,10)),development_people=[11,12,13],user_mentor_material=False,
  observed_branch_word_guide_inputs=False,prior_models_frozen=True,
  conditions=['geometry_only','rgb_geometry'],seeds=[42,43,44],epochs=120,optimizer='AdamW lr0.001 wd0.01',
  temporal_window=13,center_supervision_only=True,unknown_loss_masked=True,
  architecture='Small separate goal-blind neural relation head, same cached SignRep RGB and body-normalized geometry; preserve existing shape/phase branches.',
  decision='No seed selection. Fixed0.5 binary metrics, confidence0.2/0.8 reserved before results. Do not call physical edit a linguistically wrong sign.',
  no_extra_backbone_or_download=True,source_manifest_sha256=hashlib.sha256((P/'INPUT_MANIFEST.json').read_bytes()).hexdigest()))
 print('16 public review sheets prepared; no model predictions shown')
if __name__=='__main__':main()
