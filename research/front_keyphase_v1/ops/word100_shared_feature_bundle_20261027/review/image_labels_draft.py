"""Direct image review entries, not detector-derived or dictionary-derived labels."""
from pathlib import Path
import json,hashlib,datetime,re
U=Path(__file__).resolve().parents[1]
FIELDS=['R_index_only','L_index_only','R_open_four','L_open_four','R_face_overlap','L_face_overlap','hands_overlap']
# Values ordered exactly as FIELDS. '?' means unknown, never negative.
DATA={
 'X01':(['1100000','1100000','1100000'],[0,0], [4,5,8,11,13], 'Both visible index configurations remain separate in front of torso; no broad-open to grouped change witnessed.'),
 'X02':(['1100000','1100000','1100000'],[0,0], [3,5,8,11,13], 'Both index configurations alternate upper/lower positions; separate hand projections.'),
 'X03':(['1100000','1100000','1100000'],[0,0], [3,5,8,11,14], 'Both index configurations; no broad-open change among visible sampled states.'),
 'X04':(['0011100','1100000','1100010'],[1,1], [5,8,11], 'Open hands then index-only. Raw fixed sample 5 confirms broadly extended fingers; right fingertip projects at face edge. Sample11 left index projects over lower face.'),
 'X05':(['0011100','1100000','11000?0'],[1,1], [5,8,11], 'Open hands then index-only. Raw fixed sample5 reviewed; sample11 left fingertip near face edge is ambiguous at contact-sheet resolution.'),
 'X06':(['0011011','0011011','1100000'],[1,1], [5,8,11], 'Open hands alternate near face; left upper hand projects over lower face at samples5,8 and projections overlap. Both index-only later.'),
 'X07':(['1000100','1000001','1000001'],[0,0], [3,5,8,11,13], 'Right index near lower face then extends to curled left hand. Left index itself is curved, not clearly extended. No broad-open four state witnessed.'),
 'X08':(['1?0?000','1000000','1000001'],[0,0], [5,6,8,11,13], 'Sample5 right index is below chin; lowered left fingers insufficient for exact handshape. Sample8 hands separated, sample11 projections meet. No open-four transition witnessed in adequately visible central states.'),
 'X09':(['1000000','1000000','1000001'],[0,0], [3,5,8,11,13], 'Right index near upper neck then approaches curled left hand; sample8 gap remains, sample11 projections meet. No open-four transition witnessed.'),
 'X10':(['001?100','100?000','100?000'],[1,None], [5,8,11], 'Raw fixed sample5 shows edge-on extended right fingers overlapping mouth/chin, then right index-only with thumb free. Lowered left hand has visible multiple fingers but four-finger status is uncertain.'),
 'X11':(['001?100','100?000','100?000'],[1,None], [3,5,8,11], 'Right edge-on flat/open state at lower face followed by index-only; left hand remains low and partly edge-on.'),
 'X12':(['001?100','100?000','100?000'],[1,None], [5,8,11], 'Raw fixed sample5 confirms right fingers extended at lower face; later index-only. Lowered left finger extent incompletely visible.'),
 'X13':(['001?100','000?000','000?000'],[1,None], [5,8,11], 'Right open-four hand over nose/mouth becomes closed fist near chest. Left rests low with incomplete finger visibility.'),
 'X14':(['001?100','000?000','000?000'],[1,None], [5,8,11], 'Right open hand over lower central face changes to fist; low left hand shape remains partly hidden.'),
 'X15':(['001?100','000?000','000?000'],[1,None], [5,8,11], 'Right open hand at face changes to fist below face; left lowered edge-on hand is not a full four-finger observation.'),
 'X16':(['0010000','001000?','001000?'],[0,0], [3,5,8,11,13], 'Right flat open hand lies over left forearm, left fist. Raw sample8 places right hand at left wrist boundary; hand-vs-forearm overlap boundary remains ambiguous. No broad-open/grouped change observed.'),
 'X17':(['0010000','001000?','001000?'],[0,0], [3,5,8,11], 'Right flat hand over left forearm, left fist. Later wrist-edge projection is ambiguous for hand-hand overlap. No open/grouped state change witnessed.'),
 'X18':(['0010000','001000?','0010001'],[0,0], [3,5,8,11], 'Right flat hand over left forearm, left fist. Raw sample11 shows hand projections meeting at the fist base; sample8 wrist/forearm boundary remains uncertain. No open/grouped change observed.'),
 'X19':(['0000010','0000111','0000010'],[0,0], [3,5,8,11,13], 'Both fists with thumbs extended; alternate up/down. Sample8 both hand projections meet over face; sample11 left fist over forehead, right below chin. No open-four change.'),
 'X20':(['0000010','0000100','0000010'],[0,0], [3,5,8,11,13], 'Fists with extended thumbs alternate heights. Left then right then left hand overlaps facial projection at selected samples; hands remain separate in those samples.'),
 'X21':(['0000010','0000100','0000010'],[0,0], [4,5,8,11,13], 'Thumb-up fists alternate in front of torso/face; no broad-open fingers. Selected hand projections are separate.'),
 'X22':(['0011111','0011000','0011000'],[0,0], [4,5,8,11], 'Flat open hands form a peak in front of lower face then separate sideways. Raw sample8 supports broadly extended stacked fingers. No grouped/curled transition observed.'),
 'X23':(['0011110','0011000','0011000'],[0,0], [1,3,5,8,11], 'Raw sample5 confirms fingertips at lower-face edges with a gap between hands. Both remain broadly extended after separating; no grouped/curled transition.'),
 'X24':(['0011111','0011110','0011000'],[0,0], [4,5,8,11], 'Open hands meet in peaked configuration over face then separate. Raw sample8 shows separate hands still projected over cheeks/nose edges; later clear of face. No grouped/curled change observed.'),
}
def main():
 rows=[];seq=[];pins={}
 for alias,(states,change,evidence,note) in DATA.items():
  contact=U/'visual_inputs'/alias/'detail_contact.png'
  pins[str(contact)]=hashlib.sha256(contact.read_bytes()).hexdigest()
  for ordinal,chars in zip([5,8,11],states):
   assert len(chars)==7
   matches=list((U/'visual_inputs'/alias).glob(f'frame_{ordinal:02d}_f*.png'))
   assert len(matches)==1
   frame=matches[0]
   pins[str(frame)]=hashlib.sha256(frame.read_bytes()).hexdigest()
   row={'alias':alias,'ordinal':ordinal,'frame_id':int(re.search(r'_f(\d+)',frame.name)[1]),'image':str(frame),'note':note}
   row.update({key:None if value=='?' else int(value) for key,value in zip(FIELDS,chars)})
   rows.append(row)
  seq.append({'alias':alias,'R':change[0],'L':change[1],'evidence_ordinals':evidence,'note':note,'scope':'Only visible change in 16 listed sampled images; zero is not continuous-video absence.'})
 out={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'image_first_separate_Codex_draft','review_scope':'New X aliases only. Root labels and SOURCE_PROFILES not consulted. Manifest word IDs seen before images, so not fully blind. Two AI reviews do not create independent human truth. No correctness or contact labels.','rubric_sha256':hashlib.sha256((U/'RUBRIC.json').read_bytes()).hexdigest(),'rows':rows,'sequences':seq,'image_pins':pins}
 (U/'review'/'IMAGE_LABELS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(len(DATA),len(rows),len(seq))
if __name__=='__main__':main()
