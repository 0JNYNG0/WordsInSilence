from pathlib import Path
import json,hashlib,collections
U=Path(__file__).resolve().parents[1]
fields=['R_index_only','L_index_only','R_open_four','L_open_four','R_face_overlap','L_face_overlap','hands_overlap']
mine=json.loads((U/'review/IMAGE_LABELS_FIRST_PASS.json').read_text(encoding='utf-8'))
root=json.loads((U/'ROOT_VISUAL_LABELS_RAW.json').read_text(encoding='utf-8'))
dif=[];matches=0;knownmatches=0;bothnull=0
for row in mine['rows']:
 vals=root['rows_by_alias'][row['alias']]['values'][[5,8,11].index(row['ordinal'])]
 for k,v in zip(fields,vals):
  if row[k]!=v: dif.append({'alias':row['alias'],'ordinal':row['ordinal'],'feature':k,'review':row[k],'root':v})
  else:
   matches+=1
   if v is None:bothnull+=1
   else:knownmatches+=1
seqdif=[]
for row in mine['sequences']:
 for i,h in enumerate(['R','L']):
  v=root['rows_by_alias'][row['alias']]['seq'][i]
  if row[h]!=v:seqdif.append({'alias':row['alias'],'hand':h,'review':row[h],'root':v})
out={'scope':'24 new X clips; agreement of two Codex drafts, not accuracy or independent truth','static_slots':len(mine['rows'])*7,'static_equal':matches,'known_equal':knownmatches,'both_unknown':bothnull,'static_disagreements':dif,'sequence_slots':len(mine['sequences'])*2,'sequence_disagreements':seqdif,'first_pass_sha256':hashlib.sha256((U/'review/IMAGE_LABELS_FIRST_PASS.json').read_bytes()).hexdigest(),'root_draft_sha256':hashlib.sha256((U/'ROOT_VISUAL_LABELS_RAW.json').read_bytes()).hexdigest()}
(U/'review/FIRST_PASS_COMPARISON.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=True,indent=2))
