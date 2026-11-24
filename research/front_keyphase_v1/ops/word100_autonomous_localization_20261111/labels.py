from localization_core import *
assert not (U/'LABELS_FROZEN.json').exists()
# Ranges below were reviewed on the original public RGB sheets before model scores.
specs={
(5,'0003'):(4,[([31,33],[38,40],0),([38,40],[44,46],2),([45,47],[50,52],0)],[[54,60]],[[18,23],[72,78]]),
(5,'0500'):(0,[([72,74],[78,80],0)],[[84,93]],[[39,45],[102,111]]),
(5,'0678'):(0,[([63,65],[71,73],0)],[[78,87]],[[30,36],[99,108]]),
(5,'1139'):(3,[([35,37],[41,43],0)],[[45,54]],[[18,24],[69,78]]),
(5,'1290'):(4,[([42,43],[44,45],0),([44,45],[48,49],2),([48,50],[54,55],0),([54,55],[60,62],2)],[[64,69]],[[27,33],[81,87]]),
(5,'2551'):(0,[],[[72,90]],[[42,48],[105,111]]),
(14,'0003'):(4,[([65,67],[74,76],0),([75,77],[83,85],2),([84,86],[91,93],0)],[[99,105]],[[36,45],[123,132]]),
(14,'0500'):(0,[([61,63],[68,70],0)],[[75,87]],[[24,33],[102,111]]),
(14,'0678'):(0,[([55,57],[67,70],0)],[[75,90]],[[12,21],[108,117]]),
(14,'1139'):(3,[([41,43],[49,52],0)],[[57,72]],[[18,24],[96,105]]),
(14,'1290'):(4,[([48,50],[53,54],0),([53,55],[56,58],2),([57,59],[62,64],0),([63,65],[68,70],2)],[[72,84]],[[27,33],[105,114]]),
(14,'2551'):(0,[],[[60,78]],[[18,27],[102,108]])}
fit=[];evaluation=[]
for r in read(U/'REVIEW_INPUTS.json')['rows']:
 w=r['word_id'][4:];axis,events,holds,bg=specs[(r['person'],w)];sheet=U/'public_rgb'/f"dense_{w}_{r['person']}.jpg";sheet=sheet if sheet.exists() else Path(r['sheet'])
 a=dict(clip_id=r['clip_id'],role=r['role'],person=r['person'],word_id=r['word_id'],axis=axis,moving='R',anchor='L',events=[dict(onset=s,offset=e,direction=d) for s,e,d in events],holds=holds,background=bg,source_sha256=r['source_sha256'],sheet=str(sheet),sheet_sha256=sha(sheet),notes='Observed 2D wrist/body relationship only. Handshape, physical contact and educational repetition counts are NOT labeled by these ranges; motion beyond reviewed spans remains unknown.')
 for e in a['events']:assert 0<=e['onset'][0]<=e['onset'][1]<=e['offset'][1]<r['original_frames']
 (fit if r['person']==5 else evaluation).append(a)
write(U/'FIT_LABELS.json',dict(rows=read(J/'ALL_FIT_BOUNDARIES.json')['rows']+fit,additional=fit,reviewer='Codex original RGB review, not independent human/expert',same100word_fit=True))
write(U/'EVALUATION_LABELS.json',dict(rows=evaluation,reviewer='Codex original RGB review before model scores; not independent human/expert',scope='Public record14 new fine motion annotations, not globally unexposed test'))
write(U/'LABELS_FROZEN.json',dict(files={str(U/n):sha(U/n) for n in ['FIT_LABELS.json','EVALUATION_LABELS.json','REVIEW_INPUTS.json']},fine_evaluation_records=[14],training_records=[1,3,5,7],evaluation_not_training=True))
event('labels_frozen',fit_videos=30,added_fit_events=sum(len(a['events']) for a in fit),new_eval_events=sum(len(a['events']) for a in evaluation))
