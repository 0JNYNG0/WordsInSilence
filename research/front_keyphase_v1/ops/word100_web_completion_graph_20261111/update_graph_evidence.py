from graph import *
O=U.parent
audit=read(U/'reviews/label_audit.json')
for src in audit['sources']:
 assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
cap=dict(scope='Common descriptor capabilities for first four supported words; never whole-sign correctness',rows=audit['feature_support_gaps'],clause_support=audit['clause_support'],correction=dict(previous_interpretation='four_open alone can assess terminal fingertip gathering',corrected='false: finger extension and thumb-to-finger aperture are different features; extended fingers can coexist with fingertip gathering',new_common_feature='thumb_finger_projected_aperture',direct_evidence=str(U/'reviews/egypt_fit_shape_audit.json'),contact='not proven',model_support='not yet validated'),visual_second_review=dict(reviewer='root Codex after separate agent; not independent human',panels=['REAL01_sequence.png','REAL04_sequence.png','REAL07_sequence.png'],observation='Wide thumb-to-finger gap narrows during separating/lowering. Four fingers remain a largely extended bundle. Exact hidden fingertip contact and individual joint angles not resolved.'),policy='Missing capability cannot be passed or corrected by another feature output')
write(U/'CAPABILITY_CONTRACT.json',cap)
if next(n for n in read(U/'TASK_GRAPH.json')['nodes'] if n['id']=='G1')['status']!='PASS':
 record('G1','PASS',dict(provenance=True,guide_variant_consistency=True,unknown_not_negative=True),[U/'reviews/label_audit.json',U/'LABEL_CHECK.json',U/'CAPABILITY_CONTRACT.json'],'Existing direct body-context labels verified for candidate linkage; thumb-aperture gap separately identified and NOT promoted as supported. Unsupported source requirements excluded from training truth.')
for label,path,note in [
 ('body_only',U/'LINKER_EVAL_COMPLETE.json','Body-only candidate loses normal/speed evidence and some anchors; not adopted.'),
 ('block_temporal',U/'BLOCK_EVAL_COMPLETE.json','Actual block input recomputed, role-only fine tuning; some anchors regress, order ambiguity remains; not adopted.'),
 ('source_decoder',U/'SOURCE_DECODER_COMPLETE.json','Normal/speed preserved; reverse diagnostic remains 2 supports and 7 ambiguous; not sufficient.'),
 ('joint_order',U/'JOINT_ORDER_COMPLETE.json','Normal/speed9 pairs preserved all seeds; 3 reversed now detected, 5 supports remain on compound reverse. Partial diagnostic gain; full gate not passed.')]:
 if path.exists() and not any(a['note'].startswith(label+':') for a in next(n for n in read(U/'TASK_GRAPH.json')['nodes'] if n['id']=='G2')['attempts']):
  record('G2','REWORK',dict(normal_preserved=label in ['source_decoder','joint_order'],false_support_reduced=False,not_abstention_only=label=='joint_order',three_seeds=label!='source_decoder'),[path],label+': '+note)
write(U/'CURRENT_EXECUTION.json',dict(status='G2_REWORK_SEQUENCE_TRUTH_AND_FEATURE_GAP_AUDIT',pid=None,training_running=False,web_changed=False,camera=False,private_access=False,graph=str(U/'TASK_GRAPH.json'),pending='Verify physical source sequence truth and thumb aperture observability; no final test opened before model gate'))
