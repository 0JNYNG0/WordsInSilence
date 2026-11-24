from common import *
from copy import deepcopy
oldp=PREV/'feedback_contract/GUIDE_FINGER_BINDINGS.json';old=read(oldp)
labels=read(U/'shape_recovery/EXTRA_LABELS.json')['rows'];out=[];unbound=[]
for r in labels:
    if r['person']!=1 or r['word'] not in {'WORD0739','WORD1133','WORD1567','WORD2523'}:continue
    matches=[g for g in old['stages'] if g['word_id']==r['word'] and g['native_frame_interval_inclusive'][0]<=r['frame']<=g['native_frame_interval_inclusive'][1]]
    if not matches:unbound.append(dict(clip_id=r['clip_id'],frame=r['frame'],side=r['side'],reason='No saved native interval; no extrapolation'));continue
    for g in matches:
      assert g['reference_clip_id']==r['clip_id'] and g['reference_source_sha256']==r['source_sha256']
      for finger,state in zip(['thumb','index','middle','ring','little'],r['states']):
        if state=='U':continue
        out.append(dict(binding_id=f"{g['stage_id']}:{r['frame']}:{r['side']}:{finger}",word_id=r['word'],stage_id=g['stage_id'],hand=r['side'],finger=finger,reference_state=state,action_phrase=g['action_phrase'],reference_anchors=[dict(clip_id=r['clip_id'],frame=r['frame'],native_time_us=r['pts_us'],source_sha256=r['source_sha256'],image=r['image'],image_sha256=r['image_sha256'],annotation_file=str(U/'shape_recovery/EXTRA_LABELS.json'),annotation_sha256=sha(U/'shape_recovery/EXTRA_LABELS.json'))],clause_ids=[],source_directly_names_finger_state=False,source_requirement_level='reference_rgb_component_only',comparison_scope='sparse_reference_anchor_not_mandatory_word_rule',stage_interval_provenance=g,educational_tolerance_verified=False,attribute_validated=False,phase_binding_validated=False))
merged={x['binding_id']:x for x in old['bindings']}
for x in out:
    if x['binding_id']in merged:assert merged[x['binding_id']]['reference_state']==x['reference_state']
    else:merged[x['binding_id']]=x
write(U/'feedback_contract/GUIDE_FINGER_BINDINGS.json',dict(status='SOURCE_ANCHOR_ADDITION_ONLY_NOT_RUNTIME_VALIDATED',parent_path=str(oldp),parent_sha256=sha(oldp),stages=old['stages'],bindings=list(merged.values()),new_bindings=out,unbound=unbound,old_unbound_preserved=old['unbound_anchors'],note='Egypt f49/f55 direct evidence joins existing interval; f59 not pulled into it. No new mandatory educational rule or automated phase correctness claimed.'))
# Audit only already-admitted metadata: never open final8 or sealed raw sources.
rr=read(U/'DATA_READY.json')['rows'];groups={}
for role in ['fit','development']:
    rs=[r for r in rr if r['role']==role];groups[role]=dict(clips=len(rs),record_ids=sorted({r['person'] for r in rs}),sources=len({r['source_sha256'] for r in rs}),status='Encoder/training exposed'if role=='fit'else'Historical or exposed development/preparation; common300 repeatedly evaluated, selected_extra18 prepared/annotated')
write(U/'EXPOSURE_AUDIT.json',dict(status='NO_FRESH_EVALUATION_OPENED',metadata_source=str(U/'DATA_READY.json'),metadata_sha256=sha(U/'DATA_READY.json'),groups=groups,eligible_fresh_confirmed_here=0,statement='This admitted cache supplies no fresh independent test. This does not prove other eligible public material does not exist.',sealed_and_final8_opened=False,next_gate='After P3/P4 pass, inspect authorization/exposure metadata for an eligible frozen final set before reading payloads.'))
event('contracts_prepared',new_source_bindings=len(out),merged_bindings=len(merged),fresh_evaluation_opened=False)
