"""Keep action identity separate from the anatomical point used to measure direction."""
from boundary_direction_core import *
from collections import Counter
meta=read(M/'INPUTS_COMPLETE.json');audit=read(M/'INPUT_AUDIT.json');z=np.load(meta['path'])
review=dict(at=stamp(),reviewer='root Codex inspected source RGB contact sheets; not independent KSL expert',training_status='SCOPED_AXES_1_TO_4_ONLY',findings=[
 'In REAL05/08/09 the fingers fold and change orientation while the body wrist height is stable or slightly increases relative to shoulders. Hand-down wording does not imply every body wrist sample descends.',
 'The original action-extent annotations remain useful to locate the observed hand-closing action. They do not automatically supply a negative body-wrist-height direction target.',
 'Do not alter physical labels by taking the model prediction or coordinate sign as truth. Preserve initial mismatch audit. New family-derived axis0 targets are withheld.',
 'Existing broad direction targets include one training pair whose wrist displacement is near zero/opposite (REAL05 52..64). This limits interpreting old limited-label agreement as independent physical-direction accuracy.'
 ],sources=[dict(path=str(M/f'conflict_REAL{n:02d}.jpg'),sha256=sha(M/f'conflict_REAL{n:02d}.jpg'))for n in [5,8,9]],generic_fix='Define observation point/group and body anchor explicitly. Handshape transition, palm/fingertip height, body wrist height and forearm orientation are separate quantities. No per-word score exception.',scope='For this new fine-tuning experiment only, use verified forearm orientation, along-other-forearm, hand-separation and relative-height axes1..4. Axis0 predictions remain the original frozen output and are not claimed repaired.',no_relabel_from_coordinates=True,old_annotations_unchanged=True,old_checkpoints_unchanged=True,web_changed=False)
write(M/'TARGET_REVIEW.json',review)
eligible=np.array([r['axis']in[1,2,3,4]for r in meta['rows']]);assert not any(r['axis']!=0 for r in audit['coordinate_direction_conflicts'])
rows=[dict(r,input_index=i)for i,r in enumerate(meta['rows'])if eligible[i]];path=M/'scoped_inputs.npz';assert not path.exists();np.savez_compressed(path,**{k:z[k][eligible]for k in z.files})
write(M/'SCOPED_INPUTS.json',dict(at=stamp(),path=str(path),sha256=sha(path),rows=rows,source=str(M/'INPUTS_COMPLETE.json'),source_sha256=sha(M/'INPUTS_COMPLETE.json'),held_out_axis0=int((~eligible).sum()),counts=dict(Counter(r['split']+'_'+r['origin']for r in rows)),word_specific_branch=False,axis0_original_frozen=True,requires_context=read(M/'FROZEN.json'),review=str(M/'TARGET_REVIEW.json')))
print('Verified typed-direction inputs',len(rows),'axis0 withheld',int((~eligible).sum()))
