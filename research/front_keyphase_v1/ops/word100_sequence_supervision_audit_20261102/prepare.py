"""Prepare weak sequence witnesses; do not call sparse samples exhaustive gold."""
from pathlib import Path
import sys,collections
J=Path(__file__).resolve().parent;F=J.parent/'word100_separate_action_context_20261102';sys.path.insert(0,str(F));from model import *

def main():
    assert not (J/'SEQUENCE_INPUT_AUDIT.json').exists();rows=fit_rows();out=[]
    for r in rows:
        witnesses=[(int(i),int(r['mapping'][i]),int(r['episode'][i])) for i in np.flatnonzero((r['episode']>=0)&(r['episode']<8))]
        sequence=[];runs=[]
        for i,source,action in witnesses:
            if not sequence or action!=sequence[-1]:sequence.append(action);runs.append(dict(action=action,frames=[],source_frames=[]))
            runs[-1]['frames'].append(i);runs[-1]['source_frames'].append(source)
        assert sequence and len(sequence)<=len(r['x'])
        out.append(dict(clip_id=r['clip_id'],word=r['word_id'],person=r['person'],kind=r['kind'],frames=len(r['x']),witness_sequence=sequence,witness_runs=runs,
          unseen_action_absence_proven=False,complete_transcript_gold=False,edit_seam_unlabeled=r['seam_unlabeled'],all_images_available=bool(r['available'].all()),
          input=r['input'],input_sha256=r['input_sha256'],cache=r['cache'],cache_sha256=r['cache_sha256']))
    native=collections.defaultdict(list)
    for r in read(D/'ACTION_TARGETS.json')['rows']:
        seq=[]
        for p in r['points']:
            if p['action_family']<8 and (not seq or p['action_family']!=seq[-1]):seq.append(p['action_family'])
        native[r['word_id']].append(dict(person=r['person'],sequence=seq))
    consensus={word:dict(rows=rr,all9_observed_sequences_agree=len({tuple(r['sequence']) for r in rr})==1) for word,rr in native.items()}
    assert len(out)==216 and all(r['person'] in range(1,10) for r in out) and all(v['all9_observed_sequences_agree'] for v in consensus.values())
    sources=[dict(url='https://www.cs.toronto.edu/~graves/icml_2006.pdf',title='Graves et al. ICML2006 CTC',verified_claim='Sequence objective sums possible alignments for an ordered target label sequence; original evidence is speech, not a KSL feedback accuracy guarantee.'),
      dict(url='https://docs.pytorch.org/docs/2.15/generated/torch.nn.CTCLoss.html',title='Official PyTorch CTCLoss',verified_claim='Continuous input and target sequence, blank separate from labels, targets cannot contain blank, target length bounded by input length.')]
    result=dict(at=clock(),status='WEAK_SEQUENCE_WITNESSES_PREPARED_NOT_GOLD',rows=out,original_consensus=consensus,
      old_observation_issue='F normal rest readout30/107; earlier T result already reports normal rest96/107 for its retained study. Do not silently relearn or replace the retained observation with an inferior role head.',
      objective_gap='F optimizes sparse per-time action CE and role CE, while final support is an all-occurrence order decision. RNNs use temporal context, but this trial has no explicit whole-label-sequence loss. Earlier pair auxiliary experiments do not establish this whole-sequence objective.',
      sources=sources,
      next_contract=[
       'Reuse retained physical/rest/shape/body/movement observations as explicit concurrent evidence, with model disagreement held rather than silently overwritten.',
       'These216ordered witness lists are weak supervision proposals, NOT proof all intermediate frames are meaningless or no other action occurred. Native agreement across9recordedIDs does not establish linguistic necessity.',
       'If using CTC, use an auxiliary sequence branch with sparse direct-observation supervision retained. CTC blank is not rest, missingness, unimportant action, or an action-duration boundary.',
       'Do not enforce expected order at decoding or use reference-directed beam search to hide a reversal. Short contrary content, image gaps and actual motion cuts remain separately guarded.',
       'Review whether weak sequence supervision is supported for each input/edited seam before freezing a small CE-only versus auxiliary-sequence comparison. No new training has been run here.'
      ],pins={str(p):sha(p) for p in [D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',B/'PREPARED.json',F/'FROZEN.json',F/'VERIFIED.json',F/'train.py']},new_dense_labels=0,new_training=0,new_DEV_evaluation=0,whole_goal_complete=False)
    exclusive(J/'SEQUENCE_INPUT_AUDIT.json',result);print('Prepared',len(out),'FIT weak witness sequences; native9ID consensus', {w:v['rows'][0]['sequence'] for w,v in consensus.items()})
if __name__=='__main__':main()
