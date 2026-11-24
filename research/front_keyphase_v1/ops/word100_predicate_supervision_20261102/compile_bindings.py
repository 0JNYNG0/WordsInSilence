"""Verify direct source-component witnesses without promoting them to full rules."""
from collections import Counter,defaultdict
from compile_supervision import U,O,read,sha,save
import numpy as np

def main():
    manifest_path=O/'word100_action_episode_supervision_20261102/REVIEW_MANIFEST.json'
    manifest={r['clip_id']:r for r in read(manifest_path)['rows']}
    inputs={r['clip_id']:r for r in read(U/'SUPERVISION.json')['sources']}
    clauses={r['clause_id']:r for r in read(U/'CLAUSE_COMPONENT_LINKS.json')['rows']}
    paths=[U/'DIRECT_BINDINGS_ROOT.json',U/'DIRECT_BINDINGS_1567.json',U/'DIRECT_BINDINGS_2523.json']
    rows=[];seen=[];counts=Counter();coverage=defaultdict(Counter);pins={str(p):sha(p) for p in paths+[manifest_path,U/'BINDING_REVIEW_PLAN.json']}
    for path in paths:
        for i,r in enumerate(read(path)['rows']):
            cid=r['clip_id'];assert cid not in seen;seen.append(cid)
            m=manifest[cid];assert r['role']=='fit' and r['person']==m['person'] and 1<=r['person']<=9
            sheet=r['reviewed_sheet'];assert sheet['path']==m['sheet'] and sha(sheet['path'])==sheet['sha256']==m['sheet_sha256']
            frames=r.get('reviewed_frames',r.get('inspected_native_frames',r.get('reviewed_native_frames')))
            by={p['frame']:p for p in m['frames']};assert sorted(frames)==sorted(by)
            a=np.load(inputs[cid]['input'],allow_pickle=False)
            extra=r.get('additional_full_frame_views',[])+([r['additional_large_frame_review']] if 'additional_large_frame_review' in r else [])
            for e in extra:assert sha(e['path'])==e['sha256']==by[e['frame']]['sha256']
            for j,c in enumerate(r['claims']):
                assert c['clause_id'] in clauses and c['kind'] in ['point','ordered_witness_sequence']
                assert c['support'] in ['supported','visibly_contrary','unknown']
                witness=c['witness_frames'];assert len(witness)>0 and sorted(set(witness))==witness
                assert set(witness)<=set(by)
                for f in witness:assert sha(by[f]['path'])==by[f]['sha256']
                if c['kind']=='ordered_witness_sequence':assert len(witness)>=2
                if 'ordered_states' in c:assert [s['frame'] for s in c['ordered_states']]==witness
                source=clauses[c['clause_id']]['source'];assert sha(source['path'])==source['sha256']
                # Every source phrase remains context. This new label concerns
                # explicitly stated visible components only, never entire text.
                row=dict(id=f'{cid}:claim{j}',clip_id=cid,word_id=r['word_id'],person=r['person'],role='fit',
                    clause_id=c['clause_id'],source_text=clauses[c['clause_id']]['text'],source=source,
                    component=c['supported_component'],kind=c['kind'],support=c['support'],
                    witness_frames=witness,witness_timestamps_us=[int(a['timestamp_us'][f]) for f in witness],
                    unsupported=c['unsupported'],boundary=c['boundary'],
                    provenance=dict(path=str(path),pointer=f'/rows/{i}/claims/{j}',sha256=sha(path)),
                    point_mask=(c['support']!='unknown') if c['kind']=='point' else None,
                    sequence_mask=(c['support']!='unknown') if c['kind']=='ordered_witness_sequence' else None,
                    source_clause_truth=None,educational_error_label=None,dense_frame_supervision=False,
                    inspected_images=[by[f] for f in witness])
                if 'observable_values' in c:row['observable_values']=c['observable_values']
                if 'ordered_states' in c:row['ordered_states']=c['ordered_states']
                rows.append(row);counts[c['support']]+=1;coverage[c['clause_id']][c['support']]+=1
    assert len(seen)==36 and set(seen)==set(manifest) and len(rows)==117
    for r in rows:
        if r['support']=='unknown':assert not (r['point_mask'] or r['sequence_mask'])
        assert r['source_clause_truth'] is None and r['educational_error_label'] is None
    save('BINDING_WITNESSES.json',dict(rows=rows,pins=pins,scope='Explicit physical component support attached to source text; not text-wide truth or learner-error labels',
        context_only_source_promoted=False,old_event_family_labels_changed=False,unshown_frames_labelled=False))
    counts_by_clause={k:dict(v) for k,v in coverage.items()}
    save('BINDING_VERIFIED.json',dict(status='36_FIT_DIRECT_COMPONENT_BINDINGS_SOURCE_TIME_VERIFIED',clips=36,
        sampled_images=sum(len(r['frames']) for r in manifest.values()),claims=len(rows),counts=dict(counts),
        per_clause=counts_by_clause,unknown_not_negative=True,point_and_sequence_separate=True,
        native_timestamps_preserved=True,first_last_witness_not_exact_boundary=True,
        labels_reviewed_by='root plus two Codex agents; no independent human/KSL expert gold',
        new_training=0,new_candidate_inference=0,whole_goal_complete=False))
    print(dict(clips=36,claims=len(rows),counts=dict(counts),per_clause=counts_by_clause))
if __name__=='__main__':main()
