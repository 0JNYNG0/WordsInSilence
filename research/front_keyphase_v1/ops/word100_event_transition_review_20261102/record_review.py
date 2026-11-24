"""Manual native-RGB review decisions, not automatic label interpolation."""
from prepare_review import *

# Values were entered after inspecting all 40 chronological panels.  A role can
# be known while the boundary between two adjacent action families is unclear.
# Family -1 never means absence, fist, rest, or a wrong performance.
REVIEWS = {
 'WORD0739_02': dict(families={0:[42,44,45,60,61],1:[63,65,75],2:[78,80,81,98]}, main_unknown=[77], context=[40,100,101], rest=[],
   note='Open hand arrives before the first old witness. Closing/relocation, then elbow reorientation are visible. Final lowering retains the closed-hand appearance.'),
 'WORD0739_04': dict(families={0:[68,69,71,72,85],1:[89,90,92,93,104],2:[107,109,110,127,130]}, main_unknown=[88,106], context=[64,65,66,132,133,134,135], rest=[],
   note='Approach rises toward face; a closing hand then moves toward torso. Forearm changes from crosswise to upright. Return still has a similar closed hand.'),
 'WORD0739_08': dict(families={0:[42,44,45],1:[63,64,66,67,70,71,73],2:[76,77,96,97]}, main_unknown=[60,61,74], context=[40,100], rest=[],
   note='Existing f62 ACTION0 conflicts with visible folded fingers and descending hand; inspect the full-resolution source frame. This is our derived annotation, not an official AIHub error.'),
 'WORD1133_02': dict(families={3:[46,57,58],4:[62,73,75]}, main_unknown=[60], context=[78,79,80], rest=[],
   note='Traversal and near-wrist configuration form a continuous action. Do not require a pause. Both arms begin returning while the right hand still resembles the endpoint.'),
 'WORD1133_04': dict(families={3:[73,75,83],4:[89,90]}, main_unknown=[85,86,87], context=[70,71,105,106,107,108,109], rest=[],
   note='Early hand moves to the upper arm before traversal. Transition to near-wrist evidence has no sharp visually defensible single-frame boundary; mark main role without a guessed family.'),
 'WORD1133_08': dict(families={3:[71,73,85,86],4:[90,100,101]}, main_unknown=[88], context=[104,105,106], rest=[],
   note='Hands remain visible during continuous arm traversal and return. Endpoint/main versus return is distinguished by the joint lowering/release, not exact wrist coordinates.'),
 'WORD1567_02': dict(families={5:[44,46,47,98,99]}, main_unknown=[], context=[39,40,104,105,106], rest=[],
   note='Two thumb/closed-hand configurations enter the alternating movement. Last configuration remains as both arms lower; this does not add a repetition-count label.'),
 'WORD1567_04': dict(families={5:[65,66,69,115,116,118]}, main_unknown=[], context=[57,58,59,123,124,125,126,127], rest=[],
   note='Early upward preparation and late joint lowering are separable from clear crossing/alternating configurations. Unclear intermediate boundary points remain unlabelled.'),
 'WORD1567_08': dict(families={5:[63,64,66,109,110,112,113]}, main_unknown=[], context=[118,119,120], rest=[],
   note='Crossing of the two hands is retained as main evidence. A similar final configuration precedes joint lowering. No exact required number of cycles inferred.'),
 'WORD2523_02': dict(families={6:[46,48],7:[57,59,60,80]}, main_unknown=[], context=[43,44,84,85], rest=[],
   note='Two hands approximate, then visibly separate. Skin contact is not asserted. Final lowering is context although the outward configuration persists.'),
 'WORD2523_04': dict(families={6:[72,74,84,86],7:[89,91]}, main_unknown=[87,88], context=[71,111,112,113,114,115], rest=[],
   note='Converged presentation changes into outward configuration; exact boundary is uncertain at87/88. Lowering after endpoint is context.'),
 'WORD2523_08': dict(families={6:[41,43,55],7:[57,59,74]}, main_unknown=[], context=[35,36,37,38,39,77,78,79,80], rest=[30,31,32,33],
   note='Full native frames show genuine initial rest, then preparation and convergence. Missing old labels do not imply missing images. Return is visible after the final outward configuration.'),
}

def main():
    manifest=read(U/'REVIEW_INPUTS.json');old={r['clip_id']:r for r in read(E/'CONTEXT_TARGETS.json')['rows']};rows=[]
    for r in manifest['rows']:
        key=f'{r["word_id"]}_{r["person"]:02}';decision=REVIEWS[key]
        lookup={p['frame']:p['action_family'] for p in old[r['clip_id']]['points'] if p['mask']}
        seen={f:p for p in r['panels'] for f in p['frames']};proposed={}
        for family,frames in decision['families'].items():
            for f in frames:proposed[f]=(family,1)
        for f in decision['main_unknown']:assert f not in proposed;proposed[f]=(-1,1)
        for f in decision['context']:assert f not in proposed;proposed[f]=(-1,0)
        for f in decision['rest']:assert f not in proposed;proposed[f]=(8,2)
        points=[]
        for f,(family,role) in sorted(proposed.items()):
            assert f in seen and f not in lookup,(key,f,'already labelled or not visually displayed')
            panel=seen[f];assert sha(panel['image'])==panel['sha256']
            points.append(dict(frame=f,timestamp=panel['times'][panel['frames'].index(f)],action_family=family,role=role,
                action_mask=family>=0,role_mask=True,evidence_image=panel['image'],evidence_sha256=panel['sha256'],
                provenance='same_Codex_direct_native_RGB_chronological_review',language_correctness=None,
                exact_transition_boundary_claimed=False))
        rows.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],source=r['source'],source_sha256=r['source_sha256'],
            points=points,unlabelled_reviewed_frames=sorted(set(seen)-set(lookup)-set(proposed)),note=decision['note']))
    correction=dict(clip_id='NIA_SL_WORD0739_REAL08_F',frame=62,previous_action_family=0,proposed_action_family=1,role=1,
        status='REQUIRES_SOURCE_PIXEL_AND_DEFINITION_CHECK_BEFORE_TRAINING',
        reason='Full-resolution f62 shows fingers flexed and hand descending toward chin, unlike family0 open-hand face presentation. f63-67 continue closing/relocation; existing family1 includes that transition.',
        full_image=str(D/'review_frames/NIA_SL_WORD0739_REAL08_F_f0062.jpg'),
        full_image_sha256=sha(D/'review_frames/NIA_SL_WORD0739_REAL08_F_f0062.jpg'),
        old_records_and_scores_unchanged=True)
    save(U/'REVIEW_DECISIONS.json',dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),rows=rows,correction_candidates=[correction],
         panels_reviewed=40,clips_reviewed=12,reviewer='Same Codex visual review; not independent KSL teacher annotation.',
         new_points=sum(len(r['points']) for r in rows),new_known_family_points=sum(p['action_mask'] for r in rows for p in r['points']),
         training_authorized_by_this_file=False,training_gate='Verify original decoding/frame/PTS, label non-overlap, source provenance and correction before building new training overlay.',
         no_precision_contact_or_rotation_truth=True,whole_goal_complete=False))
    print('Visual decisions',sum(len(r['points']) for r in rows),'new points;1 existing-label correction candidate;not yet training input')

if __name__=='__main__':main()
