"""Record the video verdicts for LOSS_LIST.json (judged by viewing frames/). No computation of verdicts."""
import json, pathlib
H = pathlib.Path(__file__).resolve().parent
doc = json.loads((H / 'LOSS_LIST.json').read_text(encoding='utf-8'))
NORMAL = 'NORMAL_HAND_LOST - the baseline held this real hand on its correct side'
WRONG = 'WRONG_SIDE_BINDING_REMOVED - baseline had bound the real RIGHT hand to L; candidate correctly drops it but does NOT restore it to R'
V = {}
for c, fr in {'NIKL_GAM_01': [16, 17, 18, 20], 'NIKL_JOESONGHADA_01': [12, 13, 14], 'NIKL_MAEKJU_01': [15, 17, 20],
              'NIKL_MEOKDA_01': [11, 12], 'NIKL_MOREUDA_01': [13, 14, 15, 16, 18], 'NIKL_MUL_01': [13, 14, 15]}.items():
    for f in fr: V[(c, f, 'R')] = NORMAL
V[('NIKL_GONGBU_01', 22, 'L')] = NORMAL + ' (both hands at the frame bottom)'
V[('NIKL_GONGBU_01', 22, 'R')] = NORMAL + ' (both hands at the frame bottom)'
for c, f in [('NIKL_GAM_01', 21), ('NIKL_GAM_01', 22), ('NIKL_MAEKJU_01', 16), ('NIKL_MEOKDA_01', 14),
             ('NIKL_MOREUDA_01', 19), ('NIKL_MUL_01', 17)]:
    V[(c, f, 'L')] = WRONG
V[('NIKL_MUL_01', 18, 'L')] = 'DUPLICATE_REMOVED - both detections on the right hand; baseline L was the second copy'
V[('NIKL_MOREUDA_01', 17, 'R')] = ('OTHER_OBSERVATION_CHANGE - baseline R lies on the real right hand and correct side but covers only the '
                                   'fingers (badly localised); zoom frames/NIKL_MOREUDA_01_zoom16-18.png')
tally, disp = {}, {'frame_displayed': 0, 'run_level_only': 0}
for r in doc['rows']:
    k = (r['clip'], r['frame'], r['side'])
    if r['existing'] is None:
        r['verdict'] = V[k]; r['verdict_scope'] = 'frame displayed in this unit (contact sheet; MOREUDA f16-18 zoomed)'
    else:
        r['verdict'] = r['existing']['category']; r['verdict_scope'] = r['existing']['source'] + (
            ' - frame displayed' if r['existing']['frame_displayed'] else ' - run-level verdict, frame NOT displayed')
        disp['frame_displayed' if r['existing']['frame_displayed'] else 'run_level_only'] += 1
    key = r['verdict'].split(' ')[0]
    tally[key] = tally.get(key, 0) + 1
assert len(V) == doc['to_review'] == 30
doc['tally_all_55'] = tally
doc['existing_24_plus_L15_scope'] = disp
doc['cause_of_normal_losses'] = ('all 22 are clip-start frames where BOTH pose wrists are unusable (hand low at the frame bottom) while the '
    'shoulders are usable and no track exists yet. The candidate refuses (refused_no_pose_reference) - behaviour inherited from the original '
    'temporal candidate, NOT from the arm rule (which needs both pose wrists). The operating path falls back to the MediaPipe handedness label, '
    'which was right in 22 of the 23 reviewed no-pose-wrist slots here and wrong once (MAEKJU f16, right hand labelled L).')
(H / 'LOSS_LIST.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
print(tally, disp)
