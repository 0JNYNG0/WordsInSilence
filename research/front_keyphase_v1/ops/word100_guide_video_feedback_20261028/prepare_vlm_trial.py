"""Prepare a local-only, word-blind test manifest. Does not acquire or load a model."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
PROMPT='''These are time-stamped frames from a front-view recording of a person. Describe only actions visible in these frames. Do not identify the sign or infer an intended word. Use the person's anatomical left and right, not screen sides. If a side, finger shape, palm facing, contact, count, or intervening movement is not visible, say unknown. Do not assume continuity across a hidden interval. Return JSON with: events (each has start_seconds, end_seconds, hand, visible_hand_shape, body_or_other_hand_relation, movement, supporting_frame_timestamps, uncertain_fields), and limitations. Statements about touching require evidence beyond 2D overlap; otherwise say projected overlap. Do not claim linguistic correctness.'''
def main():
    plan=read(U/'MODEL_TRIAL_PLAN.json');idx=read(O/'word100_shared_feature_bundle_20261027/visual_inputs/INDEX.json')['rows']
    by={r['alias']:r for r in idx};rows=[]
    for alias in plan['frozen_public_aliases']:
        r=by[alias];frames=[]
        for f in r['frames']:
            p=Path(f['file']);assert p.exists()
            frames.append({'timestamp_seconds':f['pts_us']/1e6,'path':str(p),'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        # Evaluation-only metadata is held outside the actual observation prompt.
        rows.append({'case_id':f'clip_{len(rows)+1:02d}','frames':frames,'prompt':PROMPT,
            'evaluation_only':{'alias':alias,'word_id':r['word_id'],'person':r['recorded_person_id']}})
    out={'status':'PREPARED_NOT_INFERRED','model':plan['candidate'],'rows':rows,
        'prompt_policy':'Do not serialize evaluation_only into model input. Guide-dependent conditions are separate later tests.',
        'network_policy':'All model input is local. No remote inference or image upload.',
        'execution_gate':'Specific external download confirmation required. Download and inference have not happened.'}
    (U/'VLM_INPUTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print({'clips':len(rows),'rgb_inputs':sum(len(r['frames']) for r in rows),'inference_runs':0})
if __name__=='__main__':main()
