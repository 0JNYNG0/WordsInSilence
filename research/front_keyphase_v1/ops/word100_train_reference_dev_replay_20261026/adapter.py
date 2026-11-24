"""Frozen public replay only; no live personal input or model execution."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class PublicFeedbackReplay:
    def __init__(self):
        self.manifest=read(U/'MANIFEST.json')
        assert self.manifest['schema']=='train_reference_public_replay_v1'
        self.rows={r['query_id']:r for r in self.manifest['rows']}
        assert len(self.rows)==300 and not self.manifest['allow_personal_input']
    def get(self,query_id,model='C'):
        if model not in ('C','T3F','E3'):raise ValueError('Unknown preserved model')
        if query_id not in self.rows:raise ValueError('Frozen public query only; personal input is not accepted')
        row=self.rows[query_id];p=Path(row['receipt']);assert sha(p)==row['sha256'];receipt=read(p)
        return dict(schema='train_reference_public_response_v1',query_id=query_id,model=model,
            practice_target=receipt['practice_target'],prediction=receipt['predictions'][model],
            references=receipt['references'],reference_aggregation='none; all named TRAIN references retained',
            whole_gesture_correct=None,validated_correction=None,skill_score=None,
            scope='Public DEV research replay; supplied target selects references, never prediction. Same-MediaPipe image corroboration is not anatomical truth.',
            live_webcam_supported=False,training_performed=False,inference_performed=False,personal_data=False)
