"""Read-only public research replay. No camera, training, classifier inference or writes."""
from pathlib import Path
import json,hashlib

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class PublicResearchReplay:
    def __init__(self, manifest):
        self.manifest=read(manifest)
        if self.manifest['schema']!='public_recognition_feedback_v1':
            raise ValueError('Unsupported research bundle')
        for path,digest in self.manifest['pins'].items():
            if sha(path)!=digest:
                raise ValueError('Frozen source changed: '+path)
        self.entries={r['query_id']:r for r in self.manifest['rows']}
        if len(self.entries)!=300 or self.manifest['allow_personal_input']:
            raise ValueError('Public DEV-only contract mismatch')

    def get(self, query_id, model='C'):
        if model not in ('C','T3F','E3'):
            raise ValueError('Choose a preserved research model')
        if query_id not in self.entries:
            raise ValueError('Only frozen public clips are accepted; no personal recordings')
        row=self.entries[query_id]
        if sha(row['query_path'])!=self.manifest['pins'][row['query_path']]:
            raise ValueError('Frozen query changed')
        query=read(row['query_path'])
        references=[]
        for ref in query['references']:
            path=ref['receipt']
            if sha(path)!=self.manifest['receipt_pins'][path]:
                raise ValueError('Frozen feedback changed')
            receipt=read(path)
            if receipt['query_id']!=query_id or receipt['reference_id']!=ref['reference']:
                raise ValueError('Clip identity mismatch')
            references.append(receipt)
        return dict(schema='public_recognition_feedback_response_v1',
            query_id=query_id,recorded_person_id=row['recorded_person_id'],
            practice_target=dict(word_code=row['word_code'],label=row['label']),
            model=model,prediction=row['predictions'][model],
            classification_scope='Saved100-class result; not proof that execution is correct',
            feedback_target='The practice target; never silently replaced by predicted word',
            references=references,reference_combination='separate_people_no_pose_average',
            whole_gesture_correct=None,validated_correction=None,skill_score=None,
            observation_policy='Missing/uncertain/reference shortage is not user movement error',
            inference_performed=False,training_performed=False,personal_data=False,
            live_webcam_supported=False)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query_id');parser.add_argument('--model',choices=['C','T3F','E3'],default='C')
    args=parser.parse_args()
    result=PublicResearchReplay(Path(__file__).with_name('MANIFEST.json')).get(args.query_id,args.model)
    print(json.dumps(result,ensure_ascii=False))
