from apple_io import *
def main():
 q=read(U/'web/CONSUMER_CHECK.json');assert len(q['rows'])==7 and all(r['supported']and r['rank']['word']=='WORD0099'for r in q['rows'])
 assert all(r['same_evidence']and r['top1_before']['word']==r['top1_after']['word']for r in q['regression'])
 assert read(U/'corrected_checks/CONTROLS.json')['passed']==70
 assert read(U/'holdout/RESULT.json')['passed']==14
 files=list(U.glob('*.py'))+list((U/'runs').glob('*.pt'))+[U/'web/REFERENCE.json',U/'web/reference_features.npz',V/'web/ADMISSION.json',O/'basic_demo_feedback_20261115/apple_classifier.pt']
 write(U/'web/ADMISSION.json',dict(admitted=True,target='WORD0099',scope='Bounded visible source-validated lesson, not all KSL/physical contact/hidden fingers',pins={str(p):sha(p)for p in files},evidence=dict(development_people=[11,12,13,14,15],candidate_held_people=[16,17],controlled_development=70,held_checks=14),limitations=['Actual learner errors and physical webcam not evaluated','Model-selection-independent people16/17, not claimed never used elsewhere in project','Original100 weights unchanged, probabilities renormalized with apple output'],actor='Codex root source review/implementation/verification, no new independent expert review'))
 print('APPLE_WEB_ADMITTED')
if __name__=='__main__':main()
