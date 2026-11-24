from train_shape import *
def main():
 out=V/'web';assert not(out/'ADMISSION.json').exists()
 full=read(V/'sparse_full_regression/RESULT.json');diff=read(V/'sparse_distinctions/RESULT.json');held=read(V/'holdout/RESULT.json');web=read(out/'CONSUMER_CHECK.json');shape=read(V/'view_conditioned/COMPARISON.json');sparse=read(V/'sparse_checks/RESULT.json')
 assert full['passed']==full['total'] and diff['failed']==0 and held['passed']==held['total'] and shape['component_gate']
 assert all(r['supported']and r['rank']['word']=='WORD1290'for r in web['rows'])
 assert all(not r['supported']for r in sparse['rows']if r['kind']in['seven_missing','first_only','constant'])
 files=[V/n for n in['visible_lesson.py','observed_gap.py','target_region.py','plane_model.py','view_conditioned_shape.py','shape_candidate.py','evaluate_parts.py','worker.py','presentation.py','web_adapter.py']]
 files += [p for d in['shape_source_repair','plane']for p in(V/d).glob('*.pt')]
 files += [V/'target_region/FROZEN.json',V/'web/REFERENCE.json',V/'web/reference_features.npz']
 write(out/'ADMISSION.json',dict(admitted=True,target='WORD1290',required_ids=['right_shape','left_shape','right_orientation','left_orientation','relative_target','repeat_order'],
  scope='Gratitude lesson: projected visible forms, target relation and repeated movement; not all-word/3D-contact certification.',
  evidence=dict(normal_public_people=[10,14,15,16,17,18],known_regression=35,synthetic_distinction_checks=40,held_person=10,
   one_frame_recovery='5 of 6 diagnostic clips supported; person18 remains unknown, never declared wrong',shape_development=shape['development']),
  limitations=['No new user webcam test yet','Synthetic keypoint controls are not actual learner-error accuracy','Palm-front/back and physical contact depth cannot be certified from this head','Shared implementation, only this lesson qualified'],
  pins={str(p):sha(p)for p in files},actor='Codex root implementation and own source/code verification; no new independent expert review',
  old_C_T3F_changed=False,user_training=False))
 print('ADMITTED_LIMITED_VISIBLE_GRATITUDE')
if __name__=='__main__':main()
