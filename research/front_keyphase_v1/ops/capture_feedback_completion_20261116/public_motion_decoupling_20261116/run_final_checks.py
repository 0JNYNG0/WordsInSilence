from common import *
from palm_geometry import PalmMotionLesson
from check import main as compare
from check_regression import load

def main():
 compare(PalmMotionLesson,'palm_checks')
 out=E/'palm_regression';out.mkdir(exist_ok=True)
 m=load('palm_motion_distinctions',H/'check_distinctions.py');m.HERE=out;m.main('distinctions',PalmMotionLesson)
 m=load('palm_motion_patterns',D/'check_confirmation.py');m.D=out;m.FinalFormLesson=PalmMotionLesson;m.ThanksCandidate=QualifiedFormLesson
 write(out/'confirmation/INPUTS.json',dict(rows=[]));m.main()
 write(E/'PROGRESS.json',dict(status='PUBLIC_CHECKS_COMPLETE_REVIEW_BEFORE_ADMISSION',deployed='previous D unchanged',current_candidate='PalmMotionLesson',new_relation='51 body/palm root and MCP geometry, no distal finger joints',last_checks='palm_checks + palm_regression + palm_views',private_data=False))
if __name__=='__main__':main()
