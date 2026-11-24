from form_candidate import *
from form_v4 import JointInspectionLesson
import shutil
def main():
 import check_form_v2 as checks
 checks.main('joint_inspection_checks',JointInspectionLesson)
 import check_confirmation as confirm
 out=D/'joint_inspection_confirmation';(out/'confirmation').mkdir(parents=True,exist_ok=True)
 shutil.copyfile(D/'confirmation/INPUTS.json',out/'confirmation/INPUTS.json')
 confirm.D=out;confirm.FinalFormLesson=JointInspectionLesson;confirm.main()
if __name__=='__main__':main()
