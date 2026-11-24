from form_candidate import *
from form_v5 import QualifiedFormLesson
import shutil
def main():
 import check_form_v2 as checks
 checks.main('qualified_checks',QualifiedFormLesson)
 import check_confirmation as confirm
 out=D/'qualified_confirmation';(out/'confirmation').mkdir(parents=True,exist_ok=True)
 shutil.copyfile(D/'confirmation/INPUTS.json',out/'confirmation/INPUTS.json')
 confirm.D=out;confirm.FinalFormLesson=QualifiedFormLesson;confirm.main()
if __name__=='__main__':main()
