from form_candidate import *
from form_v3 import GuardFormLesson
import importlib.util,shutil
def main():
 import check_form_v2 as checks
 checks.main('guard_checks',GuardFormLesson)
 import check_confirmation as confirm
 # Same frozen public cases evaluated with the corrected inspection assembly.
 # The new camera views have now been inspected; no fresh-holdout claim.
 out=D/'guard_confirmation';(out/'confirmation').mkdir(parents=True,exist_ok=True)
 shutil.copyfile(D/'confirmation/INPUTS.json',out/'confirmation/INPUTS.json')
 confirm.D=out;confirm.FinalFormLesson=GuardFormLesson;confirm.main()
if __name__=='__main__':main()
