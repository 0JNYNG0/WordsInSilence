from cycle_common import *
import importlib.util
def main():
 s=importlib.util.spec_from_file_location('cycle_new_camera',D/'prepare_confirmation.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.D=M;m.main(8,'confirmation','cycle_candidate.py')
 p=M/'confirmation/FROZEN.json';q=read(p);q['purpose']='New unselected camera views of knownFITperson8, not independent-person accuracy.';write(p,q)
if __name__=='__main__':main()
