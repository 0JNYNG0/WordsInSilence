from cycle_candidate import *
import importlib.util
def main():
 s=importlib.util.spec_from_file_location('cycle_public_regression',J/'check_regression.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.J=M;m.main(CycleLesson,'regression')
if __name__=='__main__':main()
