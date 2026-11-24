from final_temporal import *
import importlib.util
def main():
 s=importlib.util.spec_from_file_location('final_view_regression',J/'check_regression.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.J=L;m.main(TemporalVisibilityLesson,'qualified_regression')
if __name__=='__main__':main()
