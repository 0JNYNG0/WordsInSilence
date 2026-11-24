from candidate import *
import importlib.util
def main():
 s=importlib.util.spec_from_file_location('prior_view_regression',J/'check_regression.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.J=L;m.main(ShapeLesson,'shape_regression')
if __name__=='__main__':main()
