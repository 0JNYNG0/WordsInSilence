from endpoint_v2 import *
import importlib.util
def main():
 s=importlib.util.spec_from_file_location('fresh_public_camera',D/'prepare_confirmation.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.D=L;m.main(4,'confirmation','endpoint_v2.py')
 p=L/'confirmation/FROZEN.json';q=read(p);q['purpose']='New camera views of known FIT person4, never training or candidate selection for this repair; not unseen-person accuracy.';write(p,q)
if __name__=='__main__':main()
