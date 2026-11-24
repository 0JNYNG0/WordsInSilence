from endpoint_candidate import *
import importlib.util
def main():
 spec=importlib.util.spec_from_file_location('endpoint_regression_checks',P/'check_short.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.FrontalLesson=ShortArrivalLesson;m.ShortArrivalLesson=EndpointLesson;m.D=D
 original=m.write
 m.write=lambda p,x:original(D/'checks/ENDPOINT_CHECK.json'if Path(p).name=='SHORT_CHECK.json'else p,x)
 m.main()
if __name__=='__main__':main()
