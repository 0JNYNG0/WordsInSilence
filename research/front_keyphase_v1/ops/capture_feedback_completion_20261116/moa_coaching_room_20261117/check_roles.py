from probe_endpoint_roles import *
import importlib.util
P=G/'seated_stop_short_cycle_20261117'
def main():
 spec=importlib.util.spec_from_file_location('role_regression',P/'check_short.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.FrontalLesson=ShortArrivalLesson;m.ShortArrivalLesson=RoleLesson;m.D=D
 original=m.write;m.write=lambda p,x:original(D/'checks/ROLE_CHECK.json' if Path(p).name=='SHORT_CHECK.json' else p,x)
 m.main()
 spec=importlib.util.spec_from_file_location('role_partials',G/'rest_stability_endpoint_20261117/check_reviewed_partials.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.EndpointLesson=RoleLesson;m.D=D;m.main()
if __name__=='__main__':main()
