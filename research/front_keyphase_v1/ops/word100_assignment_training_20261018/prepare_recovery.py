"""Serialization-only recovery; frozen computations untouched.

Original first error and orphan arrays are retained under first_failure/.
This wrapper adds JSON conversion of NumPy scalar records. It does not
change decisions, arrays, training gates, source roles, or context hashes.
"""
import argparse
import common
import prepare
import numpy as np
def scalar(x):
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def write_numpy(path,obj,new=False):
 path=common.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if new:assert not path.exists(),str(path)
 temp=path.with_name(path.name+'.writing')
 temp.write_text(common.json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=scalar)+'\n',encoding='utf8')
 common.os.replace(temp,path)
if __name__=='__main__':
 record=common.read(common.U/'SERIALIZATION_RECOVERY.json');common.check(record['code_pins']);common.check(record['preserved_pins'])
 common.write=write_numpy;prepare.write=write_numpy
 p=argparse.ArgumentParser();p.add_argument('--role',default='train_candidate');p.add_argument('--limit',type=int,default=1199);a=p.parse_args()
 prepare.main(a.role,a.limit)
