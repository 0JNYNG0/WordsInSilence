from common import *
def main():
    x=np.arange(7*280,dtype=np.float32).reshape(7,280)
    ts=np.array([0,31000,70000,102000,146000,179000,220000],np.int64)
    x[:,279]=np.r_[0,np.diff(ts)/1e6]
    y,tt=reverse_interval(x,ts,1,5)
    assert np.array_equal(tt,np.array([0,31000,64000,108000,140000,179000,220000]))
    assert np.array_equal(y[1:6,:279],x[1:6,:279][::-1])
    assert np.array_equal(y[[0,6]],x[[0,6]])
    assert np.allclose(y[:,279],np.r_[0,np.diff(tt)/1e6])
    z,zt=reverse_interval(y,tt,1,5);assert np.array_equal(z,x)and np.array_equal(zt,ts)
    write(HERE/'TRANSFORM_TEST.json',dict(at=stamp(),status='PASS',checks=['nonuniform timestamp reversal','outside frame and time preserved','dt matches actual reflected times','double reversal identity']))
    print('Transform contract PASS')
if __name__=='__main__':main()
