from darknet import *
import concurrent.futures
import time

set_gpu(0) # running on GPU 0
net1 = load_net(b"cfg/yolov3-lp_vehicles.cfg", b"backup/yolov3-lp_vehicles.backup", 0)
meta1 = load_meta(b"data/lp_vehicles.data")

set_gpu(1) # running on GPU 0
net2 = load_net(b"cfg/yolov3-lp_vehicles.cfg", b"backup/yolov3-lp_vehicles.backup", 0)
meta2 = load_meta(b"data/lp_vehicles.data")


def f(x):
    if x[0] == 0: # gpu 0
        return detect_np_lp(net1, meta1, x[1])
    else:
        return detect_np_lp(net2, meta2, x[1])



def func1(): # without threading
    a = cv2.imread("lp_tester/bug1.jpg")
    r1 = f( (0, a) )
    r2 = f( (1, a) )
    #print('out f1')
    #return [r1, r2]
    

def func2(): # with threading
    a = cv2.imread("lp_tester/bug1.jpg")
    nums = [(0, a), (1, a)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        r_m = [val for val in executor.map(f, nums)]
    #print('out f2')
    #return r_m

av_fps = 0.
for _ in range(100):
    t1 = time.time()
    func1()
    t2 = time.time()
    print(f'fps: {1/(t2-t1)}')
    av_fps += (1/(t2-t1))/100.

print(f'Average: {av_fps}')

av_fps = 0.
for _ in range(100):
    t1 = time.time()
    func2()
    t2 = time.time()
    print(f'fps: {1/(t2-t1)}')
    av_fps += (1/(t2-t1))/100.

print(f'Average: {av_fps}')