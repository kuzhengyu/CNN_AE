from time import *
import numpy as np

# 看看读取速度

begin_time = time()

a = np.load('../data/image_data_1000.npy')

end_time = time()
run_time = end_time - begin_time
print('该循环程序运行时间：', run_time, 's')