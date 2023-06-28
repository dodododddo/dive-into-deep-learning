import time
import numpy as np
import torch


class Timer(object):
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()
    
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times) / len(self.times)
    
    def sum(self):
        return sum(self.times)
    
    def cumsum(self):
        return np.array(self.times).cumsum().tolist
    
if __name__ == '__main__':
    n = 1000000
    a,b = torch.ones(n),torch.randn(n)
    c = torch.zeros(n)
    timer = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    time1 = timer.stop()
    
    timer.start()
    d = a + b
    time2 = timer.stop()

    print(f'{time1:.5f} sec')
    print(f'{time2:.5f} sec')
    print(f'加速 {time1 / time2:.5f} 倍')

    