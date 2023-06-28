from d2l import torch as d2l
import math
import numpy as np

def normal(x,mu,sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)

if __name__ == "__main__":
    x = np.arange(-7,7,0.01)
    params = [(0,1),(0,2),(0,3)]
    d2l.plot(x,[normal(x,mu,sigma) for mu, sigma in params],"x","p(x)",figsize=(4.5,2.5),legend=[f'mean{mu},std{sigma}'for mu, sigma in params])
    d2l.plt.show()