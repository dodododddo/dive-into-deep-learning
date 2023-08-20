import torch

def trans_conv(X: torch.tensor, kernel: torch.tensor, stride = 1, padding = 0)->torch.tensor:
    h, w = X.shape
    kh, kw = kernel.shape
    assert(stride <= kh and stride <= kw)
    
    Y = torch.zeros((h + kh + stride - 2, w + kw + stride - 2))
    for i in range(h):
        for j in range(w):
            Y[i * stride: i * stride + kh, j * stride: j * stride + kw] += X[i, j] * kernel
    
    if padding == 0:
        return Y
    else:
        return Y[padding: -padding, padding: -padding]
 
if __name__ == "__main__":
       
    X = torch.tensor([[1, 2],[3, 4]])
    kernel = torch.tensor([[0, 1], [2, 3]])

    res = trans_conv(X, kernel, stride=2, padding=1)
    print(res)
        
    