import torch
from torch import nn 
from dataclasses import dataclass
import math

@dataclass
class Config(object):
    vocab_size: int = -1
    embedding_size: int = 2048
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    dropout_prob: int = 0.5
    norm_eps = 1e-5
    
class AttentionLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.m_head = config.num_heads
        assert self.dim % self.m_head == 0
        
        self.W_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def _split_m_head(self, W):
        batch_size, seq_len, _ = W.shape
        return W.view(batch_size, seq_len, self.m_head, -1).permute(0, 2, 1, 3)
        
    def forward(self, X, attention_mask = None, head_mask = None):
        q, k, v = self.W_q(X), self.W_k(X), self.W_v(X)
        q, k, v = self._split_m_head(q), self._split_m_head(k), self._split_m_head(v)
        
        weight = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.dim / self.m_head)
        if attention_mask is not None:
            weight += attention_mask
        weight = self.dropout(torch.softmax(weight, dim=-1))
        
        if head_mask is not None:
            weight *= head_mask
        
        v = torch.matmul(weight, v)
        batch_size, num_heads, seq_len, head_dims = v.shape
        v = v.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dims)
        
        return v


class LayerNorm(nn.Module):
    def __init__(self, normalized_size, eps):
        super().__init__()
        self.normalized_size = normalized_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_size))
        self.beta = nn.Parameter(torch.zeros(normalized_size))
        
    def _norm(self, X):
        return X / (X.mean(dim=-1, keepdim=True) + self.eps)
        
    def forward(self, X):
        return self.gamma * self._norm(X.float()).type_as(X) + self.beta


class RMSNorm(nn.Module):
    def __init__(self, normalized_size, eps):
        super().__init__()
        self.normalized_size = normalized_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_size))

    def _norm(self, X):
        return X * torch.rsqrt(X.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, X):
        # 考虑到X可能被量化过？
        return self.gamma * self._norm(X.float()).type_as(X)
    
class DeepNorm(nn.Module):
    pass


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = AttentionLayer(config)
        self.ffw = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        self.norm1 = LayerNorm(config.hidden_size, config.norm_eps)
        self.norm2 = LayerNorm(config.hidden_size, config.norm_eps)
        self.dropout = torch.nn.Dropout(config.dropout_prob)
        
    def forward(self, X):
        output = X + self.norm1(self.dropout(self.attention(X)))
        output = output + self.norm2(self.dropout(self.ffw(output)))
        return output
    
def RoPE(q: torch.Tensor, k: torch.Tensor, freq_cis: torch.Tensor):
    '''
    同时对q,k做旋转编码，减少一次freq_cis的变形
    '''
    q = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    print(f'q = {q}, k = {k}')
    freq_cis = reshape_for_broadcast(freq_cis, q)
    q = torch.view_as_real(q * freq_cis).flatten(3)
    k = torch.view_as_real(k * freq_cis).flatten(3)
    print(f'freq = {freq_cis}, q = {q}, k = {k}')
    return q, k

def reshape_for_broadcast(freq_cis: torch.Tensor, x: torch.Tensor):
    '''
    将freq_cis从(batch, dim)转换为(batch, 1, 1, dim),方便在seqlen维度与head维度的广播
    '''
  
    dim = x.ndim
    assert dim > 1
    assert freq_cis.shape == (x.shape[1], x.shape[-1])
    # (batch, dim)->(1,head, 1, dim)
    shape = [d if i == 0 or i == dim - 1 else 1 for i, d in enumerate(x.shape)]
    return freq_cis.view(*shape)

def precompute_freq_cis(dim, end, theta = 10000.0):
    '''
    预计算旋转矩阵
    '''
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
    

if __name__ == '__main__':
    
    # config = Config(
    #     vocab_size = 100,
    #     embedding_size = 256,
    #     hidden_size = 512,
    #     num_layers = 6,
    #     num_heads = 8,
    # )

    # layer = AttentionLayer(config)
    # RN = RMSNorm(512, 1e-5)
    # trans = TransformerBlock(config)
    # X = torch.randn(2, 16, 512)
    # print(layer(X).shape)
    # print(RN(X).shape)
    # print(trans(X).shape)
    
    freq = precompute_freq_cis(4, 6, theta=2)
    q, k = torch.ones(4, 2, 3, 6), torch.randn(4, 2, 3, 6)
    q, k = RoPE(q, k, freq)