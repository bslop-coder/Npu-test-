import torch, torch.nn as nn, torch.nn.functional as F
import ai_edge_torch

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class Block(nn.Module):
    def __init__(self, d=2048, h=16, m=8192):
        super().__init__()
        self.h = h
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.qkv = nn.Linear(d, d * 3)
        self.o = nn.Linear(d, d)
        self.up = nn.Linear(d, m)
        self.down = nn.Linear(m, d)
        self.mod = nn.Linear(d, d * 2)
    def forward(self, x, t):
        s, b = self.mod(F.silu(t)).chunk(2, -1)
        y = self.n1(x) * (1 + s) + b
        B, N, D = y.shape
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).unbind(2)
        q, k, v = [z.transpose(1, 2) for z in (q, k, v)]
        a = F.softmax(q @ k.transpose(-1, -2) / (D // self.h) ** 0.5, -1) @ v
        x = x + self.o(a.transpose(1, 2).reshape(B, N, D))
        return x + self.down(F.gelu(self.up(self.n2(x))))

m = Block().eval()
ex = (torch.randn(1, 1024, 2048), torch.randn(1, 1, 2048))
ai_edge_torch.convert(m, ex).export("block.tflite")
