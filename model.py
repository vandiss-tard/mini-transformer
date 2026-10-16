import torch
import torch.nn as nn

torch.manual_seed(0)


class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                    # (B, T, head_size)
        q = self.query(x)                  # (B, T, head_size)
        scores = q @ k.transpose(-2, -1)   # (B, T, T)

        # ① 遮罩：把"未来"涂成 -inf
        mask = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # ② 归一化：每行加起来 = 1
        wei = torch.softmax(scores, dim=-1)   # (B, T, T)

        # ③ 用权重去取 value，加权求和
        v = self.value(x)                     # (B, T, head_size)
        return wei @ v                        # (B, T, head_size)


x = torch.randn(2, 8, 32)
head = Head(n_embd=32, head_size=16)
out = head(x)
print('输入 shape:', x.shape)
print('输出 shape:', out.shape)

# 因果性测试：把位置 3 及之后的内容全部改成 999
x2 = x.clone()
x2[:, 3:, :] = 999.0

out2 = head(x2)
print()
print('前 3 个位置的输出没受影响:', torch.allclose(out[:, :3], out2[:, :3]))
print('后 5 个位置的输出变化了  :', not torch.allclose(out[:, 3:], out2[:, 3:]))