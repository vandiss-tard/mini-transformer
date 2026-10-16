"""最小 Transformer（字符级 GPT，decoder-only）。"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """单头因果自注意力"""

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # 下三角矩阵，用来遮掉“未来”的位置
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)                                          # (B, T, head_size)
        q = self.query(x)                                        # (B, T, head_size)
        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)

        # 因果遮罩：把“未来”的分数涂成 -inf
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # 归一化成权重
        wei = F.softmax(scores, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)                                        # (B, T, head_size)
        return wei @ v                                           # (B, T, head_size)

class MultiHeadAttention(nn.Module):
    """多头注意力：把 n_head 个 Head 拼起来"""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)      # (B, T, n_embd)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    """每个位置自己过一遍 MLP"""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer 基本块：注意力 + 前馈，都带残差"""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4,
                 block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok = self.token_embedding(idx)                                  # (B,T,n_embd)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok + pos

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                         # (B,T,vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- 测试 1：形状 ----
    x = torch.randn(2, 8, 32)
    mha = MultiHeadAttention(n_embd=32, n_head=4, block_size=16)
    print("注意力 :", tuple(x.shape), "->", tuple(mha(x).shape))

    model = MiniGPT(vocab_size=65, n_embd=32, n_head=4, n_layer=2, block_size=16)
    logits, loss = model(torch.randint(0, 65, (2, 8)), torch.randint(0, 65, (2, 8)))
    print("模型输出:", tuple(logits.shape))
    print("初始 loss:", round(loss.item(), 3), "（应接近 ln(65)=4.174）")

    # ---- 测试 2：因果性 ----
    mha.eval()
    x2 = torch.randn(1, 8, 32)
    with torch.no_grad():
        y1 = mha(x2)
        x3 = x2.clone()
        x3[:, 3:, :] = 999.0
        y2 = mha(x3)
    print("前 3 个位置没受影响:", torch.allclose(y1[:, :3], y2[:, :3]))
    print("后 5 个位置变化了  :", not torch.allclose(y1[:, 3:], y2[:, 3:]))