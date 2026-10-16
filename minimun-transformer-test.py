import torch
import torch.nn as nn
import torch.nn.functional as F

#输入 x ➡️ 带Mask的多头自注意力 ➡️ 残差+归一化 ➡️ FFN ➡️ 残差+归一化 ➡️ 输出

class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model=4, nhead=2, dim_feedforward=16):
        super().__init__()
        # 1. 多头自注意力 (PyTorch 官方封装)
        # batch_first=True 意味着输入形状是 (B, L, D)

        #embed_dim是每个词的维度，self.self_attn 包含了权重参数和计算逻辑，输入数据并调用它时，它才会输出 Tensor（张量）格式的数据
        #nn.MultiheadAttention可以理解成一个自带参数的函数
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # 2. 前馈神经网络 FFN (线性 -> 激活 -> 线性)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # 3. LayerNorm (用于残差连接后)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x 形状: (B, L, D) -> 这里对应 (1, 3, 4)
        B, L, D = x.shape
        
        # --- 步骤 1: 生成因果 Mask (左下角为0，右上角为 -inf) ---
        # 形状 (L, L) -> (3, 3)。torch.triu(..., diagonal=1)，triu 是取上三角矩阵（Upper Triangle）。diagonal=1 表示从主对角线的右上方开始保留，其余部分（包括主对角线）全部变成 0
        mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1)
        
        # --- 步骤 2: 多头自注意力 + 残差 + LayerNorm ---
        # PyTorch 的 MultiheadAttention 会自动帮你做 Q, K, V 的投影和分头
        # 输入 x 既是 Q, 也是 K, 也是 V
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        # attn_output 形状: (1, 3, 4)
        
        x = self.norm1(x + attn_output) # 残差连接并归一化
        
        # --- 步骤 3: FFN + 残差 + LayerNorm ---
        ffn_output = self.ffn(x)
        # ffn_output 形状: (1, 3, 4)
        
        x = self.norm2(x + ffn_output) # 残差连接并归一化
        
        return x

# ================= 测试运行 =================
# 设定: Batch=1, Seq_Len=3, d_model=4
B, L, D = 1, 3, 4
x = torch.randn(B, L, D) # 模拟输入词嵌入+位置编码

print(f"输入形状: {x.shape}")

# 实例化 Transformer Block
block = SimpleTransformerBlock(d_model=D, nhead=2, dim_feedforward=16)
output = block(x)

print(f"输出形状: {output.shape}")