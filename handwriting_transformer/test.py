import torch
from torch import nn
import torch.nn.functional as F
import math

#定义Embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size,d_model):
        super().__init__(vocab_size,d_model,padding_idx=1)

#创建(max_len,d_model)的位置编码矩阵self.encoding
class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,max_len,device):
        super().__init__()
        self.encoding = torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad = False

        #arrange生成0开始到max_len-1的整数
        pos = torch.arange(0,max_len,device=device)
            
        #unsqueeze(dim = 1)延展成2维张量，dim=0就是一维张量
        pos = pos.float().unsqueeze(dim = 1)
        _2i = torch.arange(0,d_model,step=2,device=device).float()

        #self.encoding[:, 0::2]：意思是“取所有的行，列从 0 开始，步长为 2”。
        # 把值填给 self.encoding 矩阵的所有偶数维度（0, 2, 4, 6...）
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2]  = torch.cos(pos / (10000 ** (_2i/d_model)))

    def forward(self,x):
            #默认输入x是一个2 维张量，形状是 (批次大小, 句子长度)
        batch_size,seq_len = x.size()
            #:seq_len表示从第0到第seq_len行
        return self.encoding[:seq_len,:]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model ,max_len, drop_prob ,device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size,d_model)
        self.pos_emb = PositionalEmbedding(d_model,max_len,device)

        #drop_prob随机丢弃率，用于正则化
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self,x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb+pos_emb)

# ==========================================
# 🚀 测试代码
# ==========================================
if __name__ == "__main__":
    # 固定随机种子，让每次运行结果一致
    torch.manual_seed(42)

    # 1. 设置较小的测试参数
    vocab_size = 10
    d_model = 8
    max_len = 20
    drop_prob = 0.5  # 设大一点，方便观察 Dropout 的效果
    device = "cpu"

    print("========== 1. 初始化 TransformerEmbedding ==========")
    emb_model = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, device)

    # 2. 构造输入数据 (Batch=2, 句子长度 L=4)
    # 故意放入 ID 为 1 的词（PAD）
    input_ids = torch.tensor([
        [5, 1, 9, 2],  # 第一句话
        [1, 1, 4, 7]   # 第二句话，前两个是 PAD
    ])
    
    print("输入原始 Token IDs 形状:", input_ids.shape)
    print(input_ids)
    print("-" * 50)

    # 设置 PyTorch 打印格式，保留3位小数，不要科学计数法
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=150)

    # ==========================================
    # 🔍 测试步骤 A: 拆解看 TokenEmbedding
    # ==========================================
    print("\n========== 2. 测试 TokenEmbedding ==========")
    tok_out = emb_model.tok_emb(input_ids)
    print("Token 向量形状:", tok_out.shape) # 应该是 [2, 4, 8]
    print("第一句话的 Token 向量:\n", tok_out[0])
    print("👉 注意看第 2 行（对应输入的 ID=1），是不是全 0？")
    print("-" * 50)

    # ==========================================
    # 🔍 测试步骤 B: 拆解看 PositionalEmbedding
    # ==========================================
    print("\n========== 3. 测试 PositionalEmbedding ==========")
    pos_out = emb_model.pos_emb(input_ids)
    print("位置编码形状:", pos_out.shape) # 应该是 [4, 8] (因为 seq_len=4)
    print("截取出的位置编码:\n", pos_out)
    print("👉 注意看：它是按行广播的，不需要 batch 维度，直接和 Token 向量相加。")
    print("-" * 50)

    # ==========================================
    # 🔍 测试步骤 C: 综合前向传播 + Dropout
    # ==========================================
    print("\n========== 4. 测试完整前向传播 (训练模式 train) ==========")
    emb_model.train()  # 开启训练模式，激活 Dropout
    train_out = emb_model(input_ids)
    print("最终输出形状:", train_out.shape) # [2, 4, 8]
    print("第一句话的输出:\n", train_out[0])
    print("👉 观察重点：里面有大量 0.000，这就是 drop_prob=0.5 随机丢弃的效果！")
    
    print("\n========== 5. 测试完整前向传播 (推理模式 eval) ==========")
    emb_model.eval()   # 开启推理模式，关闭 Dropout
    eval_out = emb_model(input_ids)
    print("第一句话的输出:\n", eval_out[0])
    print("👉 观察重点：0.000 变少了（只有 PAD 的位置是 0），其他都是真实相加的值。")