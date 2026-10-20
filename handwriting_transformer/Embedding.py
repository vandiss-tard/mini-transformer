import torch
from torch import nn
import torch.nn.functional as F
import math


# 只有第一次实例化的时候才会触发class中的init，而forward相当于实例化后可以直接调用的函数。
# 当出现emb_model = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, device)时，则是第一次实例化，
# 这个时候会准备好整个字典，位置编码矩阵。此时所有的矩阵都已固定，但 forward 一次都没运行过，模型还没见过你的输入数据。

#定义TokenEmbedding,nn.Embedding建立(vocab_size, d_model) 的二维浮点数矩阵。也就是整个大字典
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
#Embedding部分到这里结束


#测试用例
# 1. 准备极简参数
vocab_size = 10
d_model = 4
max_len = 10
drop_prob = 0.1
device = "cpu"

# 2. 实例化模型
model = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, device)

# 3. 造一个最简单的输入：1句话，3个词 (Batch=1, L=3)
# 数字代表字典里的 ID
input_ids = torch.tensor([[2, 3, 4]]) 
print(input_ids.shape)

# 4. 将数据喂给模型 (自动触发 forward)
output = model(input_ids)

# 5. 看结果
print("输入形状:", input_ids.shape)  # 预期: (1, 3)
print("输出形状:", output.shape)     # 预期: (1, 3, 4)
print("\n模型吐出的特征向量:\n", output)

  