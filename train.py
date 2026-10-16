"""训练一个字符级 mini GPT。"""
import torch
from model import MiniGPT

# ---------------- 1. 设备 ----------------
torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备:", device)

# ---------------- 2. 数据与词表 ----------------
text = open("data/input.txt", encoding="utf-8").read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"文本 {len(text)} 字符 | 词表 {vocab_size} | 训练 {len(train_data)} | 验证 {len(val_data)}")

# ---------------- 3. 超参数 ----------------
block_size = 128      # 上下文长度
batch_size = 32       # 每批样本数
steps = 200          # 训练步数（想更快可以先改成 300 看效果）

# ---------------- 4. 取样 ----------------
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i     : i + block_size    ] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------- 5. 模型 ----------------
model = MiniGPT(vocab_size, n_embd=128, n_head=4, n_layer=4,
                block_size=block_size, dropout=0.1).to(device)
n_param = sum(p.numel() for p in model.parameters())
print(f"参数量: {n_param / 1e6:.2f} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# ---------------- 6. 评估 ----------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(20)
        for k in range(20):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ---------------- 7. 训练 ----------------
for step in range(steps):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 500 == 0 or step == steps - 1:
        ls = estimate_loss()
        print(f"step {step:5d} | train {ls['train']:.4f} | val {ls['val']:.4f}")

# ---------------- 8. 保存 ----------------
torch.save(model.state_dict(), "model.pt")
print("模型已保存到 model.pt")

# ---------------- 9. 生成 ----------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n===== 生成结果 =====")
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))