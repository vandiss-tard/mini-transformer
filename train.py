import torch

text = open('data/input.txt', encoding='utf-8').read()
print('文本长度:', len(text))

chars = sorted(set(text))
vocab_size = len(chars)
print('词表大小:', vocab_size)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
print('data shape:', data.shape, data.dtype)
print('校验:', decode(encode('Hello')) == 'Hello')
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 128
batch_size = 32

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i     : i + block_size    ] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('xb shape:', xb.shape)
print('yb shape:', yb.shape)
print('输入:', repr(decode(xb[0].tolist())))
print('答案:', repr(decode(yb[0].tolist())))