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