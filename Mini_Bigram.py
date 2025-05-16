#!/usr/bin/env python
# coding: utf-8

# In[4]:


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(len(text))  


# In[5]:


print(text[:1000])


# In[6]:


char= sorted(list(set(text)))
vocab_size=len(char)
print("".join(char))
print(vocab_size)


# character level tokenizer

# In[7]:


stoi = {ch: i for i, ch in enumerate(char)}
itos = {i: ch for i, ch in enumerate(char)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

print(encode("hii"))
print(decode(encode("hii")))


# In[8]:


import torch
import torch.nn as nn
from torch.nn import functional as F

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# In[9]:


block_size=8
train_data[:block_size+1]


# In[10]:


x= train_data[:block_size]
y=train_data[1:block_size+1]

for t in range(block_size):
    context= x[:t+1]
    target= y[t]
    print(f"when input {context} my target {target}")


# In[11]:


torch.manual_seed(1337)
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# In[12]:


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
    return x, y
xb, yb = get_batch('train')


# In[15]:


@torch.no_grad()

def estimateloss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses= torch.zeros(eval_iters)
        for k in range (eval_iters):
            X,Y= get_batch(split)
            logits, loss= model(X,Y)
            
            losses[k]= loss.item()
        out[split]= losses.mean()
    model.train()
    return out


# In[25]:


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m=BigramLanguageModel(vocab_size)
logits, loss= m(xb, yb)
print(logits.shape)
print(loss)

start_token = torch.tensor([[0]], dtype=torch.long)
generated_ids = m.generate(start_token, 100)  # shape: (1, 101)
decoded_text = decode(generated_ids[0].tolist())  # shape: (101,) → list of ints
print(decoded_text)


# In[ ]:


optimizer = torch.optim.AdamW(m.parameters(), lr=0.01)

for iter in range(100):

    # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


# mathematical trick in self attention

# In[20]:


torch.manual_seed(1337)
B,T,C= 4,8,2
x= torch.randn(B,T,C)
x.shape


# In[23]:


wei= torch.tril(torch.ones(T,T))
wei= wei / wei.sum(1, keepdim=True)
xbow= wei@x
xbow


# In[ ]:




