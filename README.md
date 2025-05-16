# 🔥 Transformer & Bigram Language Models - From Scratch in PyTorch

This repository contains two educational implementations of language models:

1. **Bigram Language Model** — A simple statistical baseline using character-level prediction.
2. **GPT-style Transformer Language Model** — A powerful deep learning model based on the original Transformer architecture, built entirely from scratch in PyTorch.

Perfect for anyone learning how large language models work under the hood.

---

## 🚀 What's Inside?

### 🧠 1. Bigram Language Model

- Character-level model based on bigram statistics
- Learns the probability of each character given the previous one
- Implements counting, log-likelihood evaluation, sampling, and training using PyTorch
- Great for educational comparison to Transformers

📄 File: `bigram.py`

---

### 🤖 2. Transformer Language Model (GPT-style)

- Built using multi-head self-attention, feedforward layers, layer normalization, residual connections, and token embeddings
- Supports training and autoregressive text generation
- Fully compatible with PyTorch GPU acceleration

📄 File: `transformer.py`

---

## 🛠️ Model Architecture

### 🔷 GPTLanguageModel

- **Token + Positional Embeddings**
- **Multi-Head Self-Attention**
- **Feedforward MLPs**
- **LayerNorm + Residuals**
- **Autoregressive Sampling (`generate()`)**

Built using:

- `Head` → single self-attention head
- `MultiHeadAttention` → combines multiple heads
- `FeedForward` → 2-layer MLP
- `Block` → Transformer block (Attention + FF + Norm)
- `GPTLanguageModel` → stacks multiple blocks + sampling

---

## 📊 Training Setup

- Tokenized dataset using character-level vocabulary
- Supports batching, training, and evaluation loss computation
- Hyperparameters:
  - `n_embd`: embedding size (e.g., 384)
  - `n_layer`: number of transformer blocks
  - `n_head`: number of attention heads
  - `block_size`: max context length

---

## 📈 Sample Usage

```python
from transformer import GPTLanguageModel

model = GPTLanguageModel()
output = model.generate(start_token=torch.tensor([[0]]), max_new_tokens=100)
print(decode(output[0].tolist()))
```

For the bigram model:
```python
from bigram import BigramLanguageModel
model = BigramLanguageModel()
# Train or sample from the bigram model
```

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- (Optional) `tiktoken` for OpenAI-compatible tokenization

---

## 📚 Learning Objectives

- Understand core building blocks of modern LLMs
- See how simple bigram models compare to full Transformers
- Practice training, sampling, and visualization of self-attention

---

## 🙌 Credits

Inspired by the excellent teaching style of [Andrej Karpathy](https://github.com/karpathy)  
Assembled with ❤️ for learning and educational clarity.

---
