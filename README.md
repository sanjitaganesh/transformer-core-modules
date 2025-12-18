# Transformer Core Modules (GPT-2 Style)

This project implements and documents the core building blocks of a **decoder-only Transformer architecture (GPT-2 style)**, with emphasis on **control flow and data flow through the model**, closely matching an actual implementation.

The repository is intended to demonstrate *implementation-level understanding* of how GPT-style Transformers process input tokens.

---

## Architecture Overview

The Transformer follows the GPT-2 design:
- Decoder-only stack
- Causal (masked) self-attention
- Pre-LayerNorm architecture
- Residual connections
- Weight-tied output projection

A detailed control-flow diagram is included to explicitly show how tensors move through the model.

---

## Deliverables

- **GPT-2 Control Flow Diagram** (tokenization → embeddings → transformer blocks → output projection)
- Positional Encoding module
- Multi-Head Self-Attention module (implemented from scratch)

---

## Files

- `positional_encoding.py`  
  Generates **positional encodings** (used for conceptual understanding; GPT-2 itself uses learned positional embeddings).

- `multi_head_attention.py`  
  Standalone NumPy implementation of **causal multi-head self-attention**, including:
  - QKV projection
  - Scaled dot-product attention
  - Causal masking
  - Head concatenation

- `gpt2_control_flow_diagram.pdf`  
  Detailed control-flow diagram illustrating how input tokens are processed through a GPT-2 style Transformer, including:
  - Tokenization
  - Embedding lookup
  - Attention internals
  - Residual paths
  - Layer normalization
  - Output projection

---

## Requirements

- Python 3.9+
- NumPy

---

## Author

Sanjita Bhaavya Ganesh
