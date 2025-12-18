import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def multi_head_attention(X, Wq, Wk, Wv, Wo, h):
    """
    GPT-2 style causal multi-head self-attention (NumPy).

    X  : (B, T, d_model)
    Wq : (d_model, d_model)
    Wk : (d_model, d_model)
    Wv : (d_model, d_model)
    Wo : (d_model, d_model)
    h  : number of heads
    """

    B, T, d_model = X.shape
    assert d_model % h == 0, "d_model must be divisible by number of heads"

    d_head = d_model // h

    # Linear projections
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Split into heads
    Q = Q.reshape(B, T, h, d_head).transpose(0, 2, 1, 3)  # (B, h, T, d_head)
    K = K.reshape(B, T, h, d_head).transpose(0, 2, 3, 1)  # (B, h, d_head, T)
    V = V.reshape(B, T, h, d_head).transpose(0, 2, 1, 3)  # (B, h, T, d_head)

    # Scaled dot-product attention
    scores = (Q @ K) / np.sqrt(d_head)  # (B, h, T, T)

    # Causal mask (prevent attending to future tokens)
    mask = np.triu(np.ones((T, T)), k=1)
    scores = scores + mask * -1e9

    weights = softmax(scores)
    head_output = weights @ V  # (B, h, T, d_head)

    # Merge heads
    output = head_output.transpose(0, 2, 1, 3).reshape(B, T, d_model)

    # Final linear projection
    output = output @ Wo
    return output
