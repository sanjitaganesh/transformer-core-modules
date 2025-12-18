import numpy as np

def positional_encoding(max_len, d_model):
    """
    Sinusoidal positional encoding (Attention Is All You Need).

    NOTE:
    GPT-2 uses learned absolute positional embeddings.
    This implementation is included for conceptual understanding
    of how positional information can be injected into token embeddings.

    max_len : maximum sequence length
    d_model : embedding dimension
    """

    pe = np.zeros((max_len, d_model))

    position = np.arange(max_len)[:, np.newaxis]

    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe

