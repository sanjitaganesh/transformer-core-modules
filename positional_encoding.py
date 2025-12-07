import numpy as np
def positional_encoding(max_len, d_model):
    #max_len: max len positional encoding table can take
    #d_model: embedding dimensionality
    pe = np.zeros((max_len, d_model))
    #creates a postional encoding table with rows of the psotions and columns of their encodings

    position = np.arange(max_len)[:, np.newaxis]
    #reshapes [0,1,2,...] into (T,1) so broadcasting can produce (T,1) × (1,d_model/2) → (T,d_model/2)

    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    #indexes for even embedding dimensions then This exponent term generates geometrically decreasing frequencies

    pe[:, 0::2] = np.sin(position * div_term)
    #Every even column (0,2,4...) receives sine waves of different frequencies
    pe[:, 1::2] = np.cos(position * div_term)
    #Every odd column (1,3,5...) receives cos waves of different frequencies

    #alternating sin and cos are used to avoid overlapping to encode relative positions using shifts

    return pe
