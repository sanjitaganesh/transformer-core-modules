import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) #subtracts max value
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True) #divides by sum to get probability

def multi_head_attention(X,Wq,Wk,Wv,Wo,h):
    B,T,d_model=X.shape
    assert d_model%h==0

    d_head = d_model // h

    #linear projections
    Q=X@Wq
    K=X@Wk
    V=X@Wv

    #split heads
    Q=Q.reshape(B,T,h,d_head)
    Q=Q.transpose(0,2,1,3)

    K=K.reshape(B,T,h,d_head)
    #creating two transposes to finally get (B,h,d,T)
    K=K.transpose(0,2,1,3)
    K=K.transpose(0,1,3,2)

    V=V.reshape(B,T,h,d_head)
    V=V.transpose(0,2,1,3)

    #scaled dot product-attention
    scores=Q@K
    scores=scores/np.sqrt(d_head)
    weights=softmax(scores)

    head_output=weights@V

    #merging the heads
    output=head_output.transpose(0,2,1,3)
    output=output.reshape(B,T,d_model)

    #final linear projection
    output=output@Wo
    return output

