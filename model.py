from torch import nn
import torch
import numpy as np

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_h, mask=False):
        super().__init__()

        # Dimensions of the model
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_h = d_h

        self.mask = mask

        # Generate Q, K, V
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_v)

        # Weights that project Q, K, V to different layers
        self.q_layer_weights = [nn.Linear(d_k, d_k) for _ in range(d_h)]
        self.k_layer_weights = [nn.Linear(d_k, d_k) for _ in range(d_h)]
        self.v_layer_weights = [nn.Linear(d_v, d_v) for _ in range(d_h)]

        # Projects the concatenated representations to a single vector
        self.final_proj = nn.Linear(self.d_h * self.d_v, self.d_model)

        # Create mask
        if mask:
            self.mask = np.ones(d_t, d_t)
            for r in range(0, d_t):
                for c in range(r+1, d_t):
                    self.mask[r][c] = - np.inf

    def forward(self, X):
        # X (d_t, d_model)
        d_t = X[0]

        Q = self.q_proj(X) # (d_t, d_k)
        K = self.k_proj(X) # (d_t, d_k)
        V = self.v_proj(X) # (d_t, d_v)

        softmax = nn.Softmax(d_t)

        representations = list()

        # Loop through each layer of the head
        for w_q, w_k, w_v in \
            zip(self.q_layer_weights, self.k_layer_weights, self.v_layer_weights):
            q_layer = w_q(Q) # (d_t, d_k)
            k_layer = w_k(K) # (d_t, d_k)
            v_layer = w_v(V) # (d_t, d_v)

            attention = np.matmul(q_layer,k_layer.T) # (d_t, d_t)
            attention = attention / torch.sqrt(self.d_k)

            if self.mask:
                attention = np.multiply(attention, self.mask)

            attention = softmax(attention)

            representation = np.matmul(attention, v_layer) # (d_t, d_v)
            representations.append(representation)

        representations = np.hstack(representations) # (d_t, h x d_v)

        return self.final_proj(representations)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

        self.expand_layer = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU(inplace=True)
        self.shrink_layer = nn.Linear(d_ff, d_model)

    def forward(self, X):
        # X (d_t, d_model)
        X = self.expand_layer(X)
        self.relu(X)
        return self.shrink_layer(X)

        
        

