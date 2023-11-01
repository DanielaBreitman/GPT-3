from torch import nn
import torch

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_h):
        super().__init__()

        # Dimensions of the model
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_h = d_h

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

    def forward(self, X):
        # X (d_t, d_model)
        d_t = X[0]

        Q = self.q_proj(X) # (d_t, d_k)
        K = self.k_proj(X) # (d_t, d_k)
        V = self.v_proj(X) # (d_t, d_v)

        representations = list()

        # Loop through each layer of the head
        for w_q, w_k, w_v in \
            zip(q_layer_weights, k_layer_weights, v_layer_weights):
            q_layer = w_q(Q) # (d_t, d_k)
            k_layer = w_k(K) # (d_t, d_k)
            v_layer = w_v(V) # (d_t, d_v)

            attention = np.matmul(q_layer,k_layer.T) # (d_t, d_t)
            attention = attention / torch.sqrt(self.d_k)
            attention = softmax(attention)

            representation = np.matmul(attention, v_layer) # (d_t, d_v)
            representations.append(representation)

        representations = np.hstack(representations) # (d_t, h x d_v)

        return final_proj(representations)




        

        
        

