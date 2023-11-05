from torch import nn
class FF(nn.Module):
    def __init__(self,
                 inp_len : int,
                 out_len : int,
                 ):
        self.ff = nn.Sequential(torch.linear(inp_len, out_len),
                                     nn.ReLU(),
                                     torch.linear(out_len, out_len),
                                     )
    def forward(x : torch.Tensor) -> torch.Tensor:
        return self.ff(x)
class WordEmbedding(nn.Module):
    pass
class Dropout(nn.Module):
    pass
class PositionalEncoding(nn.Module):
    def __init__(self, 
            d_model : int,
                 seq_len : int = 10000,
                 ):
        pos = torch.arange(0, seq_len)
        even_terms = 2*torch.arange(0,d_model)
        freq = 10000.**(even_terms/d_model)
        sins = torch.sin(pos/freq)
        coss = torch.cos(pos/freq)
        pe = torch.zeros((seq_len, d_model))
        pe[:,even_terms] = sins
        pe[:,even_terms+1] = coss
        self.pe = pe

    def forward(self, 
                embedding: torch.Tensor) -> torch.Tensor:
        return embedding + self.pe

class MultiheadAttention(nn.Module):
    pass

