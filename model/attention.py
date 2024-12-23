import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result to shape (batch_size, num_heads, seq_len, depth).
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)                    # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(self.wk(k), batch_size)                    # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(self.wv(v), batch_size)                    # (batch_size, num_heads, seq_len_v, depth)

        attention, weights = self.scaled_dot_product_attention(q, k, v, mask)
        attention = attention.permute(0, 2, 1, 3).contiguous()          # (batch_size, seq_len_q, num_heads, depth)
        attention = attention.view(batch_size, -1, self.d_model)        # (batch_size, seq_len_q, d_model)

        output = self.fc_out(attention)                                 # (batch_size, seq_len_q, d_model)

        return output, weights
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))                # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        dk = q.size(-1)                                                 # Depth of the query
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += mask * -1e9

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # Normalize over seq_len_k
        output = torch.matmul(attention_weights, v)                     # Shape: (batch_size, num_heads, seq_len_q, depth)

        return output, attention_weights