import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoD_Attention_layer(nn.Module):
    def __init__(self, n=64, c=64):
        super(TwoD_Attention_layer, self).__init__()
        self.n = n
        self.c = c

        self.convq = nn.Conv2d(in_channels=n, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.convk = nn.Conv2d(in_channels=n, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.convv = nn.Conv2d(in_channels=n, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(in_channels=2 * c, out_channels=n, kernel_size=3, stride=1, padding=1)

        self.bnq = nn.BatchNorm2d(c)
        self.bnk = nn.BatchNorm2d(c)
        self.bnv = nn.BatchNorm2d(c)
        self.ln = nn.LayerNorm(n)

        self.final_conv1 = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, stride=1, padding=1)
        self.bnf1 = nn.BatchNorm2d(n)
        
        self.final_conv2 = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, stride=1, padding=1)
        self.bnf2 = nn.BatchNorm2d(n)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        residual = inputs

        q = self.bnq(self.convq(inputs))
        k = self.bnk(self.convk(inputs))
        v = self.bnv(self.convv(inputs))

        q_time, k_time, v_time = q.transpose(1, 3), k.transpose(1, 3), v.transpose(1, 3)
        q_fre, k_fre, v_fre = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        scaled_attention_time, _ = self.scaled_dot_product_attention(q_time, k_time, v_time)
        scaled_attention_fre, _ = self.scaled_dot_product_attention(q_fre, k_fre, v_fre)

        scaled_attention_time = scaled_attention_time.transpose(1, 3)
        scaled_attention_fre = scaled_attention_fre.transpose(1, 2)

        out = torch.cat([scaled_attention_time, scaled_attention_fre], dim=1)   

        out = self.conv(out) + residual     
        out = out.permute(0, 2, 3, 1)       
        out = self.ln(out)                  
        out = out.permute(0, 3, 1, 2)       

        final_out = self.bnf1(self.final_conv1(out))
        final_out = self.activation(final_out)
        final_out = self.bnf2(self.final_conv2(final_out))
        final_out = self.activation(final_out + out)

        return final_out

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        d_k = q.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=q.device))
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
    
class Pre_Net(nn.Module):
    def __init__(self, num_M=2, n=64, c=64):
        super(Pre_Net, self).__init__()
        self.num_M = num_M

        self.downsample1 = nn.Conv2d(in_channels=3, out_channels=n, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(n)

        self.downsample2 = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

        self.twoD_layers = nn.ModuleList([TwoD_Attention_layer(n, c) for _ in range(num_M)])

        nn.init.xavier_normal_(self.downsample1.weight)
        nn.init.xavier_normal_(self.downsample2.weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 3, 1, 2)

        out = F.tanh(self.bn1(self.downsample1(inputs)))
        out = F.tanh(self.bn2(self.downsample2(out)))

        for layer in self.twoD_layers:
            out = layer(out)

        return out

