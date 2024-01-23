import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.cuda.amp import autocast


EPS = 1e-6


class ScaleDotProduct(nn.Module):
    def __init__(self, scale=1) -> None:
        super(ScaleDotProduct, self).__init__()
        self.scale = scale
    
    def forward(self, Q, K, V, mask=None):
        attn = torch.matmul(Q / self.scale, K.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask==0, torch.tensor(-1e6).half())
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        return out, attn
    

class RoPE(nn.Module):
    def __init__(self, d_model, max_len=5000) -> None:
        super(RoPE, self).__init__()
        self.freqs_cis = self.precompute_freqs_cis(d_model, max_len)

    def precompute_freqs_cis(self, d_model, max_len, theta=10000.0):
        freqs = 1.0/(theta ** (torch.arange(0, d_model, 2)[: (d_model//2)].float()/d_model))
        t = torch.arange(max_len)
        freqs = torch.outer(t, freqs).float()  
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    def broadcast(self, freqs_cis, x):
        dim_n = len(x.shape)
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]) 
        shape = [d if i == 1 or i == dim_n - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape).to(x.device)
    
    def forward(self, x):
        with autocast(enabled=False):
            x_= x.reshape(*x.shape[:-1], -1, 2).float()

            x_ = torch.view_as_complex(x_)
            freqs_cis = self.broadcast(self.freqs_cis[: x.shape[1]], x_)
            x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
        return x_out.type_as(x)
    

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, bias=True):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=bias)  
        self.w2 = nn.Linear(hidden, d_model, bias=bias)  
        self.w3 = nn.Linear(d_model, hidden, bias=bias)  

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) 

    def _norm(self, x):
        # RMSNorm
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.n_head = config.n_head
        self.q_k_dim = config.q_k_dim  
        self.v_dim = config.v_dim  

        self.Q_weight = nn.Linear(config.d_model, self.n_head * self.q_k_dim, bias=False)
        self.K_weight = nn.Linear(config.d_model, self.n_head * self.q_k_dim, bias=False)
        self.V_weight = nn.Linear(config.d_model, self.n_head * self.v_dim, bias=False)
        
        self.out_weight = nn.Linear(self.n_head * self.v_dim, config.d_model, bias=False)

        self.attention = ScaleDotProduct(config.d_model)
        self.pe = RoPE(config.q_k_dim)

    def forward(self, q, k, v, mask=None):
        batch_n, q_l, k_v_l = q.shape[0], q.shape[1], k.shape[1]

        Q = self.Q_weight(q).view(batch_n, q_l, self.n_head, self.q_k_dim)
        K = self.K_weight(k).view(batch_n, k_v_l, self.n_head, self.q_k_dim)
        V = self.V_weight(v).view(batch_n, k_v_l, self.n_head, self.v_dim)

        Q, K = self.pe(Q), self.pe(K)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  
            
        out_Q, attn = self.attention(Q, K, V, mask)
        out_Q = out_Q.transpose(1, 2).contiguous().view(batch_n, q_l, -1)
        out_Q = self.out_weight(out_Q)

        return out_Q
    

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config.d_model, config.d_model * 2, config.ff_bias!=0)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x, mask=None):
        norm_x = self.attention_norm(x)
        h = x + self.attention(norm_x, norm_x, norm_x, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class _Embedding(nn.Module):
    def __init__(self, config):
        super(_Embedding, self).__init__()
        assert config.seq_len % config.patch_len == 0
        self.patch_len = config.patch_len
        self.lin = nn.Linear(config.patch_len*config.in_dim, config.d_model)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, self.patch_len, self.x.shape[-1]).flatten(2)
        emb = self.lin(x.contiguous())
        return emb


class Transformer(nn.Module):
    def __init__(self, config, readout=True) -> None:
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(config.n_layer):
            self.layers.append(TransformerBlock(config))
        self.to_embedding = nn.Linear(config.in_dim, config.d_model, bias=False)
        self.norm = RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.emb_dim, bias=False)
        self.readout = readout
        if readout:
            self.cls_token = nn.Parameter(torch.rand(1, 1, config.d_model))

    def forward(self, x: torch.Tensor):
        h = self.to_embedding(x.contiguous())
        if self.readout:
            cls_token = self.cls_token.expand((h.shape[0], -1, -1))
            h = torch.cat((cls_token, h), dim=1).contiguous()

        for layer in self.layers:
            h = layer(h, mask=None)
        
        if self.readout:
            cls_emb = h[:, 0, :] 
            out = self.output(self.norm(cls_emb))
        else:
            h = self.norm(h)
            out = self.output(h)

        return out


class DomainLoss(nn.Module):
    def __init__(self):
        super(DomainLoss, self).__init__()
        self.ce_loss_none = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, sample_idx=None):
        loss_out = self.ce_loss_none(pred, target)
        if sample_idx is not None:
            loss = torch.mul(loss_out, sample_idx).mean()
        else:
            loss = loss_out.mean()
        return loss
    

class FreqEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(FreqEncoder, self).__init__()
        self.conv_a = nn.Sequential(
            nn.Conv1d(config.in_dim, config.d_model, kernel_size=(5, ), padding=(2, ), bias=False),
            nn.BatchNorm1d(config.d_model),
            nn.ReLU(),
        )
        
        self.conv_p = nn.Sequential(
            nn.Conv1d(config.in_dim, config.d_model, kernel_size=(5, ), padding=(2, ), bias=False),
            nn.BatchNorm1d(config.d_model),
            nn.ReLU(),
        )

        self.merge_conv = nn.Sequential(
            nn.Conv1d(config.d_model*2, config.d_model, kernel_size=(5, ), padding=(2, ), bias=False),
            nn.BatchNorm1d(config.d_model)
            )
        self.conv1 = nn.Sequential(
            nn.Conv1d(config.d_model, config.d_model, kernel_size=(5, ), padding=(2, ), bias=False),
            nn.BatchNorm1d(config.d_model),
            nn.ReLU(),      
            nn.Conv1d(config.d_model, config.emb_dim, kernel_size=(3, ), padding=(1, ), bias=False),
            nn.BatchNorm1d(config.emb_dim)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(config.d_model, config.d_model, kernel_size=(3, ), padding=(1, ), bias=False),
            nn.BatchNorm1d(config.d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),  
            
            nn.Conv1d(config.d_model, config.d_model, kernel_size=(3, ), padding=(1, ), bias=False),
            nn.BatchNorm1d(config.d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2), 

            nn.Flatten(1),
            nn.Linear(config.d_model*config.n_freq//4, config.emb_dim),
            nn.Dropout()
        )
    
    def forward(self, x_a, x_p):
        conv_a = self.conv_a(x_a)
        conv_p = self.conv_p(x_p)
        merge_ap = torch.cat((conv_a, conv_p), dim=1)
        merge_emb = self.merge_conv(merge_ap)
        emb = self.conv1(merge_emb)
        out = self.conv2(merge_emb + emb)
        return out


class CNNEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(config.in_dim, config.d_model, kernel_size=(7, ), padding=(3, ), stride=1, bias=False),
            nn.BatchNorm1d(config.d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(config.d_model, config.d_model*2, kernel_size=(5, ), padding=(2, ), stride=1, bias=False),
            nn.BatchNorm1d(config.d_model*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(config.d_model*2, config.d_model, kernel_size=(3, ), padding=(1, ), stride=1, bias=False),
            nn.BatchNorm1d(config.d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 

            nn.Conv1d(config.d_model, config.emb_dim, kernel_size=(3, ), padding=(1, ), stride=1, bias=False),
            nn.BatchNorm1d(config.emb_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 

            nn.Flatten(1, 2),
            nn.Linear(config.emb_dim*8, config.emb_dim),
            nn.Dropout(0.5),
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        out = self.conv3(c2)

        return out
    

class Chomp1d(nn.Module):
    def __init__(self, chopm_size):
        super(Chomp1d, self).__init__()
        self.chopm_size = chopm_size

    def forward(self, x):
        return x[:, :, : -self.chopm_size].contiguous()
    

class TemporalBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(out_dim, out_dim, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.short_cut = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Sequential()
    
    def forward(self, x):
        out = self.conv(x)
        res = self.short_cut(x)
        return F.relu(out+res)    


class TCNEncoder(nn.Module):
    def __init__(self, config):
        super(TCNEncoder, self).__init__()
        layers = []
        dilation = 1
        i = 0
        while dilation <= config.seq_len:
            in_dim = config.in_dim if i == 0 else config.d_model
            out_dim = config.d_model
            padding = 2*dilation
            layers.append(TemporalBlock(in_dim, out_dim, 3, 1, dilation, padding, dropout=0.5))
            i += 1
            dilation = 2**i
        self.net = nn.Sequential(*layers)
        self.to_emb = nn.Sequential(
            nn.Conv1d(config.d_model, config.d_model, 3, padding=1, bias=False),
            nn.BatchNorm1d(config.d_model),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2), 

            nn.Conv1d(config.d_model, config.d_model, 3, padding=1, bias=False),
            nn.BatchNorm1d(config.d_model),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(config.d_model, config.d_model, 3, padding=1, bias=False),
            nn.BatchNorm1d(config.d_model),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(config.d_model, config.d_model, 3, padding=1, bias=False),
            nn.BatchNorm1d(config.d_model),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2),

            nn.Flatten(1),
            nn.Linear(config.d_model*8, config.emb_dim),
            nn.Dropout()          
        )

    def forward(self, x):
        out_tcn = self.net(x.transpose(1, 2))  
        emb = self.to_emb(out_tcn)
        return emb
    

class RNNEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(RNNEncoder, self).__init__()
        self.rnn = nn.GRU(config.in_dim, config.d_model, config.n_layer, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Sequential(
            nn.Linear(config.d_model, config.emb_dim),
            nn.Dropout(0.5)
        )

    def forward(self, x) -> torch.Tensor:
        self.rnn.flatten_parameters()
        seq_out, embs = self.rnn(x)
        out = self.fc(embs[-1])

        return out
    

class F_Encoder(nn.Module):
    def __init__(self, config):
        super(F_Encoder, self).__init__()
        if config.freq:
            self.encoder = FreqEncoder(config)
        else:
            self.encoder = None

    def forward(self, xa, xp):
        return None if self.encoder is None else self.encoder(xa, xp)
    

class T_Encoder(nn.Module):
    def __init__(self, config):
        super(T_Encoder, self).__init__()
        self.Encoder_dict = {
            'cnn': CNNEncoder,
            'rnn': RNNEncoder,
            'transformer': Transformer,
            'tcn': TCNEncoder}
        if config.time:
            self.encoder = self.Encoder_dict[config.encoder](config)
        else:
            self.encoder = None
    
    def forward(self, x):
        return None if self.encoder is None else self.encoder(x)    
