import math

import torch
from torch import nn


class Module(nn.Module):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))


class FeedForwardNetword(Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear_in = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.linear_out = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.activation(x)
        x = self.linear_out(x)
        return x


# the following implementation of Multi Head Attention is slow and unsuitable for training
class ScaledDotProductAttention(Module):

    def __init__(self, d_model, head_size, use_mask=False):
        super().__init__()
        self.head_size = head_size
        self.w_q = nn.Linear(d_model, head_size, bias=False)  # (batch_size, head_size)
        self.w_k = nn.Linear(d_model, head_size, bias=False)
        self.w_v = nn.Linear(d_model, head_size, bias=False)
        self.use_mask = use_mask
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        q = self.w_q(q)  # (batch_size, sec_len, d_model) @ (d_model, head_size) = (batch_size, sec_len, head_size)
        k = self.w_k(k)
        v = self.w_v(v)

        rel = q @ k.transpose(-2, -1)  # (batch_size, sec_length, sec_length)
        rel = rel * self.head_size ** -0.5

        if self.use_mask:
            sec_len = rel.size(-1)
            mask = torch.tril(torch.ones(sec_len, sec_len, requires_grad=False)).to(self.device)
            rel = rel.masked_fill(mask == 0, float('-inf'))

        value_weights = self.softmax(rel)

        return value_weights @ v  # (batch_size, sec_length, sec_length) @ ()


# TODO: implement multi head attention with broadcasting for better performance
class MultiHeadAttention(Module):

    def __init__(self, d_model, num_heads, use_mask=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_mask = use_mask
        self.head_size = d_model // self.num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # (batch_size, head_size)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        q = self.w_q(q)  # (batch_size, sec_len, d_model) @ (d_model, d_model) = (batch_size, sec_len, d_model)
        k = self.w_k(k)
        v = self.w_v(v)

        q_heads = self.split_heads(q)  # (batch_size, num_heads, sec_len1, head_size)
        k_heads = self.split_heads(k)  # (batch_size, num_heads, sec_len2, head_size)
        v_heads = self.split_heads(v)  # (batch_size, num_heads, sec_len2, head_size)

        # (batch_size, num_heads, sec_len1, head_size) @ (batch_size, num_heads, head_size, sec_len2) = (batch_size, num_heads, sec_len1, sec_len2)
        rel = q_heads @ k_heads.transpose(-2, -1)
        rel = rel * self.head_size ** -0.5

        if self.use_mask:
            sec_len = rel.size(-1)
            mask = torch.tril(torch.ones(sec_len, sec_len, requires_grad=False, )).to(self.device)
            rel = rel.masked_fill(mask == 0, float('-inf'))

        value_weights = self.softmax(rel)

        # (batch_size, num_heads, sec_len1, sec_len2) @ (batch_size, num_heads, sec_len2, head_size) = (batch_size, num_heads, sec_len1, head_size)
        attn_heads = value_weights @ v_heads

        merged = self.merge_heads(attn_heads)

        return self.w_o(merged)

    def split_heads(self, x):
        B, S, C = x.shape
        head_size = C // self.num_heads
        x = x.view(B, S, self.num_heads, head_size)

        return x.transpose(2, 1)  # (B, num_heads, S, head_size)

    def merge_heads(self, x):
        B, H, S, C = x.shape
        x = x.transpose(1, 2)  # (B, S, num_heads, head_size)

        return x.reshape(B, S, H * C)  # H*C=d_model


class EncoderLayer(Module):

    def __init__(self, d_model, d_ff, num_heads, dropout_rate):
        super().__init__()
        self.encoderLayers = []

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetword(d_model, d_ff)
        self.dropout_2 = nn.Dropout(p=dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        att_out = self.attention(x, x, x)
        x = self.layer_norm_1(x + self.dropout_1(att_out))

        ffn_out = self.ffn(x)
        x = self.layer_norm_2(x + self.dropout_2(ffn_out))
        return x


class DecoderLayer(Module):

    def __init__(self, d_model, d_ff, num_heads, dropout_rate):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, use_mask=True)
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(d_model)

        self.enc_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout_2 = nn.Dropout(p=dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetword(d_model, d_ff)
        self.dropout_3 = nn.Dropout(p=dropout_rate)
        self.layer_norm_3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        att_out = self.attention(x, x, x)
        x = self.layer_norm_1(x + self.dropout_1(att_out))

        enc_att_out = self.enc_attention(q=x, k=enc_out, v=enc_out)
        x = self.layer_norm_2(x + self.dropout_2(enc_att_out))

        fnn_out = self.ffn(x)
        x = self.layer_norm_3(x + self.dropout_3(fnn_out))
        return x


class PositionalEncoding(Module):

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Embedding(Module):

    def __init__(self, vocab_size, d_model, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, 2)  # 2 is the eos token, we also pad with it
        self.dropout = nn.Dropout(dropout_rate)
        # TODO: implement positional encodings as part of the model (compute during initialization, use register buffer)
        self.pos_enc = PositionalEncoding(d_model, 0.1, 1000)

    def forward(self, x):
        emb_out = self.embedding(x)  # (batch, sequence, embedding)
        emb_out = self.pos_enc(emb_out)
        return self.dropout(emb_out)


class Transformer(Module):

    def __init__(self, vocab_size, d_model, d_ff, num_heads=8, N=6, seed=5012025, dropout_rate=0.1):
        super().__init__()
        torch.manual_seed(seed)
        self.source_embedding = Embedding(vocab_size, d_model, dropout_rate=dropout_rate)
        self.target_embedding = Embedding(vocab_size, d_model, dropout_rate=dropout_rate)

        self.encoder_stack = nn.ModuleList([EncoderLayer(d_model, d_ff, num_heads, dropout_rate) for _ in range(N)])
        self.decoder_stack = nn.ModuleList([DecoderLayer(d_model, d_ff, num_heads, dropout_rate) for _ in range(N)])

        self.linear = nn.Linear(d_model, vocab_size)

        self.to(self.device)

        torch.seed()

    def forward(self, source, target):

        enc_out = self.source_embedding(source)
        for encoder_layer in self.encoder_stack:
            enc_out = encoder_layer(enc_out)

        dec_out = self.target_embedding(target)
        for decoder_layer in self.decoder_stack:
            dec_out = decoder_layer(dec_out, enc_out)

        return self.linear(dec_out)
