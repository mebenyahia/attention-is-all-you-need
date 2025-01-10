import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, normalizers
from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import json
import os


class Module(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))


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
        self.w_q = nn.Linear(d_model, head_size, bias=False) # (batch_size, head_size)
        self.w_k = nn.Linear(d_model, head_size, bias=False)
        self.w_v = nn.Linear(d_model, head_size, bias=False)
        self.use_mask = use_mask        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v):
        q = self.w_q(q) # (batch_size, sec_len, d_model) @ (d_model, head_size) = (batch_size, sec_len, head_size)
        k = self.w_k(k) 
        v = self.w_v(v)
        
        rel = q @ k.transpose(-2, -1) # (batch_size, sec_length, sec_length)
        rel = rel * self.head_size**-0.5 
        
        if self.use_mask:
            sec_len = rel.size(-1)
            mask = torch.tril(torch.ones(sec_len, sec_len, requires_grad=False,)).to(self.device)
            rel = rel.masked_fill(mask == 0, float('-inf'))
        
        value_weights = self.softmax(rel)
        
        return value_weights @ v # (batch_size, sec_length, sec_length) @ ()

        
class MultiHeadAttention(Module):
    
    def __init__(self, d_model, num_heads, use_mask=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_mask = use_mask
        
        head_size = d_model // num_heads
        self.heads = nn.ModuleList([ScaledDotProductAttention(d_model, head_size, use_mask) for _ in range(num_heads)])
        self.w_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, q, k, v):
        merged = torch.cat([head(q, k, v) for head in self.heads], dim=-1)
        return self.w_o(merged)


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


class Embedding(Module):
        
    def __init__(self, vocab_size, d_model, pos_enc, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, 2) # 2 is the eos token, we also pad with it
        self.dropout = nn.Dropout(dropout_rate)
        self.pos_enc = pos_enc
            
    def forward(self, x):
        emb_out = self.embedding(x) # (batch, sequence, embedding)
        sec_len = emb_out.shape[1]
        return self.dropout(emb_out + self.pos_enc[:sec_len,:])
    

class Transformer(Module):
        
    def __init__(self, vocab_size, d_model, d_ff, pos_enc, num_heads=8, N=6, seed=5012025, dropout_rate=0.1):
        super().__init__()
        torch.manual_seed(seed)
        self.source_embedding = Embedding(vocab_size, d_model, pos_enc, dropout_rate=dropout_rate) 
        self.target_embedding = Embedding(vocab_size, d_model, pos_enc, dropout_rate=dropout_rate)
        

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