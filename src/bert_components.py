import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings=512, type_vocab_size=3):
        super().__init__()
        # embedding takes sice of dictionary of embeddings and size of each embedding vector
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size) # Token embeddings
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size) # Position embeddings
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size) # Segment embeddings include PAD
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, token_type_ids):
        # alternatively, we can create position ids using max_position_embeddings
        # position_ids = torch.tensor([i for i in range(max_position_embeddings)], dtype=torch.long)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) # (seq_length, ) -> (batch_size, seq_length)
        
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = token_embeddings + position_embeddings + token_type_embeddings # Final embedding is sum of token, position and segment embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, hidden_size):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(hidden_size) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn 

class BertMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, Q, K, V, attention_mask=None):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        query_layer = self.query(Q).view(batch_size, -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        key_layer = self.key(K).view(batch_size, -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        value_layer = self.value(V).view(batch_size, -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        
        context, attn = ScaledDotProductAttention()(query_layer, key_layer, value_layer, attn_mask, self.attention_head_size )
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_attention_heads * self.hidden_size) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(self.num_attention_heads * self.attention_head_size, self.hidden_size)(context)
        return nn.LayerNorm(self.hidden_size)(output + residual), attn # output: [batch_size x len_q x d_model]


class BertPoswiseFeedForwardNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(F.gelu(self.fc1(x)))



class BertEncoderLayer(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        self.enc_self_attn = BertMultiHeadAttention()
        self.pos_ffn       = BertPoswiseFeedForwardNet()
        
    def forward(self, enc_inputs, attention_mask=None):
            enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, attention_mask) # enc_inputs to same Q,K,V
            enc_outputs = self.pos_ffn(enc_outputs)
            return enc_outputs


class Bert(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        self.embeddings = BertEmbeddings()
        self.layers = nn.ModuleList([BertEncoderLayer(num_hidden_layers, hidden_size, num_attention_heads, intermediate_size) for _ in range(num_hidden_layers)])
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(hidden_size, hidden_size) # for MLM
        self.norm = nn.LayerNorm(hidden_size)      # for MLM
        self.nsp_linear = nn.Linear(hidden_size, 2)       # for NSP
        # decoder is shared with embedding layer
        embed_weight = self.embeddings.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))


    def forward(self, enc_inputs, attention_mask=None):
        outputs = self.embeddings(enc_inputs)
        attention_mask = get_attn_pad_mask(enc_inputs)
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, attention_mask)
        return enc_outputs