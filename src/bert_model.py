import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from bert_components import BertEncoderLayer
from bert_components import BertEmbeddings
from bert_components import BertPooler


def create_mlm_inputs_and_labels(input_ids, mask_token_id=50258, pad_token_id=50259, mask_prob=0.15):
    """
    Create masked language model inputs and labels. BERT randomly assigns masks to 15% of sequence. 
    In this 15%, 80% are replaced with [MASK], 10% are replaced with random token, and 10% are left unchanged.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        mask_token_id: Token ID for [MASK]
        pad_token_id: Token ID for [PAD] 
        mask_prob: Probability of masking each token
    
    Returns:
        masked_input_ids: Input with some tokens masked
        mlm_labels: Labels for MLM task (-100 for non-masked tokens)
    """
    masked_input_ids = input_ids.clone()
    mlm_labels = input_ids.clone()
    
    # Create probability matrix for masking based on mask_prob
    probability_matrix = torch.full(input_ids.shape, mask_prob) # 15% of tokens to be masked
    
    # Don't mask special tokens ([CLS], [SEP], [PAD])
    special_tokens_mask = (
        (input_ids == 50256) |  # [CLS]
        (input_ids == 50257) |  # [SEP]  
        (input_ids == pad_token_id)  # [PAD]
    )
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # Create mask
    masked_indices = torch.bernoulli(probability_matrix).bool() # 80% masked
    
    # Only compute loss on masked tokens
    mlm_labels[~masked_indices] = -100 # loss ignored for non masked tokens
    
    # 80% of masked tokens become [MASK]
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices # Out of 15% masked, 80% to [MASK]
    masked_input_ids[indices_replaced] = mask_token_id # replace mask index with [MASK] token id
    
    # 10% of masked tokens stay the same and rest 10% are assigned a random word (but we still predict them)
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced # Out of  15% masked, 10%/20% to random
    random_words = torch.randint(50260, input_ids.shape, dtype=torch.long)
    masked_input_ids[indices_random] = random_words[indices_random]
    
    return masked_input_ids, mlm_labels


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


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
        attention_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, attention_mask)

        # 1. next sentence prediction
        h_pooled   = self.activ(self.fc(enc_outputs[:, 0])) # [batch_size, d_model]
        logits_nsp = self.classifier(h_pooled) # [batch_size, 2]

        # 2. predict the masked token
        masked_pos = masked_pos[:, :, None].expand(-1, -1, enc_outputs.size(-1)) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(enc_outputs, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked  = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_nsp



class BertForPreTraining(nn.Module):
    def __init__(self, vocab_size=50260, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512):
        super().__init__()
        
        self.bert = BertModel(vocab_size, hidden_size, num_hidden_layers, 
                             num_attention_heads, intermediate_size, max_position_embeddings)
        
        # MLM head
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        
        # NSP head  
        self.nsp_head = nn.Linear(hidden_size, 2)
        
    def forward(self, input_ids, token_type_ids, attention_mask=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        
        # MLM predictions
        mlm_scores = self.mlm_head(sequence_output)
        
        # NSP predictions
        nsp_scores = self.nsp_head(pooled_output)
        
        return mlm_scores, nsp_scores