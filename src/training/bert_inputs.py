import torch
import torch.nn as nn
import torch.nn.functional as F


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
    probability_matrix = torch.full(input_ids.shape, mask_prob, device=input_ids.device) # 15% of tokens to be masked
    
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
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=input_ids.device)).bool() & masked_indices # Out of 15% masked, 80% to [MASK]
    masked_input_ids[indices_replaced] = mask_token_id # replace mask index with [MASK] token id
    
    # 10% of masked tokens stay the same and rest 10% are assigned a random word (but we still predict them)
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=input_ids.device)).bool() & masked_indices & ~indices_replaced # Out of  15% masked, 10%/20% to random
    random_words = torch.randint(50260, input_ids.shape, dtype=torch.long, device=input_ids.device)
    masked_input_ids[indices_random] = random_words[indices_random]
    attention_mask = (masked_input_ids != pad_token_id)

    return masked_input_ids, mlm_labels, attention_mask
