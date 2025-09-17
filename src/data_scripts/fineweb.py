"""
FineWeb-Edu dataset (for BERT pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data for BERT-style pretraining with sentence pairs.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B" formatted for BERT.
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
from google.cloud import storage # pip install google-cloud-storage
import io
import tempfile
import random
import re

# ------------------------------------------
# GCS Configuration
gcs_bucket_name = "bert-training-data-sahil"
gcs_prefix = "edu_fineweb10B/"
storage_client = storage.Client()  # Create client
bucket = storage_client.bucket(gcs_bucket_name)  # Get bucket reference

# Local Configuration
local_dir = "../data/edu_fineweb10B"
remote_name = "sample-10BT" # 'sample-10BT'; 1 Billion tokens, 100 M token per shard and 100 shards in total -> ls edu_fineweb10B/ | wc -l
shard_size = int(1e7) # 10M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

# BERT special tokens
CLS_TOKEN = 50256  # [CLS] token
SEP_TOKEN = 50257  # [SEP] token  
MASK_TOKEN = 50258  # [MASK] token
PAD_TOKEN = 50259  # [PAD] token

def split_into_sentences(text):
    """Split text into sentences using simple regex"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def create_sentence_pairs(doc):
    """Create sentence pairs for NSP task"""
    text = doc["text"]
    sentences = split_into_sentences(text)
    
    if len(sentences) < 2:
        return []
    
    pairs = []
    
    # Create positive pairs (consecutive sentences)
    for i in range(len(sentences) - 1):
        pairs.append({
            'sentence_a': sentences[i],
            'sentence_b': sentences[i + 1], 
            'is_next': 1
        })
    
    # Create negative pairs (random sentences)
    for i in range(len(sentences) - 1):
        if random.random() < 0.5:  # 50% chance for negative pairs
            random_idx = random.randint(0, len(sentences) - 1)
            if random_idx != i + 1:  # Make sure it's not the actual next sentence
                pairs.append({
                    'sentence_a': sentences[i],
                    'sentence_b': sentences[random_idx],
                    'is_next': 0
                })
    
    return pairs

def tokenize_bert_pair(pair, max_length=512):
    """Tokenize sentence pair for BERT pretraining"""
    sentence_a = pair['sentence_a']
    sentence_b = pair['sentence_b']
    is_next = pair['is_next']
    
    # Tokenize sentences
    tokens_a = enc.encode_ordinary(sentence_a)
    tokens_b = enc.encode_ordinary(sentence_b)
    
    # Truncate if too long
    max_tokens = max_length - 3  # Account for [CLS], [SEP], [SEP]
    if len(tokens_a) + len(tokens_b) > max_tokens:
        total_len = len(tokens_a) + len(tokens_b)
        tokens_a = tokens_a[:len(tokens_a) * max_tokens // total_len]
        tokens_b = tokens_b[:len(tokens_b) * max_tokens // total_len]
    
    # Build BERT input: [CLS] + sentence_a + [SEP] + sentence_b + [SEP]
    tokens = [CLS_TOKEN] + tokens_a + [SEP_TOKEN] + tokens_b + [SEP_TOKEN]
    
    # Create segment IDs (0 for sentence A, 1 for sentence B)
    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    
    # Pad to max_length
    while len(tokens) < max_length:
        tokens.append(PAD_TOKEN)
        segment_ids.append(0)
    
    # Truncate if somehow longer
    tokens = tokens[:max_length]
    segment_ids = segment_ids[:max_length]
    
    return {
        'tokens': np.array(tokens, dtype=np.uint16),
        'segment_ids': np.array(segment_ids, dtype=np.uint16),
        'is_next': is_next
    }

def process_doc_for_bert(doc):
    """Process document for BERT pretraining"""
    pairs = create_sentence_pairs(doc)
    bert_examples = []
    
    for pair in pairs:
        bert_example = tokenize_bert_pair(pair)
        bert_examples.append(bert_example)
    
    return bert_examples

def write_bert_datafile_to_gcs(filename, bert_data):
    """Write BERT data (tokens, segments, labels) to GCS bucket"""
    gcs_path = f"{gcs_prefix}{filename}.npz"
    
    # Create a bytes buffer and save data to it
    buffer = io.BytesIO()
    np.savez_compressed(buffer, 
                       input_ids=bert_data['input_ids'],
                       segment_ids=bert_data['segment_ids'], 
                       nsp_labels=bert_data['nsp_labels'])
    buffer.seek(0)
    
    # Upload to GCS
    blob = bucket.blob(gcs_path)
    blob.upload_from_file(buffer, content_type='application/octet-stream')
    
    print(f"Uploaded {gcs_path} to GCS bucket {gcs_bucket_name}")

if __name__ == '__main__':

    # Verify GCS bucket access
    try:
        bucket.reload()
        print(f"Successfully connected to GCS bucket: {gcs_bucket_name}")
    except Exception as e:
        print(f"Error accessing GCS bucket {gcs_bucket_name}: {e}")
        print("Make sure you have proper authentication and bucket access.")
        exit(1)

    # download the dataset
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train[:50%]") # 50% of data for testing
    
    # Process all documents for BERT and write output shards
    nprocs = max(1, os.cpu_count() - 2) # leave some spare CPU
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        examples_per_shard = 10000  # Number of BERT examples per shard
        
        # Buffers for current shard
        current_input_ids = []
        current_segment_ids = []
        current_nsp_labels = []
        
        progress_bar = None
        
        for bert_examples in pool.imap(process_doc_for_bert, fw, chunksize=16):
            for example in bert_examples:
                current_input_ids.append(example['tokens'])
                current_segment_ids.append(example['segment_ids'])
                current_nsp_labels.append(example['is_next'])
                
                # Check if shard is full
                if len(current_input_ids) >= examples_per_shard:
                    split = "val" if shard_index == 0 else "train"
                    filename = f"edufineweb_bert_{split}_{shard_index:06d}"
                    
                    # Prepare data for saving
                    bert_data = {
                        'input_ids': np.array(current_input_ids, dtype=np.uint16),
                        'segment_ids': np.array(current_segment_ids, dtype=np.uint16),
                        'nsp_labels': np.array(current_nsp_labels, dtype=np.uint8)
                    }
                    
                    write_bert_datafile_to_gcs(filename, bert_data)
                    shard_index += 1
                    
                    # Reset buffers
                    current_input_ids = []
                    current_segment_ids = []
                    current_nsp_labels = []
                    
                    if progress_bar is None:
                        progress_bar = tqdm(desc="Processing BERT examples")
                    progress_bar.update(examples_per_shard)

        # Write any remaining examples as the last shard
        if len(current_input_ids) > 0:
            split = "val" if shard_index == 0 else "train"
            filename = f"edufineweb_bert_{split}_{shard_index:06d}"
            
            bert_data = {
                'input_ids': np.array(current_input_ids, dtype=np.uint16),
                'segment_ids': np.array(current_segment_ids, dtype=np.uint16),
                'nsp_labels': np.array(current_nsp_labels, dtype=np.uint8)
            }
            
            write_bert_datafile_to_gcs(filename, bert_data)
            
        if progress_bar:
            progress_bar.close()