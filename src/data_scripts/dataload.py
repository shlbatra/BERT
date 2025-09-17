import tiktoken
import numpy as np
import torch
import torch.nn as nn
import os
import logging
from google.cloud import storage
import io


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, master_process=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        self.logger_instance = logging.getLogger(__name__)

        # GCS configuration
        gcs_bucket_name = "bert-training-data-sahil"
        gcs_prefix = "edu_fineweb10B/"
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)
        
        # List all blobs (files) in the GCS bucket with the prefix
        blobs = bucket.list_blobs(prefix=gcs_prefix)
        
        # Filter shards based on split and collect GCS paths
        shard_blobs = []
        for blob in blobs:
            if blob.name.endswith('.npz') and split in blob.name:
                shard_blobs.append(blob)
        
        # Sort by filename for consistent ordering
        shard_blobs = sorted(shard_blobs, key=lambda x: x.name)
        
        # Store GCS blob references instead of local paths
        self.shards = shard_blobs
        assert len(shard_blobs) > 0, f"no shards found for split {split}"
        
        if master_process:
            self.logger_instance.info(f"found {len(shard_blobs)} shards for split {split}")
            # Optional: print first few shard names for verification
            for i, blob in enumerate(shard_blobs[:3]):
                self.logger_instance.info(f"  shard {i}: {blob.name}")
        
        self.reset()


    def load_bert_data(self, shard_blob):
        """Load BERT data (input_ids, segment_ids, nsp_labels) from GCS"""
        # Download blob content to memory
        blob_data = shard_blob.download_as_bytes()
        
        # Load npz file from bytes
        buffer = io.BytesIO(blob_data)
        data = np.load(buffer)
        
        # Convert to torch tensors
        input_ids = torch.tensor(data['input_ids'].astype(np.int32), dtype=torch.long)
        segment_ids = torch.tensor(data['segment_ids'].astype(np.int32), dtype=torch.long)
        nsp_labels = torch.tensor(data['nsp_labels'].astype(np.int32), dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'segment_ids': segment_ids, 
            'nsp_labels': nsp_labels
        }

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.bert_data = self.load_bert_data(self.shards[self.current_shard])
        self.current_position = self.B * self.process_rank

    def next_batch(self):
        B = self.B
        start_idx = self.current_position
        end_idx = start_idx + B
        
        # Get batch data
        input_ids = self.bert_data['input_ids'][start_idx:end_idx]
        segment_ids = self.bert_data['segment_ids'][start_idx:end_idx] 
        nsp_labels = self.bert_data['nsp_labels'][start_idx:end_idx]
        
        # advance the position
        self.current_position += B * self.num_processes
        
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position >= len(self.bert_data['input_ids']):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.bert_data = self.load_bert_data(self.shards[self.current_shard])
            self.current_position = B * self.process_rank
            
        return input_ids, segment_ids, nsp_labels