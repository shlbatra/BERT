import torch
import torch.distributed as dist
import torch.nn.functional as F
import logging
from training.bert_inputs import create_mlm_inputs_and_labels
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
import tiktoken

class Evaluator:
    def __init__(self, model, optimizer, config, ddp_config, logger, vocab_size):
        """Initialize trainer with all components"""
        self.model = model
        self.model.eval()
        self.optimizer = optimizer
        self.config = config
        self.ddp_config = ddp_config
        self.logger = logger
        self.vocab_size = vocab_size

        self.logger_instance = logging.getLogger(__name__)
        
        assert self.config.total_batch_size % (self.config.B * self.config.T * self.ddp_config['ddp_world_size']) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
        grad_accum_steps = self.config.total_batch_size // (self.config.B * self.config.T * self.ddp_config['ddp_world_size'])
        if self.ddp_config['master_process']:
            self.logger_instance.info(f"total desired batch size: {self.config.total_batch_size}")
            self.logger_instance.info(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    def evaluate(self, val_loader, step, last_step):
        """Execute evaluate step"""
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):

                input_ids, segment_ids, nsp_labels = val_loader.next_batch()
                input_ids, segment_ids, nsp_labels = input_ids.to(self.ddp_config['device']), segment_ids.to(self.ddp_config['device']), nsp_labels.to(self.ddp_config['device'])
                masked_input_ids, mlm_labels, attention_mask = create_mlm_inputs_and_labels(input_ids)

                x, y = val_loader.next_batch()
                x, y = x.to(self.ddp_config['device']), y.to(self.ddp_config['device'])
                with torch.autocast(device_type=self.ddp_config['device'], dtype=torch.bfloat16):
                    outputs = self.model(masked_input_ids, segment_ids, attention_mask=attention_mask)
                    nsp_scores = outputs['nsp_logits']
                    mlm_scores = outputs['mlm_logits']
                    mlm_loss = F.cross_entropy(mlm_scores.view(-1, self.vocab_size), mlm_labels.view(-1), ignore_index=-100)
                    nsp_loss = F.cross_entropy(nsp_scores, nsp_labels)
                    loss = mlm_loss + nsp_loss
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()

        if self.ddp_config['ddp']:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if self.ddp_config['master_process']:
            self.logger_instance.info(f"validation loss: {val_loss_accum.item():.4f}")
            return val_loss_accum

    def evaluate_ag_news_embeddings(self, step, max_samples=1000):
        """
        Evaluate BERT embeddings on AG News classification task
        Extract [CLS] embeddings and train logistic regression classifier
        """
        if not self.ddp_config['master_process']:
            return None
            
        try:
            # Load AG News dataset
            dataset = load_dataset("ag_news")
            train_data = dataset['train']
            test_data = dataset['test']
            
            # Limit samples for efficiency
            train_texts = train_data['text'][:max_samples]
            train_labels = train_data['label'][:max_samples]
            test_texts = test_data['text'][:max_samples//4]  # Smaller test set
            test_labels = test_data['label'][:max_samples//4]
            
            # Extract embeddings
            train_embeddings = self._extract_embeddings(train_texts)
            test_embeddings = self._extract_embeddings(test_texts)
            
            # Train logistic regression
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(train_embeddings, train_labels)
            
            # Evaluate
            train_pred = clf.predict(train_embeddings)
            test_pred = clf.predict(test_embeddings)
            
            train_acc = accuracy_score(train_labels, train_pred)
            test_acc = accuracy_score(test_labels, test_pred)
            
            # Log results
            self.logger_instance.info(f"Step {step} AG News Embedding Evaluation:")
            self.logger_instance.info(f"  Train Accuracy: {train_acc:.4f}")
            self.logger_instance.info(f"  Test Accuracy: {test_acc:.4f}")
            
            # Write to log file
            with open(self.logger, "a") as f:
                f.write(f"{step} ag_news_train_acc {train_acc:.4f}\n")
                f.write(f"{step} ag_news_test_acc {test_acc:.4f}\n")
            
            return test_acc
            
        except Exception as e:
            self.logger_instance.error(f"AG News evaluation failed: {str(e)}")
            return None
    
    def _extract_embeddings(self, texts, max_length=512, batch_size=32):
        """
        Extract text embeddings using BERT encoder with optimized batching
        
        Encoder Definition:
        1. Tokenize text input with special tokens [CLS] and [SEP]
        2. Convert to input_ids, attention_mask, and token_type_ids
        3. Pass through BERT encoder layers
        4. Extract [CLS] token representation (pooled output)
        5. Apply pooler projection and activation
        6. Return final text embedding vector
        """

            
        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            # Process texts in batches for efficiency
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._encode_text_batch(batch_texts, max_length)
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _encode_text_batch(self, texts, max_length):
        """
        Encode a batch of texts through the BERT model using tiktoken
        
        Encoder Steps:
        1. Tokenization with tiktoken encoder
        2. Add special tokens [CLS] at start, [SEP] at end
        3. Create attention masks for real vs padding tokens
        4. Forward pass through BERT encoder
        5. Extract and pool [CLS] token representation
        """
        # Initialize tiktoken encoder
        tokenizer = tiktoken.get_encoding("gpt2")
        
        # Define BERT special tokens
        CLS_TOKEN = 50256  # [CLS] token
        SEP_TOKEN = 50257  # [SEP] token  
        MASK_TOKEN = 50258  # [MASK] token
        PAD_TOKEN = 50259  # [PAD] token
        
        # Step 1: Tokenize batch with proper BERT formatting
        tokenized = []
        for text in texts:
            # Encode text using tiktoken (truncate to leave space for special tokens)
            tokens = tokenizer.encode(text)
            
            # Truncate if too long (reserve 2 tokens for [CLS] and [SEP])
            if len(tokens) > max_length - 2:
                tokens = tokens[:max_length - 2]
            
            # Add BERT special tokens: [CLS] + text + [SEP]
            bert_tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
            
            # Pad to max_length with [PAD] tokens
            if len(bert_tokens) < max_length:
                bert_tokens = bert_tokens + [PAD_TOKEN] * (max_length - len(bert_tokens))
            
            tokenized.append(bert_tokens)
        
        # Step 2: Convert to tensors
        input_ids = torch.tensor(tokenized, dtype=torch.long).to(self.ddp_config['device'])
        
        # Step 3: Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != PAD_TOKEN).to(self.ddp_config['device'])
        
        # Step 4: Create token_type_ids (all zeros for single sentence)
        token_type_ids = torch.zeros_like(input_ids).to(self.ddp_config['device'])
        
        # Step 5: Forward pass through BERT encoder
        outputs = self.model(input_ids, token_type_ids, attention_mask)
        
        # Step 6: Extract embeddings using encoder output
        embeddings = self._pool_embeddings(outputs, attention_mask)
        
        return embeddings.cpu().numpy()
    
    def _pool_embeddings(self, model_outputs, attention_mask):
        """
        Extract pooled sentence embedding from BERT model output
        
        Our BERT model returns 'pooler_output' which is:
        1. [CLS] token from last encoder layer
        2. Linear projection (768 -> 768) 
        3. Tanh activation
        4. Ready-to-use sentence embedding
        """
        # Use BERT's pooler output directly - this is the sentence embedding
        return model_outputs['pooler_output']
