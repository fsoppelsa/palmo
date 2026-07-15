"""
Module for calculating perplexity on pre-trained and fine-tuned models.
"""

import torch
import math
import os
from src.pretraining import load_dataset
from src.finetune import load_instruction_data
import random


def calculate_perplexity(model, dataloader, device, max_batches=30):
    """
    Calculates perplexity on a standard dataloader (raw corpus).
    
    Args:
        model: The PyTorch model
        dataloader: DataLoader with (input, target) batches
        device: torch.device on which to perform the calculation
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        tuple: (perplexity, average_loss)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_batch, target_batch in dataloader:
            if num_batches >= max_batches:
                break
            
            logits = model(input_batch.to(device))
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_batch.to(device).view(-1)
            )
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return math.exp(avg_loss), avg_loss


def calculate_instruction_perplexity(model, tokenizer, instruction_data, device, max_samples=50):
    """
    Calculates perplexity on instruction-following (Q&A) data.
    Evaluates loss only on the response, not the question.
    
    Args:
        model: The PyTorch model
        tokenizer: Tokenizer for encoding text
        instruction_data: List of dictionaries with 'instruction' and 'output'
        device: torch.device on which to perform the calculation
        max_samples: Maximum number of examples to evaluate
        
    Returns:
        tuple: (perplexity, average_loss)
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for example in instruction_data[:max_samples]:
            # Complete format: instruction + response.
            full_text = example['instruction'] + " " + example['output']
            instruction_text = example['instruction'] + " "
            
            full_ids = tokenizer.encode(full_text)
            instruction_ids = tokenizer.encode(instruction_text)
            
            response_length = len(full_ids) - len(instruction_ids)
            
            if response_length <= 0:
                continue
            
            input_tensor = torch.tensor([full_ids[:-1]], dtype=torch.long).to(device)
            target_tensor = torch.tensor([full_ids[1:]], dtype=torch.long).to(device)
            
            # Forward pass
            logits = model(input_tensor)
            
            instruction_offset = len(instruction_ids) - 1
            if instruction_offset >= logits.size(1):
                continue
                
            response_logits = logits[:, instruction_offset:, :]
            response_targets = target_tensor[:, instruction_offset:]
            
            loss = torch.nn.functional.cross_entropy(
                response_logits.reshape(-1, response_logits.size(-1)),
                response_targets.reshape(-1)
            )
            
            total_loss += loss.item()
            num_samples += 1
    
    if num_samples == 0:
        return float('inf'), float('inf')
    
    avg_loss = total_loss / num_samples
    return math.exp(avg_loss), avg_loss


def evaluate_model_perplexity(model_name, model_path, model_config, tokenizer, 
                              token_ids, device, text_name, is_finetuned=False):
    """
    Evaluates a model's perplexity, automatically selecting the appropriate method.
    
    Args:
        model_name: Descriptive model name
        model_path: Path to the model checkpoint
        model_config: Model configuration
        tokenizer: Tokenizer for encoding/decoding
        token_ids: Token IDs of the corpus (for pre-trained models)
        device: torch.device for execution
        text_name: Name of the corpus/dataset (e.g., "bohemia", "sherlock")
        is_finetuned: If True, uses instruction-following evaluation
        
    Returns:
        dict: {'perplexity': float, 'loss': float} or None on error
    """
    from src.transformer import PalmoModel
    
    if not os.path.exists(model_path):
        return None
    
    try:
        checkpoint_data = torch.load(model_path, map_location='cpu', weights_only=False)
        loaded_config = checkpoint_data.get('config', model_config)
        
        eval_model = PalmoModel(loaded_config)
        if 'model_state_dict' in checkpoint_data:
            eval_model.load_state_dict(checkpoint_data['model_state_dict'])
        elif 'model' in checkpoint_data:
            eval_model = checkpoint_data['model']
        else:
            return None
        
        eval_model = eval_model.to(device)
        eval_model.eval()
        
        # Select the evaluation method.
        if is_finetuned:
            instruction_file = f"./data/{text_name}-tune.json"
            all_data = load_instruction_data(instruction_file)
            
            random.seed(42)
            shuffled_data = all_data.copy()
            random.shuffle(shuffled_data)
            val_size = int(len(shuffled_data) * 0.2)
            val_data = shuffled_data[:val_size]
            
            ppl, loss = calculate_instruction_perplexity(
                eval_model, tokenizer, val_data, device, max_samples=50
            )
        else:
            _, val_loader = load_dataset(token_ids, loaded_config['context_length'], batch_size=64)
            ppl, loss = calculate_perplexity(eval_model, val_loader, device, max_batches=30)
        
        result = {'perplexity': ppl, 'loss': loss}
        
    except Exception as e:
        # Catch any error (OOM, qengine, and so on).
        result = None
        raise  # Re-raise for handling by the caller.
        
    finally:
        del eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    return result
