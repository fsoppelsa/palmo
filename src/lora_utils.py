import torch
from src.transformer import MultiHeadAttention, PalmoModel
from src.finetune import LinearWithLoRA


def load_lora_adapter(base_model_path, lora_adapter_path, config):
    """
    Loads a base model and applies LoRA adapters.
    
    Args:
        base_model_path: Path to the pre-trained model checkpoint
        lora_adapter_path: Path to the LoRA adapters
        config: Model configuration
        
    Returns:
        Model with LoRA adapters applied
    """
    model = PalmoModel(config)
    checkpoint = torch.load(base_model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    lora_checkpoint = torch.load(lora_adapter_path, map_location='cpu', weights_only=False)
    lora_config = lora_checkpoint.get('lora_config', {'rank': 8, 'alpha': 16})
    
    from src.finetune import replace_linear_with_lora
    model = replace_linear_with_lora(model, rank=lora_config['rank'], alpha=lora_config['alpha'])
    
    if 'lora_state_dict' in lora_checkpoint:
        lora_state_dict = lora_checkpoint['lora_state_dict']
    elif 'model_state_dict' in lora_checkpoint:
        lora_state_dict = lora_checkpoint['model_state_dict']
    else:
        lora_state_dict = lora_checkpoint
    
    model.load_state_dict(lora_state_dict, strict=False)
    
    return model


def merge_lora_weights(model):
    """
    Merge LoRA adapter weights into base model.
    Converts a model with LoRA into a standard model without adapters.
    Useful for deployment when the adapters no longer need to be modified.
    
    Args:
        model: Model with LoRA adapters
        
    Returns:
        Model with merged weights (no LoRA layers)
    """
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            if isinstance(module.W_query, LinearWithLoRA):
                # lora_A: (in_features, rank), lora_B: (rank, out_features)
                # A @ B = (in_features, out_features), need to transpose for weight matrix
                lora_weight = (module.W_query.lora.lora_A @ module.W_query.lora.lora_B) * module.W_query.lora.scaling
                merged_weight = module.W_query.linear.weight.data + lora_weight.T
                
                # Replace with a standard nn.Linear layer.
                new_linear = torch.nn.Linear(
                    module.W_query.linear.in_features,
                    module.W_query.linear.out_features,
                    bias=(module.W_query.linear.bias is not None)
                )
                new_linear.weight.data = merged_weight
                if module.W_query.linear.bias is not None:
                    new_linear.bias.data = module.W_query.linear.bias.data
                module.W_query = new_linear
            
            if isinstance(module.W_value, LinearWithLoRA):
                lora_weight = (module.W_value.lora.lora_A @ module.W_value.lora.lora_B) * module.W_value.lora.scaling
                merged_weight = module.W_value.linear.weight.data + lora_weight.T
                
                new_linear = torch.nn.Linear(
                    module.W_value.linear.in_features,
                    module.W_value.linear.out_features,
                    bias=(module.W_value.linear.bias is not None)
                )
                new_linear.weight.data = merged_weight
                if module.W_value.linear.bias is not None:
                    new_linear.bias.data = module.W_value.linear.bias.data
                module.W_value = new_linear
    
    return model


def count_lora_parameters(model):
    """
    Counts the LoRA parameters in the model.
    
    Returns:
        Dictionary with parameter statistics
    """
    lora_params = sum(
        p.numel() for name, p in model.named_parameters()
        if 'lora_A' in name or 'lora_B' in name
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'lora_params': lora_params,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'lora_percentage': 100 * lora_params / total_params if total_params > 0 else 0,
        'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0
    }
