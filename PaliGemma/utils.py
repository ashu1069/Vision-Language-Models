from paligemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

def _load_safetensors_file(safetensors_file: str, device: str):
    """Load a single safetensors file directly to target device."""
    tensors = {}
    with safe_open(safetensors_file, framework="pt", device=device) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def load_hf_model(model_path: str, device: str, num_gpus: int = 1) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Load model with optimizations:
    - Parallel loading of safetensors files
    - Direct device loading (skip CPU intermediate)
    - Optimized state dict loading
    """
    # Load the tokenizer (can be done in parallel with model loading)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")

    # Load model config first (needed to create model)
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model first (empty, will be populated)
    # For multi-GPU, we'll load directly to target devices
    if num_gpus > 1:
        # Create model on CPU first, will distribute later
        model = PaliGemmaForConditionalGeneration(config)
    else:
        model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load tensors in parallel (faster for multiple files)
    print("Loading model weights...")
    tensors = {}
    
    # For single GPU, load directly to GPU. For multi-GPU, load to CPU then distribute
    target_device = device if num_gpus == 1 else "cpu"
    
    if len(safetensors_files) > 1:
        # Parallel loading for multiple files (up to 4 workers for I/O bound operations)
        with ThreadPoolExecutor(max_workers=min(4, len(safetensors_files))) as executor:
            futures = {executor.submit(_load_safetensors_file, f, target_device): f 
                      for f in safetensors_files}
            
            for future in as_completed(futures):
                file_tensors = future.result()
                tensors.update(file_tensors)
    else:
        # Single file - direct loading
        tensors = _load_safetensors_file(safetensors_files[0], target_device)

    # Load state dict with non_blocking for faster transfer
    print("Loading state dict...")
    if num_gpus == 1:
        # Single GPU: load directly
        model.load_state_dict(tensors, strict=False)
    else:
        # Multi-GPU: load to CPU first, then distribute
        # This is faster than loading to each GPU separately
        model.load_state_dict(tensors, strict=False)
        # Distribution will happen after this function returns

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)