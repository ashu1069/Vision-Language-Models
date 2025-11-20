from PIL import Image
import torch
import fire
import time
from contextlib import contextmanager

from processing import PaliGemmaProcessor
from gemma import KVCache
from paligemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from utils import load_hf_model


def distribute_model_across_gpus(model, num_gpus):
    """
    Distribute model across multiple GPUs for model parallelism.
    Strategy:
    - Vision tower -> GPU 0
    - Multi-modal projector -> GPU 0
    - Language model layers -> Distributed across GPUs 0-N
    - LM head -> Last GPU
    
    Uses non_blocking transfers for faster GPU-to-GPU movement.
    """
    if num_gpus <= 1:
        return model
    
    print(f"\nDistributing model across {num_gpus} GPUs...")
    
    # Use non_blocking for faster transfers
    with torch.cuda.device(0):
        # Vision components on GPU 0
        model.vision_tower = model.vision_tower.to(f"cuda:0", non_blocking=True)
        model.multi_modal_projector = model.multi_modal_projector.to(f"cuda:0", non_blocking=True)
    
    # Get language model layers
    language_model = model.language_model.model
    num_layers = len(language_model.layers)
    layers_per_gpu = num_layers // num_gpus
    remainder = num_layers % num_gpus
    
    # Distribute layers across GPUs with non_blocking
    layer_idx = 0
    for gpu_id in range(num_gpus):
        # Calculate how many layers this GPU gets
        num_layers_this_gpu = layers_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = layer_idx + num_layers_this_gpu
        
        # Move layers to this GPU with non_blocking
        with torch.cuda.device(gpu_id):
            for i in range(layer_idx, end_idx):
                language_model.layers[i] = language_model.layers[i].to(f"cuda:{gpu_id}", non_blocking=True)
        
        print(f"  GPU {gpu_id}: Layers {layer_idx}-{end_idx-1} ({num_layers_this_gpu} layers)")
        layer_idx = end_idx
    
    # Embedding and norm on GPU 0
    with torch.cuda.device(0):
        language_model.embed_tokens = language_model.embed_tokens.to("cuda:0", non_blocking=True)
        language_model.norm = language_model.norm.to("cuda:0", non_blocking=True)
    
    # LM head on last GPU
    with torch.cuda.device(num_gpus - 1):
        model.language_model.lm_head = model.language_model.lm_head.to(f"cuda:{num_gpus-1}", non_blocking=True)
    
    # Synchronize all GPUs to ensure transfers complete
    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.synchronize()
    
    print("Model distribution complete.\n")
    return model


@contextmanager
def timer(description: str):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.4f}s")


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs

def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text = prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs 

def _sample_top_p(probs: torch.Tensor, p: float):
    # [b, vocab_size]
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim =-1)

    # Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking
    mask = probs_sum - probs_sort > p

    # Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0

    # Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    # Sample a token (its index) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples = 1)

    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: float,
    use_amp: bool = False,
    timing_stats: dict = None,
    num_gpus: int = 1,
):
    # Preprocessing timing
    preprocess_start = time.time()
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    preprocess_time = time.time() - preprocess_start
    
    if timing_stats is not None:
        timing_stats["preprocessing_time"] = preprocess_time

    kv_cache = KVCache()

    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    
    # Timing for generation
    generation_start = time.time()
    first_token_time = None
    token_times = []
    
    for token_idx in range(max_tokens_to_generate):
        token_start = time.time()
        
        # Get the model outputs with mixed precision if enabled
        if use_amp and device == "cuda":
            with torch.autocast(device_type=device, dtype=torch.float16):
                outputs = model(
                    input_ids = input_ids,
                    pixel_values = pixel_values,
                    attention_mask = attention_mask,
                    kv_cache = kv_cache,
                )
        else:
            outputs = model(
                input_ids = input_ids,
                pixel_values = pixel_values,
                attention_mask = attention_mask,
                kv_cache = kv_cache,
            )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        
        # Move logits to GPU 0 for processing (if using multi-GPU)
        if device == "cuda" and next_token_logits.device.index != 0:
            next_token_logits = next_token_logits.to("cuda:0")
        
        # Clear outputs to free memory (keep only what we need)
        del outputs

        # Sample the next token
        if do_sample:
            # Apply temperature
            next_token_logits = torch.softmax(next_token_logits / temperature, dim = -1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim = -1, keepdim=True) # greedy strategy
        assert next_token.size() == (1, 1)
        
        # Clear logits after sampling
        del next_token_logits

        next_token = next_token.squeeze(0) # Remove batch dimensions
        generated_tokens.append(next_token)
        
        token_time = time.time() - token_start
        token_times.append(token_time)
        
        # Track first token latency
        if token_idx == 0:
            first_token_time = token_time
            if timing_stats is not None:
                timing_stats["first_token_latency"] = first_token_time
            # Clear cache after first token on all GPUs
            if device == "cuda":
                for i in range(num_gpus):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()

        # Stop if the stop token has been generated
        if next_token.item() == stop_token:
            break

        # Append the next token to the input
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device, dtype=attention_mask.dtype)], dim = -1
        )
        
        # Periodic cache clearing for long generations on all GPUs
        if device == "cuda" and token_idx > 0 and token_idx % 50 == 0:
            for i in range(num_gpus):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

    generation_time = time.time() - generation_start
    
    if timing_stats is not None:
        timing_stats["generation_time"] = generation_time
        timing_stats["num_tokens"] = len(generated_tokens)
        timing_stats["avg_token_time"] = sum(token_times) / len(token_times) if token_times else 0
        timing_stats["tokens_per_second"] = len(generated_tokens) / generation_time if generation_time > 0 else 0

    generated_tokens = torch.cat(generated_tokens, dim = -1)

    # Decode the generated tokens
    decode_start = time.time()
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    decode_time = time.time() - decode_start
    
    if timing_stats is not None:
        timing_stats["decode_time"] = decode_time

    print(prompt + decoded)




def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.9,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
    use_amp: bool = True,
    use_torch_compile: bool = False,
    torch_compile_mode: str = "reduce-overhead",
    enable_cudnn_benchmark: bool = True,
    num_gpus: int = 1,
):
    """
    Run inference with performance optimizations and timing.
    
    Args:
        model_path: Path to the model directory
        prompt: Text prompt for the model
        image_file_path: Path to the input image
        max_tokens_to_generate: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling (True) or greedy decoding (False)
        only_cpu: Force CPU usage
        use_amp: Use Automatic Mixed Precision (FP16) for faster inference on GPU
        use_torch_compile: Use torch.compile() for model optimization (PyTorch 2.0+)
        torch_compile_mode: Compilation mode ('reduce-overhead', 'max-autotune', 'default')
        enable_cudnn_benchmark: Enable cuDNN benchmark mode for faster convolutions
        num_gpus: Number of GPUs to use for model parallelism (default: 1)
    """
    
    # Initialize timing stats
    timing_stats = {}
    total_start = time.time()

    device = 'cpu'
    available_gpus = 0

    if not only_cpu:
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            device = "cuda"
            # CUDA optimizations
            if enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            # Limit num_gpus to available GPUs
            num_gpus = min(num_gpus, available_gpus)
            
            print(f"Available GPUs: {available_gpus}")
            for i in range(available_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)")
        elif torch.backend.mps.is_available():
            device = "mps"
            num_gpus = 1

    print(f"\n{'='*60}")
    print(f"Device: {device}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Mixed Precision (AMP): {use_amp and device == 'cuda'}")
    print(f"Torch Compile: {use_torch_compile}")
    print(f"cuDNN Benchmark: {enable_cudnn_benchmark and device == 'cuda'}")
    print(f"{'='*60}\n")

    # Clear CUDA cache before loading model
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Model loading timing
    print("Loading Model...")
    load_start = time.time()
    
    # Load model with optimizations
    model, tokenizer = load_hf_model(
        model_path, 
        "cuda:0" if device == "cuda" else device,
        num_gpus=num_gpus if device == "cuda" else 1
    )
    model = model.eval()
    
    # Distribute model across GPUs if multiple GPUs requested
    if device == "cuda" and num_gpus > 1:
        dist_start = time.time()
        model = distribute_model_across_gpus(model, num_gpus)
        dist_time = time.time() - dist_start
        print(f"Model distribution took {dist_time:.4f}s")
    else:
        model = model.to(device)
    
    load_time = time.time() - load_start
    timing_stats["model_load_time"] = load_time
    print(f"Model loaded in {load_time:.4f}s")
    
    # Clear cache after model loading on all GPUs
    if device == "cuda":
        for i in range(num_gpus):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

    # Create processor (needed for warmup)
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    # Apply torch.compile if requested
    if use_torch_compile and hasattr(torch, 'compile'):
        print(f"Compiling model with mode: {torch_compile_mode}...")
        compile_start = time.time()
        model = torch.compile(model, mode=torch_compile_mode)
        compile_time = time.time() - compile_start
        timing_stats["compile_time"] = compile_time
        print(f"Model compiled in {compile_time:.4f}s")
        # Warmup run for compilation
        print("Running warmup for compilation...")
        warmup_start = time.time()
        with torch.no_grad():
            dummy_inputs = get_model_inputs(
                processor,
                "test", image_file_path, device
            )
            _ = model(
                input_ids=dummy_inputs["input_ids"],
                pixel_values=dummy_inputs["pixel_values"],
                attention_mask=dummy_inputs["attention_mask"],
                kv_cache=KVCache(),
            )
            # Clear dummy inputs
            del dummy_inputs
        warmup_time = time.time() - warmup_start
        timing_stats["warmup_time"] = warmup_time
        print(f"Warmup completed in {warmup_time:.4f}s")
        
        # Clear cache after warmup on all GPUs
        if device == "cuda":
            for i in range(num_gpus):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

    print("\nRunning Inference...")
    print("-" * 60)

    with torch.no_grad():
        test_inference(
            model,
            processor, 
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
            use_amp=use_amp and device == "cuda",
            timing_stats=timing_stats,
            num_gpus=num_gpus,
        )

    total_time = time.time() - total_start
    timing_stats["total_time"] = total_time

    # Print performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Model Loading:        {timing_stats.get('model_load_time', 0):.4f}s")
    if 'compile_time' in timing_stats:
        print(f"Compilation:          {timing_stats['compile_time']:.4f}s")
    print(f"Preprocessing:        {timing_stats.get('preprocessing_time', 0):.4f}s")
    print(f"First Token Latency:  {timing_stats.get('first_token_latency', 0):.4f}s")
    print(f"Generation Time:      {timing_stats.get('generation_time', 0):.4f}s")
    print(f"  - Tokens Generated:  {timing_stats.get('num_tokens', 0)}")
    print(f"  - Avg Time/Token:    {timing_stats.get('avg_token_time', 0)*1000:.2f}ms")
    print(f"  - Tokens/Second:     {timing_stats.get('tokens_per_second', 0):.2f}")
    print(f"Decoding:             {timing_stats.get('decode_time', 0):.4f}s")
    print(f"Total Time:           {total_time:.4f}s")
    
    # GPU memory info if available
    if device == "cuda":
        print(f"\nGPU Memory Usage:")
        for i in range(num_gpus):
            memory_allocated = torch.cuda.memory_allocated(i) / 1e9
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}:")
            print(f"    - Allocated:        {memory_allocated:.2f} GB")
            print(f"    - Reserved:         {memory_reserved:.2f} GB")
        
        # Final cache clear on all GPUs
        for i in range(num_gpus):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
    
    print("="*60)

if __name__ == "__main__":
    fire.Fire(main)