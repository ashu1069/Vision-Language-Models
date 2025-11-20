#!/bin/bash

# Set PyTorch CUDA memory allocation config to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="/home/stu12/s11/ak1825/Projects/Vision-Language-Models/PaliGemma/models/paligemma-3b-pt-224"
PROMPT="Describe this historical photograph. Identify the famous scientists and explain the context."
IMAGE_FILE_PATH="/home/stu12/s11/ak1825/Projects/Vision-Language-Models/solvary_conference.jpg"
MAX_TOKENS_TO_GENERATE=500
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

# Optimization flags
USE_AMP="True"                    # Use mixed precision (FP16) for faster inference
USE_TORCH_COMPILE="False"         # Use torch.compile() - first run will be slower due to compilation
TORCH_COMPILE_MODE="reduce-overhead"  # Compilation mode: 'reduce-overhead', 'max-autotune', or 'default'
ENABLE_CUDNN_BENCHMARK="True"     # Enable cuDNN benchmark for faster convolutions
NUM_GPUS=4                         # Number of GPUs to use for model parallelism

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path  "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate  $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \
    --use_amp $USE_AMP \
    --use_torch_compile $USE_TORCH_COMPILE \
    --torch_compile_mode $TORCH_COMPILE_MODE \
    --enable_cudnn_benchmark $ENABLE_CUDNN_BENCHMARK \
    --num_gpus $NUM_GPUS