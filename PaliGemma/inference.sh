#!/bin/bash

MODEL_PATH="/home/stu12/s11/ak1825/Projects/Vision-Language-Models/PaliGemma/models/paligemma-3b-pt-224"
PROMPT="Describe this historical photograph. Identify the famous scientists and explain the context."
IMAGE_FILE_PATH="/home/stu12/s11/ak1825/Projects/Vision-Language-Models/solvary_conference.jpg"
MAX_TOKENS_TO_GENERATE=500
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path  "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate  $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \