# Vision Language Models

Implementation of vision-language models including PaliGemma.

## PaliGemma

PaliGemma is a vision-language model for image understanding and text generation.

### Setup

1. Download the model:
   ```
   python PaliGemma/download_model.py
   ```

2. Install dependencies (PyTorch, transformers, PIL, etc.)

### Usage

Run inference:
```
python PaliGemma/inference.py --image <path> --prompt <text>
```

Or use the shell script:
```
bash PaliGemma/inference.sh
```

Fine-tune the model:
```
python PaliGemma/finetune.py --train_data <json_file>
```

Or use the shell script:
```
bash PaliGemma/finetune.sh
```

### Files

- `inference.py` - Model inference script
- `finetune.py` - Fine-tuning script
- `download_model.py` - Model download utility
- `paligemma.py` - PaliGemma model implementation
- `gemma.py` - Gemma language model components
- `processing.py` - Image and text processing utilities
