"""

This module handles preprocessing of images and text for the PaliGemma model.

Image Processing:
- Resize to model input size (e.g., 224x224)
- Normalize pixel values (0-255 -> 0-1 -> normalized)
- Convert to tensor format expected by model

Text Processing:
- Add image tokens to prompt (mark positions where image features will be inserted)
- Tokenize text
- Handle special tokens (BOS, EOS, image tokens)

Key Design:
- Image tokens are added BEFORE the text prompt
- Format: [image_token * n_patches] + [BOS] + [text_tokens]
- This tells the model where to insert image features in the sequence
"""

from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

# ImageNet normalization constants
# These are standard values used for vision models pre-trained on ImageNet
# Mean: centers pixel values around 0
# Std: scales pixel values to unit variance
# Note: [0.5, 0.5, 0.5] means pixels are in [0, 1] range (after rescale_factor=1/255)
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:

    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(
    image: np.ndarray, 
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)

    image = (image - mean) / std
    return image

def process_image(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:

    height, width = size[0], size[1]

    # Step 1: Resize all images to target size
    images = [
        resize(image=image, size=(height, width), resample = resample) for image in images
    ]

    # Step 2: Convert PIL Images to numpy arrays
    # PIL Image -> numpy array (H, W, C) format
    images = [np.array(image) for image in images]

    # Step 3: Rescale pixel values from [0, 255] to [0, 1]
    # This converts uint8 to float32 in [0, 1] range
    images = [rescale(image, scale = rescale_factor) for image in images]

    # Step 4: Normalize to have mean 0 and std 1
    # Centers and scales pixel values for model input
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]

    # Step 5: Convert from HWC to CHW format
    # PIL/numpy: [Height, Width, Channels]
    # PyTorch: [Channels, Height, Width]
    # transpose(2, 0, 1): moves channel dimension from last to first
    images = [image.transpose(2, 0, 1) for image in images]

    return images

def add_image_tokens_to_prompt(
    prefix_prompt, bos_token, image_seq_len, image_token
):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_image_tokens  # Number of image patches (e.g., 196)
        self.image_size = image_size  # Input image size (e.g., 224)

        # Add image token to tokenizer vocabulary
        # This allows the tokenizer to recognize <image> as a special token
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        
        # Add location tokens for object detection
        # Format: <loc0000>, <loc0001>, ..., <loc1023>
        # Used to specify bounding box coordinates
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]
        
        # Add segmentation tokens for object segmentation
        # Format: <seg000>, <seg001>, ..., <seg127>
        # Used for pixel-level segmentation masks
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]

        tokenizer.add_tokens(EXTRA_TOKENS)
        
        # Get token ID for image token (needed for merging image features)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # Disable automatic BOS/EOS token addition
        # We add BOS token manually in the prompt formatting
        # This gives us more control over the sequence format
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = 'longest',
        truncation: bool = True,
    ) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # Step 1: Preprocess images
        # Resize to model input size, normalize, convert to CHW format
        pixel_values = process_image(
            images,
            size = (self.image_size, self.image_size),  # e.g., 224x224
            resample = Image.Resampling.BICUBIC,  # High-quality resampling
            rescale_factor = 1/255.0,  # Convert [0, 255] to [0, 1]
            image_mean = IMAGENET_STANDARD_MEAN,  # Normalization mean
            image_std = IMAGENET_STANDARD_STD,  # Normalization std
        )

        # Step 2: Stack images into batch
        # Convert list of arrays to single array
        # [C, H, W] * batch_size -> [B, C, H, W]
        pixel_values = np.stack(pixel_values, axis=0)

        # Step 3: Convert to PyTorch tensor
        # numpy array -> torch.Tensor
        pixel_values = torch.Tensor(pixel_values)

        # Step 4: Add image tokens to text prompts
        # Format: [image_token * n_patches] + [BOS] + [text]
        # Example: "<image><image>...<image><BOS>What is in this image?"
        # The image tokens mark positions where image features will be inserted
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_length,  # Number of image patches
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Step 5: Tokenize text
        # Converts text strings to token IDs
        # Returns input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors = "pt",  # Return PyTorch tensors
            padding = padding,  # Padding strategy (though model expects no padding)
            truncation = truncation,  # Truncate if sequence too long
        )

        # Combine image and text inputs
        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data
