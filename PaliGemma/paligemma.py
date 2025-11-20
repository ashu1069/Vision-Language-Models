"""
PaliGemma: Vision-Language Model Implementation

This module implements the PaliGemma architecture, which combines:
1. SigLIP Vision Encoder: Processes images into patch embeddings
2. Multi-modal Projector: Aligns vision and language embedding spaces
3. Gemma Language Model: Generates text conditioned on image features
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel
from gemma import GemmaConfig, GemmaForCausalLM, KVCache


class PaliGemmaConfig():

    def __init__(
        self,
        vision_config = None,
        text_config = None,
        ignore_index = -100,
        image_token_index = 256000,
        vocab_size = 257152,
        projection_dim = 2048,
        hidden_size = 2048,
        pad_token_id = None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index  # Used in loss computation to ignore certain tokens
        self.image_token_index = image_token_index  # Special token ID marking image positions in sequence
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim  # Output dimension of multi-modal projector
        self.hidden_size = hidden_size  # Language model hidden dimension
        self.vision_config = vision_config
        self.is_encoder_decoder = False  # PaliGemma is decoder-only (autoregressive generation)
        self.pad_token_id = pad_token_id

        # Initialize vision config (SigLIP ViT configuration)
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        # Ensure num_attention_layers is set (defaults to num_hidden_layers if not provided)
        # This is needed for Gemma's architecture which may have different attention layer counts
        if 'num_attention_layers' not in text_config:
            text_config['num_attention_layers'] = text_config.get('num_hidden_layers')
        
        # Initialize text config (Gemma language model configuration)
        self.text_config = GemmaConfig(**text_config, pad_token_id = pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        # Calculate number of image tokens from patch configuration
        # For 224x224 image with 16x16 patches: (224/16)^2 = 14^2 = 196 tokens
        # This tells the language model how many image tokens to expect
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class PaliGemmaMultiModalProjector(nn.Module):
    """
    Multi-modal Projector: Aligns Vision and Language Embedding Spaces
    
    Purpose:
    - Vision encoder (SigLIP) outputs features in its own embedding space (e.g., 768-dim)
    - Language model (Gemma) expects features in its embedding space (e.g., 2048-dim)
    - This linear layer projects vision features to match language model dimensions
    
    Why a simple linear layer?
    - Vision and language models are pre-trained separately
    - A learned linear projection is sufficient to align the spaces (proven effective in CLIP, LLaVA, etc.)
    - More complex projections (MLPs) can overfit and don't significantly improve performance
    - Simpler = faster inference and easier to train
    
    Design choice: bias=True allows the projection to shift the feature space, not just scale/rotate
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        # Linear projection: vision_hidden_size -> projection_dim (language model hidden_size)
        # Example: 768 (SigLIP) -> 2048 (Gemma)
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [B, n_patches, vision_embed_dim] -> [B, n_patches, projection_dim]
        # Projects each patch embedding independently
        hidden_states = self.linear(image_features)
        return hidden_states




class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        
        # Vision encoder: Processes raw images into patch embeddings
        # Output: [B, n_patches, vision_hidden_size]
        self.vision_tower = SiglipVisionModel(config.vision_config)
        
        # Projector: Aligns vision and language embedding spaces
        # Output: [B, n_patches, projection_dim] where projection_dim = language_hidden_size
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        # Language model: Autoregressive text generator
        # Processes unified sequence of image + text tokens
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        # Pad token ID for handling variable-length sequences
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self, 
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None 
    ):
        """
        Merge Image and Text Embeddings into Unified Sequence
        
        This is the core of the vision-language fusion mechanism.
        
        Process:
        1. Scale image features to match language model embedding scale
        2. Create masks to identify image tokens, text tokens, and padding
        3. Insert image embeddings at positions marked by image_token_index
        4. Place text embeddings at text token positions
        5. Zero out padding positions
        
        Why scale by sqrt(hidden_size)?
        - Language model embeddings are typically scaled by sqrt(hidden_size) during initialization
        - This ensures image features have similar magnitude to text embeddings
        - Prevents one modality from dominating the other during attention
        
        Why use masked_scatter instead of torch.where?
        - image_features has shape [B, n_patches, embed_dim] (e.g., [B, 196, 2048])
        - final_embedding has shape [B, seq_len, embed_dim] (e.g., [B, 200, 2048])
        - We need to scatter image patches into specific positions, not replace entire sequence
        - masked_scatter allows us to insert multiple image tokens at different positions
        
        Example sequence:
        input_ids = [256000, 256000, ..., 256000, 1, 65, 78, 99, ...]
                      ↑ image tokens (196 of them)  ↑ BOS  ↑ text tokens
        """
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # Ensure dtype match for mixed precision (important for autocast)
        # Mixed precision training uses FP16 for speed, but we need consistent dtypes
        image_features = image_features.to(dtype=dtype)
        
        # Scale image features to match language model embedding scale
        # This normalization ensures image and text embeddings are in similar ranges
        # Formula: scale = 1/sqrt(hidden_size), similar to embedding initialization
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Initialize final embedding tensor
        # We'll fill this with text embeddings and image embeddings at appropriate positions
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        # Create masks to identify different token types in the sequence
        
        # Text mask: positions that are NOT image tokens and NOT padding
        # Shape: [B, seq_len] - boolean mask
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # Example: input_ids = [256000, 256000, 256000, 256000, 256000, 1, 65, 78, 99, 21, 11, 2]
        #          text_mask => [False, False, False, False, False, True, True, True, True, True, True, True]

        # Image mask: positions marked with image_token_index
        # Shape: [B, seq_len] - boolean mask
        image_mask = input_ids == self.config.image_token_index
        # Example: image_mask => [True, True, True, True, True, False, False, False, False, False, False, False]

        # Padding mask: positions with padding tokens
        # Shape: [B, seq_len] - boolean mask
        pad_mask = input_ids == self.pad_token_id
        # Example: pad_mask => [False, False, False, False, False, False, False, False, False, False, False, False]

        # Expand masks to embedding dimension for element-wise operations
        # [B, seq_len] -> [B, seq_len, embed_dim]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Step 1: Place text embeddings at text token positions
        # torch.where: if text_mask_expanded is True, use inputs_embeds, else keep final_embedding (zeros)
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)

        # Step 2: Insert image embeddings at image token positions
        # We use masked_scatter because image_features has different sequence length than final_embedding
        # masked_scatter: scatter image_features into final_embedding where image_mask_expanded is True
        # This efficiently inserts multiple image patch embeddings at their designated positions
        scaled_image_features = scaled_image_features.to(dtype=final_embedding.dtype)
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)

        # Step 3: Zero out padding tokens (safety measure, though model expects no padding)
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        # Create the attention mask for causal (autoregressive) attention
        # Note: We use fill_value=0 (no masking) because the model expects no padding
        # In standard causal attention, we'd mask future tokens, but here we handle it differently
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill phase: processing the entire input sequence at once
            # We don't mask any tokens because:
            # 1. Model expects attention_mask to be all ones (no padding)
            # 2. Causal masking is handled internally by the language model
            # 3. All tokens in the sequence can attend to all previous tokens
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Generation phase: generating one token at a time
            # Query length is 1 (the new token being generated)
            # KV length = cached tokens + new token
            assert q_len == 1, "During generation, only one token should be processed at a time"
            kv_len = kv_cache.num_items() + q_len
            # No masking needed: the single query token can attend to all previous tokens
            # This works because we have no padding and proper causal structure
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add head dimension for multi-head attention
        # Attention mask shape: [B, n_heads, Q_len, KV_len]
        # This allows each attention head to use the same mask
        causal_mask = causal_mask.unsqueeze(1)

        # Compute position IDs for rotary position embeddings (RoPE)
        # Position IDs tell the model where each token is in the sequence
        if kv_cache is not None and kv_cache.num_items() > 0:
            # Generation phase: position is the cumulative length of the sequence
            # The new token's position = total number of tokens processed so far
            position_ids = attention_mask.cumsum(-1)[:, -1]  # Get last cumulative sum value
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)  # Add batch dimension if needed

        else:
            # Prefill phase: create position IDs for all tokens in the sequence
            # cumsum gives sequential positions: [1, 2, 3, 4, ...]
            # masked_fill sets padding positions to 1 (minimum valid position)
            position_ids = attention_mask.cumsum(-1).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids


    def forward(
        self, input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # Step 1: Extract text token embeddings
        # Get embeddings for all tokens (including image token placeholders)
        # Shape: (B, seq_len, hidden_size)
        # Note: Image token positions will be replaced with actual image features later
        embed_device = next(self.language_model.get_input_embeddings().parameters()).device
        input_ids = input_ids.to(embed_device)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Step 2: Process image and merge with text embeddings
        # Only process image during prefill phase (first forward pass)
        if kv_cache is None or kv_cache.num_items() == 0:
            # PREFILL PHASE 
            # Process the entire input: image + initial text prompt
            
            # 2a. Encode image using vision tower
            # Move pixel_values to vision tower's device (may be on different GPU)
            vision_device = next(self.vision_tower.parameters()).device
            pixel_values = pixel_values.to(vision_device).to(inputs_embeds.dtype)
            # Vision encoder: [B, C, H, W] -> [B, n_patches, vision_embed_dim]
            # Example: [1, 3, 224, 224] -> [1, 196, 768]
            selected_image_feature = self.vision_tower(pixel_values)
            
            # 2b. Project image features to language model space
            # Multi-modal projector: [B, n_patches, vision_embed_dim] -> [B, n_patches, hidden_size]
            # Example: [1, 196, 768] -> [1, 196, 2048]
            # This aligns vision and language embedding spaces
            image_features = self.multi_modal_projector(selected_image_feature)
            
            # 2c. Move image features to text embedding device for merging
            image_features = image_features.to(inputs_embeds.device)
            
            # 2d. Merge image and text embeddings into unified sequence
            # This replaces image_token_index positions with actual image features
            # Output: [B, seq_len, hidden_size] with image features inserted
            inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, kv_cache
            )
            
            # Clear intermediate tensors to free GPU memory
            # Important for large models and multi-GPU setups
            del selected_image_feature, image_features
            
        else:
            # GENERATION PHASE
            # Only processing new text tokens (one at a time)
            # Image was already processed in prefill phase and is in KV cache
            
            batch_size = input_ids.shape[0]
            q_len = inputs_embeds.shape[1]  # Should be 1 during generation
            dtype, device = inputs_embeds.dtype, inputs_embeds.device
            
            # Save original attention_mask for position_ids calculation
            original_attention_mask = attention_mask
            
            # Create causal mask for generation
            # Query length = 1 (new token), KV length = cached tokens + new token
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            ).unsqueeze(1)  # Add head dimension: [B, n_heads, Q_len, KV_len]
            
            # Compute position IDs: new token's position = total sequence length
            position_ids = original_attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            
            # Set attention_mask to causal_mask for language_model
            attention_mask = causal_mask

        # Step 3: Move inputs to first language model layer device
        # For multi-GPU setups, different layers may be on different GPUs
        # We move tensors as needed to match each layer's device
        first_layer_device = next(self.language_model.model.layers[0].parameters()).device
        inputs_embeds = inputs_embeds.to(first_layer_device)
        attention_mask = attention_mask.to(first_layer_device) if attention_mask is not None else None
        position_ids = position_ids.to(first_layer_device) if position_ids is not None else None
        
        # Step 4: Forward through language model
        # Language model processes the unified sequence (image + text tokens)
        # It generates logits for next token prediction
        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            kv_cache = kv_cache,
        )

        return outputs