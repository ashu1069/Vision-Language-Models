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
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        # Ensure num_attention_layers is set (defaults to num_hidden_layers if not provided)
        if 'num_attention_layers' not in text_config:
            text_config['num_attention_layers'] = text_config.get('num_hidden_layers')
        
        self.text_config = GemmaConfig(**text_config, pad_token_id = pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [B, n_patches, embed_dim] -> [B, n_patches, projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states




class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

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
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # Ensure dtype match for mixed precision (important for autocast)
        image_features = image_features.to(dtype=dtype)
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Combine the embeddings fo the image tokens, the text tokens, and mask out all the padding tokens
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        # Shape: [B, seq_len]: text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # input_ids = [567, 567, 567, 567, 567, 1 (bos token), 65, 78, 99, 21, 11, 2 (eos token)]
        # after applying text_mask => [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

        # Shape: [B, seq_len]: image tokens
        image_mask = input_ids == self.config.image_token_index
        # after applying image masks => [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

        # Shape: [B, seq_len]: padding tokens
        pad_mask = input_ids == self.pad_token_id
        # apter applying padding mask => [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embedding
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)

        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features 
        # is not equal to the sequence length of the final embedding
        # Ensure dtype match for masked_scatter (important for mixed precision)
        scaled_image_features = scaled_image_features.to(dtype=final_embedding.dtype)
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)

        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        # Create the attention mask
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Don't mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we're generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attent all previous tokens
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [B, Q_len, KV_len] -> [B, n_heads_Q, Q_len, KV_len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # the position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

        else:
            # create a position_ids based on the size of attention mask
            # For masked tokens, use the number 1 as position
            position_ids = attention_mask.cumsum(-1).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids


    def forward(
        self, input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extract the input embeddings
        # shape: (B, seq_len, hidden_size)
        # Ensure input_ids are on the embedding device
        embed_device = next(self.language_model.get_input_embeddings().parameters()).device
        input_ids = input_ids.to(embed_device)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images (only process image on first forward pass)
        # Skip image processing during generation phase (when kv_cache has items)
        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill phase: process image
            # Move pixel_values to vision tower device
            vision_device = next(self.vision_tower.parameters()).device
            pixel_values = pixel_values.to(vision_device).to(inputs_embeds.dtype)
            # [B, C, H, W] -> [B, n_patches, embed_dim]
            selected_image_feature = self.vision_tower(pixel_values)
            # [B, n_patches, embed_dim] -> [B, n_patches, hidden_size] => resizing the image to text shape
            image_features = self.multi_modal_projector(selected_image_feature)
            # Move image_features to inputs_embeds device for merging
            image_features = image_features.to(inputs_embeds.device)
            # Merge the embeddings of the text tokens and the image tokens
            inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)
            # Clear image features to free memory
            del selected_image_feature, image_features
        else:
            # Generation phase: only text tokens, no image processing needed
            # inputs_embeds already contains only text embeddings (no image tokens in generation)
            # Create attention_mask (causal mask) and position_ids for text-only generation
            batch_size = input_ids.shape[0]
            q_len = inputs_embeds.shape[1]
            dtype, device = inputs_embeds.dtype, inputs_embeds.device
            
            # Save original attention_mask for position_ids calculation
            original_attention_mask = attention_mask
            
            # Create causal mask for generation (same as in _merge_input_ids_with_image_features)
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            ).unsqueeze(1)  # Add head dimension: [B, n_heads, Q_len, KV_len]
            
            # Position IDs for generation (use original attention_mask parameter)
            position_ids = original_attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            
            # Set attention_mask to causal_mask for language_model
            attention_mask = causal_mask

        # Move inputs_embeds to the first layer device (language model will handle further movement)
        first_layer_device = next(self.language_model.model.layers[0].parameters()).device
        inputs_embeds = inputs_embeds.to(first_layer_device)
        attention_mask = attention_mask.to(first_layer_device) if attention_mask is not None else None
        position_ids = position_ids.to(first_layer_device) if position_ids is not None else None
        
        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            kv_cache = kv_cache,
        )
        
        # Move logits to CPU or appropriate device for final processing
        if isinstance(outputs, dict) and 'logits' in outputs:
            # Logits are already on the last GPU (lm_head device)
            pass

        return outputs