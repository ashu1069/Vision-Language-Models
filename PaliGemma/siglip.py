"""
SigLIP Vision Encoder Implementation

SigLIP (Sigmoid Loss for Language-Image Pre-training) is a vision transformer (ViT)
used as the vision encoder in PaliGemma.

Architecture:
- Vision Transformer (ViT): Processes images as sequences of patches
- Patch embedding: Divides image into patches, embeds each patch
- Position embeddings: Adds positional information to patches
- Transformer encoder: Self-attention over patches
- No CLS token: Uses all patch embeddings (unlike some ViT variants)

Why SigLIP?
- Pre-trained on large-scale image-text pairs
- Strong vision representations for vision-language tasks
- Efficient architecture (standard ViT)
- Used in many modern vision-language models

Key design:
- Image -> Patches -> Embeddings -> Transformer -> Patch features
- Output: [B, n_patches, embed_dim] where n_patches = (image_size/patch_size)²
- Example: 224x224 image, 16x16 patches -> 196 patches
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,  # Dimension of patch embeddings (ViT standard)
        intermediate_size=3072,  # FFN hidden size (typically 4x hidden_size)
        num_hidden_layers=12,  # Number of transformer encoder layers
        num_attention_heads=12,  # Number of attention heads
        num_channels=3,  # RGB channels
        image_size=224,  # Input image size (height/width)
        patch_size=16,  # Patch size (each patch is patch_size x patch_size)
        layer_norm_eps=1e-6,  # Epsilon for layer normalization
        attention_dropout=0.0,  # Dropout rate for attention (0.0 = no dropout)
        num_image_tokens: int=None,  # Number of image tokens (calculated from image/patch size)
        **kwargs
    ):
        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.num_channels=num_channels
        self.image_size=image_size
        self.patch_size=patch_size
        self.attention_dropout=attention_dropout
        self.layer_norm_eps=layer_norm_eps
        self.num_image_tokens=num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    """
    Vision Embeddings: Convert Image to Patch Embeddings
    
    Process:
    1. Patch embedding: Convolution to extract patch features
    2. Flatten: Convert 2D patch grid to 1D sequence
    3. Position embedding: Add positional information
    
    Why use convolution for patch embedding?
    - Efficient: Single convolution operation
    - Equivalent to: split image into patches, then linear projection
    - Faster than manual patching + linear layer
    - Standard approach in ViT models
    
    Position embeddings:
    - Learnable embeddings for each patch position
    - Allows model to understand spatial relationships
    - Fixed positions (not learned per image)
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # Patch embedding via convolution
        # Convolution with kernel_size=stride=patch_size effectively extracts patches
        # Example: 224x224 image, 16x16 patches -> 14x14 patches
        # [B, 3, 224, 224] -> [B, embed_dim, 14, 14]
        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,  # 3 (RGB)
            out_channels = self.embed_dim,  # 768 (embedding dimension)
            kernel_size = self.patch_size,  # 16x16
            stride = self.patch_size,  # Non-overlapping patches
            padding = 0  # No padding (valid convolution)
        )

        # Calculate number of patches
        # Example: 224/16 = 14, so 14² = 196 patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        
        # Learnable position embeddings
        # Each patch position gets a unique embedding
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
        # Pre-compute position IDs (not a parameter, just indices)
        # Shape: [1, num_patches] = [1, 196]
        # persistent=False: don't save to checkpoint (can be recomputed)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Convert image to patch embeddings with position information.
        
        Process:
        1. Extract patches via convolution
        2. Flatten to sequence
        3. Add position embeddings
        """
        _, _, height, width = pixel_values.shape  # [B, C, H, W]
        
        # Patch embedding: convolution extracts patches
        # [B, 3, 224, 224] -> [B, embed_dim, 14, 14]
        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten spatial dimensions: convert 2D grid to 1D sequence
        # [B, embed_dim, 14, 14] -> [B, embed_dim, 196]
        embeddings = patch_embeds.flatten(2)

        # Transpose: move sequence dimension to position 1
        # [B, embed_dim, 196] -> [B, 196, embed_dim]
        embeddings = embeddings.transpose(1, 2)

        # Add position embeddings
        # Each patch gets its position embedding added
        # [B, 196, embed_dim] + [1, 196, embed_dim] -> [B, 196, embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings

class SiglipAttention(nn.Module):
    """
    Multi-Head Self-Attention (Standard Transformer Attention)
    
    Architecture:
    - Standard multi-head attention from "Attention is All You Need"
    - All heads have same dimension (unlike GQA in language model)
    - Self-attention: patches attend to other patches
    
    Why self-attention for vision?
    - Patches can attend to relevant patches (e.g., object parts)
    - Captures long-range dependencies (distant patches can interact)
    - Enables understanding of spatial relationships
    - Standard approach in vision transformers
    
    Attention mechanism:
    1. Project to Q, K, V (all same dimension)
    2. Split into heads
    3. Compute attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
    4. Concatenate heads and project output
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads  # Dimension per head
        self.scale = self.head_dim ** -0.5  # 1/sqrt(head_dim) for scaled attention
        self.dropout = config.attention_dropout

        # Projection matrices for Q, K, V
        # All project to same dimension (standard MHA, not GQA)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)  # W_k
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)  # W_q
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)  # W_v
        # Output projection: combines heads
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [B, n_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) # X * W_q
        key_states = self.k_proj(hidden_states) # X * W_k
        value_states = self.v_proj(hidden_states) # X * W_v
        # no contextualization has happened yet

        # query_states: [B, n_patches, embed_dim] -> [B, n_heads, n_patches, head_dim]
        # .view(): embed_dim -> (n_heads, head_dim)
        # .transpose() -> [B, n_pacthes, n_heads, head_dim] -> [B, n_heads, n_patches, head_dim] => better for parallelization
        # and each head should learn to relate tokens (or patches) differently.
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale) # [B, n_heads, n_patches, n_patches]
        # apply the attention mask -> multiply by values
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f"{attn_weights.size()}"
            )

        # Apply the softmax row-wise. attn_weights: [B, n_heads, n_patches, n_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training = self.training)

        # Multiply the attention weights by the value states. attn_output: [B, n_heads, n_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f"{attn_output.size()}"
            )

        # [B, n_heads, n_patches, head_dim] -> [B, n_patches, n_heads, head_dim]
        attn_output = attn_output.transpose(1,2).contiguous() # needed continuous memory block (layout) for the next operation

        # [B, n_patches, n_heads, head_dim] -> [B, n_patches, embed_dim] => concatenating all the independent heads
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # now need to mix the output from independent heads
        # [B, n_patches, embed_sim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    """
    Feed-Forward Network for Vision Transformer
    
    Architecture:
    - Two linear layers with GELU activation
    - Standard transformer FFN (not SwiGLU like language model)
    - Expands to intermediate_size then projects back
    
    Why intermediate_size = 4 * hidden_size?
    - Standard transformer scaling
    - Provides more capacity for non-linear transformations
    - Empirically good balance between capacity and efficiency
    
    GELU activation:
    - Gaussian Error Linear Unit: smoother than ReLU
    - Better for vision tasks than ReLU
    - approximate='tanh': faster computation, nearly identical results
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # Expand: hidden_size -> intermediate_size (typically 4x)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Project back: intermediate_size -> hidden_size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FFN.
        
        Process:
        1. Expand dimension
        2. Apply GELU activation
        3. Project back to original dimension
        """
        # Expand: [B, n_patches, embed_dim] -> [B, n_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # Apply GELU activation (with tanh approximation for speed)
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')

        # Project back: [B, n_patches, intermediate_size] -> [B, n_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer (Post-Norm Architecture)
    
    Architecture:
    - Post-norm: Normalize after attention/MLP (not before)
    - Residual connections: Enable gradient flow
    - Two sub-layers: Self-attention and MLP
    
    Note: This uses post-norm (different from language model's pre-norm)
    - Post-norm: attention -> norm -> residual
    - Pre-norm: norm -> attention -> residual
    - Post-norm is standard for vision transformers
    - Pre-norm is better for very deep language models
    
    Structure:
    1. Self-Attention -> LayerNorm -> Residual
    2. MLP -> LayerNorm -> Residual
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)  # Multi-head self-attention
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # Post-attention norm
        self.mlp = SiglipMLP(config)  # Feed-forward network
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # Post-MLP norm
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder layer.
        
        Post-norm architecture with residual connections:
        1. Self-attention
        2. Normalize and add residual
        3. MLP
        4. Normalize and add residual
        """
        # Save input for residual connection
        residual = hidden_states
        
        # Self-attention: patches attend to other patches
        # [B, n_patches, embed_dim] -> [B, n_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)

        # Post-norm: normalize after attention
        hidden_states = self.layer_norm1(hidden_states)
        # Residual connection: add input back
        hidden_states = residual + hidden_states 

        # Save for second residual connection
        residual = hidden_states
        
        # Post-norm: normalize before MLP
        hidden_states = self.layer_norm2(hidden_states)

        # MLP: non-linear transformation
        # Captures complex patterns and increases model capacity
        hidden_states = self.mlp(hidden_states)
        
        # Second residual connection
        hidden_states = residual + hidden_states

        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states

class SiglipVisionTransformer(nn.Module):
    """
    Complete SigLIP Vision Transformer
    
    Architecture:
    1. Embeddings: Convert image to patch embeddings with positions
    2. Encoder: Stack of transformer layers
    3. Post-layer norm: Final normalization
    
    Output:
    - [B, n_patches, embed_dim]: Feature representation for each patch
    - All patches are used (no CLS token like some ViT variants)
    - These features will be projected to language model space
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config=config
        embed_dim=config.hidden_size

        # Embeddings: image -> patch embeddings with positions
        self.embeddings = SiglipVisionEmbeddings(config)
        
        # Encoder: stack of transformer layers
        self.encoder = SiglipEncoder(config)
        
        # Final layer normalization
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision transformer.
        
        Process:
        1. Convert image to patch embeddings
        2. Pass through encoder layers
        3. Apply final normalization
        """
        # Embeddings: [B, C, H, W] -> [B, n_patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)

        # Encoder: process patches through transformer layers
        # [B, n_patches, embed_dim] -> [B, n_patches, embed_dim]
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        # Final normalization
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

class SiglipVisionModel(nn.Module):
    """
    SigLIP Vision Model Wrapper
    
    Simple wrapper around SiglipVisionTransformer.
    Provides clean interface for PaliGemma to use.
    """
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        """
        Forward pass: image -> patch features.
        
        Input: [B, C, H, W] - raw image pixels
        Output: [B, n_patches, embed_dim] - patch embeddings
        """
        # [B, C, H, W] -> [B, n_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values)