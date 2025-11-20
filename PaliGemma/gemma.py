"""
Gemma Language Model Implementation

This module implements the Gemma decoder-only transformer architecture used in PaliGemma.
Key components:
1. KVCache: Efficient caching for autoregressive generation
2. GemmaAttention: Multi-head attention with Grouped Query Attention (GQA)
3. GemmaRotaryEmbedding: Rotary Position Embeddings (RoPE) for position encoding
4. GemmaMLP: SwiGLU feed-forward network
5. GemmaRMSNorm: Root Mean Square Layer Normalization

Architecture choices:
- Decoder-only: Autoregressive generation (like GPT)
- GQA: Reduces memory for KV cache (fewer key/value heads than query heads)
- RoPE: Relative position encoding that generalizes to longer sequences
- RMSNorm: More efficient than LayerNorm, similar performance
- SwiGLU: Gated activation function (better than ReLU/GELU)
"""

import torch
from torch import nn
from typing import Optional, Tuple, List
import math

class KVCache():
    """
    Key-Value Cache for Efficient Autoregressive Generation
    
    Purpose:
    - During text generation, we generate one token at a time
    - For each new token, we need to compute attention over all previous tokens
    - Recomputing K and V for all previous tokens is wasteful
    - Cache stores K and V from previous tokens, only compute for new token
    
    Why this dramatically speeds up generation:
    - Prefill phase: Process entire prompt once, cache all K/V
    - Generation phase: Only compute K/V for the new token, reuse cached ones
    - Reduces computation from O(n²) to O(n) per token
    
    Memory layout:
    - Separate cache per layer (each transformer layer has its own attention)
    - Shape: [B, num_heads_kv, seq_len, head_dim]
    - Concatenated along sequence dimension as new tokens are generated
    """
    def __init__(self) -> None:
        # One cache entry per transformer layer
        # Each entry stores keys and values for that layer
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        """
        Returns the number of cached tokens (sequence length in cache).
        This tells us how many tokens have been processed so far.
        """
        if len(self.key_cache) == 0:
            return 0  # Cache is empty (prefill phase)
        else:
            # The shape of the key_cache is [B, num_heads_kv, seq_len, head_dim]
            # Return seq_len (the sequence dimension, which is -2)
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache for a specific layer.
        
        Process:
        1. If this is the first time this layer is cached, store the K/V
        2. Otherwise, concatenate new K/V with existing cache
        
        Why concatenate instead of append?
        - PyTorch tensors are immutable, we need to create new tensor
        - Concatenation is efficient and maintains proper tensor structure
        - Allows us to build up the sequence incrementally
        
        Device/dtype handling:
        - Ensures new K/V match cached K/V device and dtype
        - Critical for multi-GPU setups and mixed precision training
        """

        if len(self.key_cache) <= layer_idx:
            # First time caching for this layer: initialize the cache
            # This happens during prefill phase (first forward pass)
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        else:
            # Generation phase: append new K/V to existing cache
            # Each tensor has shape: [B, num_heads_kv, seq_len, head_dim]
            # New tokens have seq_len=1, cached tokens have seq_len=N
            # After concat: [B, num_heads_kv, N+1, head_dim]
            
            # Ensure dtype and device match for mixed precision and multi-GPU
            # This prevents errors when tensors are on different devices or have different dtypes
            cache_key_device = self.key_cache[layer_idx].device
            cache_key_dtype = self.key_cache[layer_idx].dtype
            cache_value_device = self.value_cache[layer_idx].device
            cache_value_dtype = self.value_cache[layer_idx].dtype
            key_states = key_states.to(device=cache_key_device, dtype=cache_key_dtype)
            value_states = value_states.to(device=cache_value_device, dtype=cache_value_dtype)
            
            # Concatenate along sequence dimension (-2)
            # This grows the cache with each new token
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim = -2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim = -2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaConfig():
    def __init__(
        self,
        vocab_size, 
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim = 256,
        max_position_embeddings = 8192,
        rms_norm_eps = 1e-6,
        rope_theta = 10000.0,
        attention_bias = False,
        attention_dropout = 0.0,
        pad_token_id = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class GemmaRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    Why RMSNorm instead of LayerNorm?
    - LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
    - RMSNorm: x / sqrt(mean(x²) + eps) * weight
    - RMSNorm removes the mean-centering step, making it faster
    - Empirically, RMSNorm performs similarly to LayerNorm for transformers
    - Used in models like LLaMA, Gemma, etc.
    
    Formula:
    - Normalize by RMS: x_norm = x / sqrt(mean(x²) + eps)
    - Scale: output = x_norm * (1 + weight)
    - The (1 + weight) formulation allows weight to be initialized to zero
    - Zero initialization means identity function at start of training (better for deep networks)
    
    Why compute in float32 then convert back?
    - More numerically stable for the sqrt operation
    - Prevents precision loss in mixed precision training
    """
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps  # Small constant to prevent division by zero
        # Learnable scale parameter (initialized to zero = identity function)
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """
        Compute RMS normalization: x / sqrt(mean(x²) + eps)
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Compute normalization in float32 for numerical stability
        output = self._norm(x.float())
        # Apply learnable scale: (1 + weight) allows zero initialization
        output = output * (1.0 + self.weight.float())
        # Convert back to original dtype (preserves mixed precision)
        return output.to(x.dtype)

class GemmaMLP(nn.Module):
    """
    Feed-Forward Network
    
    Why approximate='tanh' for GELU?
    - GELU: x * Φ(x) where Φ is CDF of standard normal
    - Exact GELU requires erf() which is expensive
    - tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    - Much faster, nearly identical results
    
    Design:
    - No bias in linear layers: reduces parameters, often performs similarly
    - intermediate_size typically 4x hidden_size (standard transformer scaling)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Gate projection: controls which parts of up_proj are active
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # Up projection: main transformation
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # Down projection: projects back to hidden_size
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Gate: [B, seq_len, hidden_size] -> [B, seq_len, intermediate_size]
        y = self.gate_proj(x)
        # Apply Swish activation (GELU with tanh approximation)
        y = torch.nn.functional.gelu(y, approximate='tanh')
        # Up projection: [B, seq_len, hidden_size] -> [B, seq_len, intermediate_size]
        j = self.up_proj(x)
        # Element-wise multiplication: gate acts as learned filter
        z = y * j
        # Down projection: [B, seq_len, intermediate_size] -> [B, seq_len, hidden_size]
        z = self.down_proj(z)
        return z

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat Key/Value states for Grouped Query Attention (GQA)
    
    Purpose:
    - In GQA, we have fewer key/value heads than query heads
    - Example: 8 query heads, 1 key/value head (Multi-Query Attention)
    - We need to repeat the single K/V head 8 times to match Q heads
    
    Why GQA?
    - Reduces memory: fewer K/V heads means smaller KV cache
    - Faster inference: less computation in attention
    - Similar performance: query heads are more important than K/V heads
    - Used in models like PaLM, LLaMA-2, Gemma
    
    Process:
    1. Expand K/V along a new dimension
    2. Repeat n_rep times (where n_rep = num_query_heads / num_kv_heads)
    3. Reshape to match query head count
    
    Example:
    - Input: [B, 1, seq_len, head_dim] (1 KV head)
    - n_rep = 8 (8 query heads)
    - Output: [B, 8, seq_len, head_dim] (repeated 8 times)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape

    if n_rep == 1:
        # No repetition needed (standard multi-head attention)
        return hidden_states

    # Expand along new dimension and repeat
    # [B, num_kv_heads, seq_len, head_dim] -> [B, num_kv_heads, n_rep, seq_len, head_dim]
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # Reshape to match query head count
    # [B, num_kv_heads, n_rep, seq_len, head_dim] -> [B, num_kv_heads * n_rep, seq_len, head_dim]
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    Purpose:
    - Encodes absolute position information into query and key vectors
    - Applied via rotation: rotates Q and K by position-dependent angles
    - Attention scores naturally encode relative positions
    
    Why RoPE instead of learned position embeddings?
    - Generalizes to longer sequences (extrapolation)
    - Relative position encoding (better for understanding relationships)
    - No learned parameters (saves memory)
    - Used in models like LLaMA, PaLM, Gemma
    
    Mathematical foundation:
    - For position m, rotate by angle: θ_m = m * θ_i
    - Where θ_i = base^(2i/dim) for i = 0, 1, ..., dim/2 - 1
    - Rotation matrix applied to pairs of dimensions
    
    Why different frequencies (theta_i)?
    - Lower frequencies (small i): capture long-range dependencies
    - Higher frequencies (large i): capture short-range dependencies
    - Similar to Fourier transform: different frequencies for different scales
    
    Base parameter:
    - base=10000: standard value, controls frequency distribution
    - Larger base = lower frequencies = better for long sequences
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim  # head_dim: dimension of each attention head
        self.max_position_embeddings = max_position_embeddings  # Maximum sequence length
        self.base = base  # Base for frequency calculation

        # Calculate inverse frequencies: theta_i = base^(2i/dim)
        # We compute 1/theta_i (inv_freq) for efficiency
        # Only need dim//2 frequencies (each frequency used for a pair of dimensions)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        # Register as buffer (not a parameter, but part of model state)
        # persistent=False: don't save to checkpoint (can be recomputed)
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        """
        Compute cos and sin for rotary embeddings.
        
        Process:
        1. Compute frequencies for each position: freq = position * inv_freq
        2. Apply cos/sin to get rotation angles
        3. These will be used to rotate Q and K vectors
        
        Why @torch.no_grad()?
        - Position embeddings are fixed (not learned)
        - No need to compute gradients, saves memory and computation
        """
        # x: [B, n_attn_heads, seq_len, head_size]
        # Move inv_freq to same device as input
        self.inv_freq.to(x.device)

        # Expand inv_freq for batch dimension
        # inv_freq: [head_dim//2] -> [B, head_dim//2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)

        # Expand position_ids for frequency dimension
        # position_ids: [B, seq_len] -> [B, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Handle device type for autocast (MPS doesn't support autocast well)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            # Compute frequencies: freq = position * inv_freq
            # Matrix multiply: [B, head_dim//2, 1] @ [B, 1, seq_len] -> [B, head_dim//2, seq_len]
            # Transpose: [B, seq_len, head_dim//2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

            # Duplicate frequencies for both dimensions in each pair
            # [B, seq_len, head_dim//2] -> [B, seq_len, head_dim]
            emb = torch.cat((freqs, freqs), dim = -1)

            # Compute cos and sin for rotation
            # cos, sin: [B, seq_len, head_dim]
            cos = emb.cos()
            sin = emb.sin()
        
        # Convert back to input dtype (preserves mixed precision)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """
    Split tensor in half and swap halves with sign flip.
    
    This implements the rotation operation for RoPE.
    For a 2D rotation by angle θ:
    [x1']   [cos(θ)  -sin(θ)] [x1]
    [x2'] = [sin(θ)   cos(θ)] [x2]
    
    Which can be written as:
    x1' = x1*cos(θ) - x2*sin(θ)
    x2' = x1*sin(θ) + x2*cos(θ)
    
    Or in vectorized form:
    [x1', x2'] = [x1, x2] * [cos, -sin; sin, cos]
    = [x1*cos - x2*sin, x1*sin + x2*cos]
    = [x1, x2] * cos + [-x2, x1] * sin
    
    This function computes [-x2, x1] part.
    """
    # Split into two halves
    x1 = x[..., : x.shape[-1] // 2]  # First half
    x2 = x[..., x.shape[-1] // 2 :]  # Second half

    # Swap and flip sign: [-x2, x1]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_embed(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply Rotary Position Embedding to query and key tensors.
    
    Formula: rotated = x * cos + rotate_half(x) * sin
    
    Why apply to both Q and K?
    - Attention score = Q @ K^T
    - Rotating both Q and K by position-dependent angles
    - Results in attention scores that encode relative positions
    
    Why not apply to V?
    - Value vectors don't need position information
    - Position is already encoded in attention weights (Q @ K^T)
    - Applying to V would be redundant
    
    Process:
    1. Expand cos/sin to match Q/K shape (add head dimension)
    2. Apply rotation formula to Q and K
    3. Return rotated Q and K
    """
    # Add head dimension to cos/sin to match Q/K shape
    # Q/K shape: [B, n_heads, seq_len, head_dim]
    # cos/sin shape: [B, seq_len, head_dim]
    # After unsqueeze: [B, 1, seq_len, head_dim] (broadcasts to n_heads)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Apply RoPE rotation formula
    # rotated = x * cos(θ) + rotate_half(x) * sin(θ)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class GemmaAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with Rotary Position Embeddings
    
    Architecture:
    - Multi-head attention with fewer key/value heads than query heads
    - Example: 8 query heads, 1 key/value head (Multi-Query Attention)
    - RoPE applied to Q and K for position encoding
    - Causal masking for autoregressive generation
    
    Why Grouped Query Attention?
    - Standard MHA: num_heads Q, K, V heads (e.g., 8 each)
    - GQA: num_heads Q heads, fewer K/V heads (e.g., 8 Q, 1 K/V)
    - Benefits:
      * Smaller KV cache (critical for long sequences)
      * Faster attention computation
      * Similar quality (query heads are more important)
    
    Attention mechanism:
    1. Project to Q, K, V (with different head counts)
    2. Apply RoPE to Q and K
    3. Repeat K/V to match Q head count
    4. Compute attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
    5. Project output back to hidden_size
    
    Causal attention:
    - During generation, each token can only attend to previous tokens
    - Implemented via attention mask (though this model expects no padding)
    """
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx  # Layer index for KV cache

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads  # Number of query heads
        self.head_dim = config.head_dim  # Dimension per head
        self.num_key_value_heads = config.num_key_value_heads  # Number of K/V heads (≤ num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # Repeat factor
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta  # Base for RoPE frequencies
        self.is_causal = True  # Autoregressive generation

        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"

        # Projection matrices for Q, K, V
        # Example: num_heads=8, num_kv_heads=1, hidden_size=1024, head_dim=128
        # Wq: [1024, 8 * 128] = [1024, 1024] - projects to 8 query heads
        # Wk: [1024, 1 * 128] = [1024, 128] - projects to 1 key head
        # Wv: [1024, 1 * 128] = [1024, 128] - projects to 1 value head
        # The single K/V head will be repeated 8 times to match Q heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias = config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        
        # Output projection: combines all heads back to hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Rotary position embeddings for Q and K
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings = self.max_position_embeddings,
            base = self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """
        Forward pass through attention layer.
        
        Process:
        1. Project to Q, K, V
        2. Reshape for multi-head attention
        3. Apply RoPE to Q and K
        4. Update KV cache (if provided)
        5. Repeat K/V to match Q head count (GQA)
        6. Compute attention scores: Q @ K^T / sqrt(head_dim)
        7. Apply attention mask and softmax
        8. Apply to values: attn_weights @ V
        9. Concatenate heads and project output
        
        Two phases with KV cache:
        - Prefill: Process entire sequence, cache all K/V
        - Generation: Process one token, reuse cached K/V
        """

        batch_size, q_len, _ = hidden_states.size()  # [B, seq_len, hidden_size]

        # Step 1: Project to Query, Key, Value
        # [B, seq_len, hidden_size] -> [B, seq_len, num_heads * head_dim]
        query_states = self.q_proj(hidden_states)
        # [B, seq_len, hidden_size] -> [B, seq_len, num_kv_heads * head_dim]
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Step 2: Reshape for multi-head attention
        # Split into heads and move head dimension to position 1
        # [B, seq_len, num_heads * head_dim] -> [B, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1,2)
        # [B, seq_len, num_kv_heads * head_dim] -> [B, num_kv_heads, seq_len, head_dim]
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

        # Step 3: Apply Rotary Position Embeddings to Q and K
        # RoPE encodes position information via rotation
        # [B, num_heads, seq_len, head_dim] -> cos/sin tensors
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # Rotate Q and K by position-dependent angles
        query_states, key_states = apply_rotary_pos_embed(query_states, key_states, cos, sin)

        # Step 4: Update KV cache (if provided)
        # During generation, this appends new K/V to cache
        # During prefill, this initializes the cache
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Step 5: Repeat K/V to match Q head count (Grouped Query Attention)
        # If num_kv_heads < num_heads, repeat K/V heads to match
        # Example: 1 K/V head -> repeat 8 times to match 8 Q heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Step 6: Compute attention scores
        # Scaled dot-product attention: Q @ K^T / sqrt(head_dim)
        # Shape: [B, num_heads_q, seq_len_q, seq_len_kv]
        # The sqrt(head_dim) scaling prevents softmax from saturating
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Step 7: Apply attention mask
        # Mask is added (not multiplied) because it uses -inf for masked positions
        # After adding mask, masked positions become -inf, softmax makes them 0
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Step 8: Apply softmax to get attention probabilities
        # Compute in float32 for numerical stability, then convert back
        # [B, num_heads_q, seq_len_q, seq_len_kv]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Step 9: Apply dropout (only during training)
        # Prevents overfitting by randomly zeroing some attention weights
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training = self.training)

        # Step 10: Apply attention to values
        # Weighted sum of values: attn_weights @ V
        # [B, num_heads_q, seq_len_q, seq_len_kv] @ [B, num_heads_q, seq_len_kv, head_dim]
        # -> [B, num_heads_q, seq_len_q, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        # Sanity check: ensure output shape is correct
        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f"{attn_output.size()}"
            )
        
        # Step 11: Reshape for concatenation
        # Move head dimension back: [B, num_heads_q, seq_len_q, head_dim] -> [B, seq_len_q, num_heads_q, head_dim]
        attn_output = attn_output.transpose(1,2).contiguous()

        # Step 12: Concatenate all heads
        # [B, seq_len_q, num_heads_q, head_dim] -> [B, seq_len_q, num_heads_q * head_dim]
        # num_heads_q * head_dim = hidden_size
        attn_output = attn_output.view(batch_size, q_len, -1)

        # Step 13: Output projection
        # [B, seq_len_q, hidden_size] -> [B, seq_len_q, hidden_size]
        # Combines information from all heads
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights



class GemmaDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer (Pre-Norm Architecture)
    
    Architecture:
    - Pre-norm: Normalize before attention/MLP (not after)
    - Residual connections: Enable gradient flow in deep networks
    - Two sub-layers: Self-attention and MLP
    
    Why Pre-Norm instead of Post-Norm?
    - Pre-norm: norm -> attention -> residual
    - Post-norm: attention -> norm -> residual
    - Pre-norm is more stable for deep networks
    - Better gradient flow, easier to train
    - Used in modern models like GPT-3, LLaMA, Gemma
    
    Structure:
    1. Input -> RMSNorm -> Self-Attention -> Residual
    2. -> RMSNorm -> MLP -> Residual -> Output
    
    Residual connections:
    - Allow information to bypass layers
    - Critical for training deep networks
    - Enable identity mapping if needed
    """
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Self-attention: processes sequence with attention mechanism
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        # MLP: feed-forward network with SwiGLU activation
        self.mlp = GemmaMLP(config)

        # Pre-norm: normalize before attention and MLP
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass through decoder layer.
        
        Pre-norm architecture with residual connections:
        1. Normalize input
        2. Self-attention
        3. Add residual
        4. Normalize again
        5. MLP
        6. Add residual
        """

        # Save input for residual connection
        residual = hidden_states

        # Pre-norm: normalize before attention
        # [B, seq_len, hidden_size]
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention: compute attention over sequence
        # [B, seq_len, hidden_size] -> [B, seq_len, hidden_size]
        hidden_states, _ = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            kv_cache = kv_cache,
        )

        # Residual connection: add input back
        # This enables gradient flow and identity mapping
        hidden_states = residual + hidden_states

        # Save for second residual connection
        residual = hidden_states

        # Pre-norm: normalize before MLP
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP: feed-forward transformation
        hidden_states = self.mlp(hidden_states)

        # Second residual connection
        hidden_states = residual + hidden_states

        return hidden_states

class GemmaModel(nn.Module):
    """
    Gemma Language Model (Transformer Decoder Stack)
    
    Architecture:
    - Token embeddings: map token IDs to dense vectors
    - Stack of decoder layers: process sequence with attention and MLP
    - Final layer norm: normalize output before language modeling head
    
    Why scale embeddings by sqrt(hidden_size)?
    - Standard initialization for transformer embeddings
    - Keeps embedding magnitudes similar to other activations
    - Prevents embeddings from being too small or too large
    - Helps with training stability
    
    Multi-GPU support:
    - Different layers can be on different GPUs (model parallelism)
    - Tensors are moved to each layer's device as needed
    - Enables training/inference of very large models
    """
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings: map token IDs to hidden_size-dimensional vectors
        # padding_idx: embeddings for padding tokens are not updated during training
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Stack of transformer decoder layers
        # Each layer applies self-attention and MLP with residual connections
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Final layer normalization before language modeling head
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        """Return the token embedding layer."""
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass through language model.
        
        Process:
        1. Scale input embeddings
        2. Pass through each decoder layer
        3. Apply final layer norm
        
        Note: inputs_embeds can contain both text and image embeddings
        (image embeddings are inserted by PaliGemma's merge function)
        """

        hidden_states = inputs_embeds

        # Scale embeddings by sqrt(hidden_size)
        # This is standard initialization for transformer embeddings
        # Keeps embedding scale consistent with other activations
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer

        # Pass through each decoder layer
        for decoder_layer in self.layers:
            # Move hidden_states to the device of the current layer (for multi-GPU)
            # This enables model parallelism where different layers are on different GPUs
            layer_device = next(decoder_layer.parameters()).device
            hidden_states = hidden_states.to(layer_device)
            
            # Move attention_mask and position_ids if needed
            if attention_mask is not None:
                attention_mask = attention_mask.to(layer_device)
            if position_ids is not None:
                position_ids = position_ids.to(layer_device)
            
            # Forward through decoder layer
            # [B, seq_len, hidden_size] -> [B, seq_len, hidden_size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids = position_ids,
                kv_cache=kv_cache
            )

        # Final layer normalization
        # Move to norm device (may be on different GPU)
        norm_device = next(self.norm.parameters()).device
        hidden_states = hidden_states.to(norm_device)
        hidden_states = self.norm(hidden_states)

        return hidden_states

class GemmaForCausalLM(nn.Module):
    """
    Gemma Model for Causal Language Modeling
    
    Architecture:
    - GemmaModel: Transformer decoder stack
    - Language modeling head: projects hidden states to vocabulary logits
    
    Weight tying:
    - Input embeddings and output projection share weights
    - Reduces parameters (vocab_size * hidden_size parameters saved)
    - Empirically improves performance (better alignment between input/output)
    - Common in modern language models (GPT, LLaMA, Gemma)
    
    Output:
    - Logits: [B, seq_len, vocab_size] - unnormalized scores for each token
    - These are used for next-token prediction via softmax + sampling/argmax
    """
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)  # Transformer decoder stack
        self.vocab_size = config.vocab_size
        
        # Language modeling head: projects hidden states to vocabulary logits
        # No bias: reduces parameters, weight tying works better without bias
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        """Return the token embedding layer."""
        return self.model.embed_tokens

    def tie_weights(self):
        """
        Tie input embeddings and output projection weights.
        
        Why weight tying?
        - Input: token -> embedding (embed_tokens)
        - Output: hidden -> token (lm_head)
        - Sharing weights makes these transformations symmetric
        - Reduces parameters and often improves performance
        - Standard practice in modern language models
        """
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """
        Forward pass for causal language modeling.
        
        Process:
        1. Pass through transformer model
        2. Project to vocabulary logits
        3. Return logits and updated KV cache
        
        Output:
        - logits: [B, seq_len, vocab_size] - scores for each token in vocabulary
        - kv_cache: Updated cache (if provided)
        """

        # Forward through transformer model
        # inputs_embeds: [B, seq_len, hidden_size]
        # outputs: [B, seq_len, hidden_size]
        outputs = self.model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            kv_cache = kv_cache,
        )

        hidden_states = outputs
        
        # Move hidden_states to lm_head device (for multi-GPU)
        # Language modeling head may be on a different GPU
        lm_head_device = next(self.lm_head.parameters()).device
        hidden_states = hidden_states.to(lm_head_device)
        
        # Project to vocabulary logits
        # [B, seq_len, hidden_size] -> [B, seq_len, vocab_size]
        logits = self.lm_head(hidden_states)
        
        # Convert to float32 for numerical stability
        # Important for loss computation and sampling
        logits = logits.float()

        return_data = {
            'logits': logits,
        }
        
        # Return updated KV cache if provided
        # This allows the caller to reuse cached K/V for next token generation
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data