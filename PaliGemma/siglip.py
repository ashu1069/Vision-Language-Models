from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768, # size of the embedding vector of the ViT
        intermediate_size=3072, # size of the linear layer used in the FFN
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3, # RGB
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int=None,
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
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding = 0  # "valid" padding means no padding
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [B, C, H, W]
        
        patch_embeds = self.patch_embedding(pixel_values) # [B, C, H, W] -> [B, embed_dim, num_patches_H, num_patches_W]

        embeddings = patch_embeds.flatten(2) # [B, embed_dim, num_patches_H, num_patches_W] -> [B, embed_dim, num_patches]

        embeddings = embeddings.transpose(1, 2) # [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]

        embeddings = embeddings + self.position_embedding(self.position_ids) # adding positional embedding to each patch

        return embeddings

class SiglipAttention(nn.Module):
    '''MHA from Transformer paper'''
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # 1/sqrt(self.head_sim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim) # W_k
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim) # W_q
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim) # W_v
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
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size) # intermediate_size = 4*hidden_size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, n_patches, embed_dim] -> [B, n_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')

        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: [B, n_patches, embed_dim]
        residual = hidden_states
        # [B, n_patches, embed_dim] -> [B, n_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)

        # Self attention block
        # [B, n_patches, embed_dim] -> [B, n_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)

        # residual connection
        hidden_states = residual + hidden_states 

        hidden_states = self.layer_norm2(hidden_states)

        #MLP -> capturing nonlinearity and increasing the degree of freedom for learning
        hidden_states = self.mlp(hidden_states)
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
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config=config
        embed_dim=config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W] -> [B, n_patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

class SiglipVisionModel(nn.Module):
    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config=config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [B, C, H, W] -> [B, n_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values)