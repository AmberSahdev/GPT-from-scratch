from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

"""
Glossary
---
- B (batch size): Number of history windows processed together.
- T (timesteps): Number of observations in each history window.
- C (embedding dimension): Number of learned features per observation.

- Embedding: Learned vector representation of an observation.
- Head size: Dimensions in one attention head's query, key, value, and output.

- Causal mask: Prevents a timestep from attending to future timesteps.

- Softmax: Converts scores into nonnegative weights that sum to one.
- LayerNorm: Normalizes each timestep's features, then applies learned scale and bias.

- Residual connection: Adds a layer's input to its output to preserve information.
- Feedforward: Network that processes each timestep independently.
- ReLU: Nonlinear activation that replaces negative values with zero.
- Logits: Raw action scores before converting them into probabilities.
"""

@dataclass
class ModelConfig:
    context_length: int =  8  # Maximum number of observations in a history window (not batch size)
    embedding_dim: int = 64  # Learned features per observation (width)
    heads: int = 4  # Parallel attention heads. width must divide by heads, so in our case each head will output 16 numbers
    layers: int = 2  # Number of sequential transformer blocks


class Head(nn.Module):
    """One head of causal self-attention"""

    def __init__(self, config, head_size):
        super().__init__()
        # head_size = embedding_dim/heads 

        # Query: what this position seeks. 
        # Key: what each position matches.
        # Value: the information a position contributes when attended to.
        self.key = nn.Linear(config.embedding_dim, head_size, bias=False) # y = Wx + b
        self.query = nn.Linear(config.embedding_dim, head_size, bias=False)
        self.value = nn.Linear(config.embedding_dim, head_size, bias=False) 
        # Each Linear applies independently to the last axis of (B, T, width).  
        # Apply the same learned 64 → head_size projection to each timestep.
        # Keep the batch and time dimensions unchanged.

        # A lower-triangular mask permits only present/past positions.
        # A buffer moves with the model and is saved, but the optimizer does not train it.
        # Borrowed from 3b1b
        # self.tril 
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    config.context_length, config.context_length, dtype=torch.bool
                )
            ),
        )

    def forward(self, x):
        # B = independent examples, T = observations, C = features per observation.
        batch, time, channels = x.shape

        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Compare each query with every key: (B, T, head_size) @ (B, head_size, T).
        weights = q @ k.transpose(-2, -1)  # (B, T, T)

        # Scale by sqrt(head_size) so large feature counts do not inflate the scores.
        weights = weights * k.shape[-1] ** -0.5

        # Future positions get -infinity, which becomes probability zero after softmax.
        weights = weights.masked_fill(~self.tril[:time, :time], float("-inf"))

        # Normalize across the available source positions for each query position (along the last dimension)
        weights = F.softmax(weights, dim=-1)

        # Weighted sum of value vectors: each position receives information from its past.
        return weights @ v  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Run several Head objects in parallel, concatenate, then mix their features."""

    def __init__(self, config):
        super().__init__()

        if config.embedding_dim % config.heads:
            raise ValueError("embedding_dim width must be divisible by heads")
        head_size = config.embedding_dim // config.heads  # Example: 64 / 4 = 16 features.

        # ModuleList registers all heads so their weights are trained and saved.
        self.heads = nn.ModuleList(
            [Head(config, head_size) for _ in range(config.heads)]
        )

        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, x):
        # Each head returns (B, T, head_size), concatenate along the feature axis.
        concatenated = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, width)
        
        return self.proj(concatenated) # TODO: Excluded dropout for now, to accomodate for my RL plans


class FeedForward(nn.Module):
    """Process features independently at each timestep after attention."""

    def __init__(self, config):
        super().__init__()
        # Expand features, apply a nonlinearity, then shrink back to the model width.
        # ReLU allows the network to learn nonlinear relationships in the observations.
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim), # 64 -> 256 
            nn.ReLU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim), # 256 -> 64
        )

    def forward(self, x):
        return self.net(x) # (B, T, width)


class TransformerBlock(nn.Module):
    """Attention communicates between positions; feedforward processes each position."""

    def __init__(self, config):
        super().__init__()
        self.selfattention = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)

        # Normalize the feature axis before each sublayer (pre-normalization).
        self.ln1 = nn.LayerNorm(config.embedding_dim) # output = normalized_x * gamma + beta
        self.ln2 = nn.LayerNorm(config.embedding_dim)

    def forward(self, x):
        # Residual connection - additions preserve the original features alongside each learned update
        x = x + self.selfattention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-style arrangement of embedding layers, transformer blocks"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        if min(config.context_length, config.embedding_dim, config.heads, config.layers) < 1:
            raise ValueError("All architecture settings must be positive")
        
        # Unlike text GPT's token lookup, project eight numerical observations into features
        # because our inputs are vectors of size 8 from the sim environment of LunarLander 
        self.state_embedding = nn.Linear(8, config.embedding_dim)
        # State embedding represents the content: what was the lander doing?

        # One learned vector per history position tells attention the observation order.
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        # Position embedding represents its place: where is this observation in our eight-step history?

        # Transformer Blocks
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.layers)])

        self.ln_final = nn.LayerNorm(config.embedding_dim)

        # Four logits score the engine commands; they are not probabilities yet.
        self.action_head = nn.Linear(config.embedding_dim, 4) # 64 -> 4

        # Apply initialization to every registered Linear/Embedding layer.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) # Small random weights let different units learn different roles.
            
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_features(self, states):
        # run the observations through the transformer and returns their learned representations, before the action head
        # Input: (B, T, 8)
        batch, time, observations = states.shape

        if observations != 8 or not 1 <= time <= self.config.context_length:
            raise ValueError(
                "Expected (batch, time, 8) with time within context_length"
            )
        
        positions = torch.arange(time, device=states.device)
        # Position features (T, width) broadcast across the batch axis.
        
        x = self.state_embedding(states) + self.position_embedding(positions)
        x = self.transformer_blocks(x)
        x = self.ln_final(x)

        return x  # (B, T, width)

    def forward(self, states):
        # The last position represents the newest state and its available history.
        last_features = self.get_features(states)[:, -1, :]  # (B, width)
        logits = self.action_head(last_features) # (B, 4)
        return logits
