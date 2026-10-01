I'm going to build GPT from scratch. 

I am going to repurpose it to play games in a simulation environment since I don't have the compute to train a language model locally. 

But it will still contain all the aspects of the GPT architecture: 
- Transformer
- Multiheaded Self-Attention
- Decoder

Since I'm going to be playing a game, it will actually be really fun to fine-tune using Reinforcement Learning as well.

---

# Design

```
INPUT: history of 8 observations
Each observation contains 8 physical measurements
Shape: (B, 8, 8)
         │
         ▼
STATE EMBEDDING: Linear(8 → 64)
Each observation becomes a 64-dimensional vector
         │
         ▼
ADD POSITION EMBEDDINGS
Shape: (B, 8, 64)
         │
         ▼
┌────────────── TRANSFORMER BLOCK 1 ──────────────┐
│                                                │
│  x ────────────────────────────────┐           │
│  │                                 │           │
│  ▼                                 │           │
│  LayerNorm                         │           │
│  │                                 │           │
│  ▼                                 │           │
│  MULTI-HEAD ATTENTION              │           │
│  Same 64 features go to EVERY head │           │
│  │                                 │           │
│  ├── Head 1 ──► 16 features         │           │
│  ├── Head 2 ──► 16 features         │           │
│  ├── Head 3 ──► 16 features         │           │
│  └── Head 4 ──► 16 features         │           │
│         │                          │           │
│         ▼                          │           │
│  Concatenate: 16 × 4 = 64           │           │
│         │                          │           │
│         ▼                          │           │
│  Projection: Linear(64 → 64)        │           │
│         │                          │           │
│         ▼                          │           │
│       ADD ◄────────────────────────┘           │
│         │              Residual connection    │
│         ├──────────────────────────┐           │
│         ▼                          │           │
│  LayerNorm                         │           │
│         │                          │           │
│         ▼                          │           │
│  FEEDFORWARD                       │           │
│  Linear(64 → 256)                  │           │
│         │                          │           │
│        ReLU                        │           │
│         │                          │           │
│  Linear(256 → 64)                  │           │
│         │                          │           │
│         ▼                          │           │
│       ADD ◄────────────────────────┘           │
│         │              Residual connection    │
│  Output: (B, 8, 64)                            │
└─────────┬─────────────────────────────────────┘
          │
          ▼
TRANSFORMER BLOCK 2
Same structure, its own learned weights
          │
          ▼
FINAL LAYERNORM
Shape: (B, 8, 64)
          │
          ▼
SELECT THE NEWEST TIMESTEP
Shape: (B, 64)
          │
          ▼
ACTION HEAD: Linear(64 → 4)
Shape: (B, 4)
          │
          ▼
Four action scores:
[nothing, left engine, main engine, right engine]
```

Inside each attention head:
```
Input: (B, 8, 64)
          │
     ┌────┼────────────────┐
     ▼    ▼                ▼
   Query  Key             Value
   64→16  64→16           64→16
     │    │                │
     └─┬──┘                │
       ▼                   │
  Q × Kᵀ / √16             │
       │                   │
  Mask future positions    │
       │                   │
    Softmax                │
       │                   │
  Attention weights        │
  Shape: (B, 8, 8)          │
       │                   │
       └─────── × ─────────┘
                │
                ▼
       Output: (B, 8, 16)
```

Total Params
```
State embedding          576
Position embedding       512
Two Transformer blocks   99,584
Final LayerNorm          128
Action head: 64×4 + 4    260
─────────────────────────────────
Total                    101,060
```
