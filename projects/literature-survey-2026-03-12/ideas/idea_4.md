# Idea 4: Token-LoRA Fusion (ToFu)

## Strategy
**Architecture Innovation** - Merge AdaFuse's token gating with LoRA for fine-grained adapter selection

## Core Concept
Instead of applying the same LoRA adapter to all tokens, ToFu uses a learned gating network to select which adapter (or combination) to apply per token.

## Mathematical Formulation

### Standard LoRA (from 2305.14314)
```
h = Wx + (A @ B)x
```

### AdaFuse Pre-Gating (from 2603.11873)
```
y^l = f^l(x^l) + Σ_i G^1(x^1)_i * E^l_i(x^l)
```

### ToFu Fusion
For N adapters and input token x_t:

```
g_t = Softmax(W_g @ x_t)  # gating weights, dim N
h_t = Wx_t + Σ_i g_t[i] * (A_i @ B_i) @ x_t
```

### Gating Network
```
g_t = MLP(x_t)  # or attention-based gating
```

## Key Innovation
Current methods (AdaFuse included) apply adapter at sequence or layer level. ToFu operates at **token level**, allowing different tokens to use different expertise.

## Architecture
```
Input Token
    ↓
[Gating Network]
    ↓
[Adapter 1] ← weight 0.1
[Adapter 2] ← weight 0.7  ← selected for this token
[Adapter 3] ← weight 0.2
    ↓
[Weighted Fusion]
    ↓
Output
```

## Applications
1. **Multi-domain**: Different tokens need different domain knowledge
2. **Multi-lingual**: Language-specific adapters per token
3. **Multi-task**: Task-specific routing during inference

## Training Strategy
```
Phase 1: Train adapters separately per domain/task
Phase 2: Freeze adapters, train gating network
Phase 3: Joint fine-tuning with small LR
```

## Validation Plan
- Setup: 3-domain setting (news, medical, code)
- Baselines: Single adapter, adapter ensemble, AdaFuse
- Metrics: Per-domain accuracy, gating entropy (measure of specialization)

## Citations
- AdaFuse (2603.11873): Token-level gating concept
- LoRA (2305.14314, actually Hu 2021): Low-rank adaptation
- QLoRA (2305.14314): Quantized training

## Risk
Gating may collapse to single adapter; need entropy regularization.

