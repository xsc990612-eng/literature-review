# Taxonomy of Efficient LLM Techniques

## Overview

```
Efficient LLM Methods
├── Training Efficiency
│   ├── Parameter-Efficient Fine-tuning (PEFT)
│   │   ├── LoRA Family
│   │   │   ├── LoRA (Low-Rank Adaptation)
│   │   │   ├── QLoRA (Quantized LoRA)
│   │   │   ├── LoftQ (LoRA-aware Quantization)
│   │   │   └── LowRA (Sub-2-bit LoRA)
│   │   └── Adapter Methods
│   │       └── AdaFuse (Token-level Pre-gating)
│   └── Quantization
│       ├── 4-bit Methods
│       │   ├── NF4 (NormalFloat)
│       │   └── FP4
│       └── Extreme Quantization
│           └── Leech Lattice VQ (<2-bit)
│
├── Inference Efficiency
│   ├── KV Cache Optimization
│   │   ├── Paging Methods
│   │   │   ├── PagedAttention
│   │   │   └── Zipage (Compressed)
│   │   ├── Compression
│   │   │   ├── LongFlow (Zero-cost)
│   │   │   └── Decoding-aligned
│   │   └── Sparse Selection
│   │       ├── LookaheadKV
│   │       └── Pre-hoc Sparsity (Theoretical)
│   ├── Attention Optimization
│   │   ├── Sparse Attention
│   │   │   └── IndexCache
│   │   └── Flash Attention
│   └── Dynamic Methods
│       └── AdaFuse (Token gating)
│
├── Model Compression
│   ├── Knowledge Distillation
│   │   ├── Output-level
│   │   │   └── Hinton KD (Softmax temperature)
│   │   ├── Intermediate-level
│   │   │   ├── FitNets (Hint-based)
│   │   │   └── DistillLens (Logit Lens + JSD)
│   │   └── Multi-stage
│   │       └── TinyBERT
│   └── Pruning + Distillation
│       └── Bielik-Minitron
│
└── Application Domains
    ├── Short Text Processing
    │   ├── Topic Modeling
    │   │   ├── GSDMM (Gibbs Sampling)
    │   │   └── Gamma-Poisson Mixture
    │   └── Sentiment Analysis
    │       └── SemEval Benchmarks
    └── Reasoning Models
        ├── KV Cache for Long Context
        └── Chain-of-Thought Optimization
```

## Dimension 1: Optimization Stage

### Training-time Methods
- **Goal**: Reduce memory during fine-tuning
- **Approach**: Quantization + Low-rank adaptation
- **Key Methods**: QLoRA, LoftQ, LowRA

### Inference-time Methods
- **Goal**: Reduce latency and memory during serving
- **Approach**: KV cache optimization, sparse attention
- **Key Methods**: PagedAttention, LongFlow, Pre-hoc Sparsity

### Compression Methods
- **Goal**: Smaller deployable models
- **Approach**: Distillation, pruning, quantization
- **Key Methods**: DistillLens, TinyBERT, Bielik-Minitron

## Dimension 2: Resource Target

### Memory Reduction
- **QLoRA**: 16× reduction for training
- **PagedAttention**: Dynamic allocation for serving
- **LongFlow**: 90%+ KV cache compression

### Compute Reduction
- **Pre-hoc Sparsity**: 3× attention sparsity
- **AdaFuse**: Token-level gating reduces FLOPs
- **FlashInfer**: Optimized kernels

### Parameter Reduction
- **LoRA**: Only train <1% parameters
- **Distillation**: 7.5× model size reduction

## Dimension 3: Theoretical Foundation

### Empirical Methods
- QLoRA (empirically designed NF4)
- PagedAttention (OS-inspired paging)
- TinyBERT (heuristic layer mapping)

### Theory-backed Methods
- **Pre-hoc Sparsity**: Mutual information bounds
- **LoftQ**: Alternating optimization convergence
- **DistillLens**: Symmetric divergence theory

### Hybrid Methods
- **LongFlow**: Contribution-based (empirical) + zero-cost (theoretical)
- **AdaFuse**: Learned gating (empirical) + SGMM kernel (theoretical)

## Key Relationships

```
QLoRA ──► LoftQ (improved initialization)
  │
  ▼
LowRA (extreme quantization)

PagedAttention ──► Zipage (compressed)
       │
       ▼
   LongFlow (reasoning-optimized)

Hinton KD ──► FitNets (intermediate)
    │
    ▼
DistillLens (symmetric + Logit Lens)
```

## Research Evolution Path

1. **2014-2015**: Foundation (FitNets, Hinton KD)
2. **2018-2020**: BERT Era (TinyBERT, DistilBERT)
3. **2021**: LoRA introduction
4. **2023**: Efficiency surge (QLoRA, PagedAttention)
5. **2025-2026**: Advanced methods (Pre-hoc, DistillLens, LongFlow)

