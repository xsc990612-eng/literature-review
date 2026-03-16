# Idea 1: Quantized Distillation (QuDist)

## Strategy
**Combination** - Merge QLoRA's 4-bit quantization with DistillLens's symmetric distillation

## Core Concept
Train a 4-bit student model using symmetric JSD distillation from a full-precision teacher, with quantization-aware loss that accounts for both distillation error and quantization error.

## Mathematical Formulation

### Loss Function
```
L_total = L_JSD + λ_quant * L_quant + λ_rank * L_rank
```

**Symmetric Distillation (from DistillLens 2602.13567):**
```
L_JSD = ½[KL(p_teacher || m) + KL(p_student || m)]
where m = (p_teacher + p_student) / 2
```

**Quantization-Aware Term (novel):**
```
L_quant = ||W_student - Q(W_student)||²_F
```
This encourages the student to learn representations that are naturally quantization-friendly.

**Low-Rank Regularization (from LoftQ 2310.08659):**
```
L_rank = ||W - Q - AB^T||_F
```

### Key Innovation
Unlike sequential "distill then quantize" or "quantize then distill", QuDist jointly optimizes for both objectives during training.

## Expected Benefits
1. **Memory**: Student is 16× smaller than teacher (4-bit vs 16-bit)
2. **Speed**: No runtime quantization overhead
3. **Quality**: Symmetric JSD avoids teacher overconfidence bias
4. **Theory**: Can prove convergence bounds combining both error sources

## Validation Plan
- Teacher: LLaMA-2-7B (16-bit)
- Student: LLaMA-2-7B (4-bit)
- Baselines: QLoRA + post-hoc KD, TinyBERT-style 2-stage
- Metrics: Perplexity, downstream task accuracy, memory usage

## Citations
- DistillLens (2602.13567): Symmetric JSD formulation
- QLoRA (2305.14314): 4-bit NormalFloat quantization
- LoftQ (2310.08659): LoRA-aware initialization

## Risk
Quantization error and distillation error may compound; need careful λ tuning.

