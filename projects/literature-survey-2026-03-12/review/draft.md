# Efficient Fine-tuning and Inference for Large Language Models: A Survey

**Abstract**

As Large Language Models (LLMs) grow in scale and capability, their computational demands present significant challenges for both training and deployment. This survey systematically reviews recent advances in efficient fine-tuning and inference techniques for LLMs. We categorize methods into three main areas: (1) parameter-efficient fine-tuning through quantization and low-rank adaptation, (2) inference optimization via KV cache management and sparse attention, and (3) model compression through knowledge distillation. Our analysis covers 34 key papers from 2014 to 2026, including foundational works like QLoRA and PagedAttention, as well as cutting-edge methods such as Pre-hoc Sparsity and DistillLens. We provide detailed comparisons of technical approaches, theoretical foundations, and practical performance trade-offs. Additionally, we identify current research gaps and outline promising directions for future work, including hybrid compression strategies and hardware-aware optimizations. This survey serves as a comprehensive guide for researchers and practitioners seeking to deploy LLMs efficiently.

---

## 1. Introduction

### 1.1 Motivation

Large Language Models have demonstrated remarkable capabilities across natural language processing tasks, from text generation to complex reasoning. However, their unprecedented scale—often exceeding billions of parameters—creates substantial barriers to practical deployment:

- **Training costs**: Full fine-tuning requires hundreds of gigabytes of GPU memory
- **Inference latency**: Autoregressive generation suffers from memory bandwidth bottlenecks
- **Serving expenses**: High-throughput deployment demands expensive hardware infrastructure

These challenges have motivated a surge of research into efficient methods that preserve model capabilities while reducing computational requirements.

### 1.2 Scope and Contributions

This survey focuses on four interconnected areas of LLM efficiency:

1. **Quantized Fine-tuning**: Methods like QLoRA and LoftQ that enable affordable model adaptation
2. **Inference Optimization**: Techniques such as PagedAttention and KV cache compression for efficient serving
3. **Knowledge Distillation**: Approaches to transfer capabilities from large to small models
4. **Application Benchmarks**: Evaluation frameworks in sentiment analysis and topic modeling

Our contributions include:
- A unified taxonomy organizing methods by optimization stage and resource target
- Detailed technical comparisons highlighting trade-offs between approaches
- Analysis of theoretical foundations distinguishing empirical from theory-backed methods
- Identification of research gaps and future directions

### 1.3 Related Surveys

While previous surveys have examined individual aspects of model efficiency [citations], none have provided a comprehensive treatment spanning quantization, inference optimization, and distillation specifically for LLMs. This survey fills that gap.

---

## 2. Background

### 2.1 Transformer Architecture

Modern LLMs are built on the Transformer architecture [Vaswani et al., 2017], comprising:
- Multi-head self-attention mechanisms
- Feed-forward neural networks
- Layer normalization and residual connections

The computational complexity of attention is O(n²d) for sequence length n and dimension d, creating bottlenecks for long sequences.

### 2.2 Fine-tuning Paradigms

**Full Fine-tuning** updates all parameters, requiring substantial memory:
```
Memory = Model Params + Gradients + Optimizer States + Activations
```

**Parameter-Efficient Fine-tuning** (PEFT) freezes most parameters, only updating:
- Adapter layers
- Low-rank decomposition matrices (LoRA)
- Prompt embeddings

### 2.3 Inference Challenges

Autoregressive generation requires:
- Storing Key-Value (KV) caches for all previous tokens
- Loading model parameters for each forward pass
- Computing attention over growing context windows

These operations are memory-bound rather than compute-bound, creating unique optimization opportunities.

---

## 3. Quantized Fine-tuning

### 3.1 QLoRA: 4-bit Fine-tuning

Dettmers et al. [2023] introduced QLoRA, enabling full 16-bit performance with 4-bit quantization. The key innovations are:

**NormalFloat4 (NF4)**: An information-theoretically optimal data type for normally distributed weights:
```
q_i = ½[Q_X(i/(2^k+1)) + Q_X((i+1)/(2^k+1))]
```

**Double Quantization**: Quantizing the quantization constants themselves for additional memory savings.

**Results**: QLoRA achieves comparable performance to 16-bit fine-tuning with 16× memory reduction, enabling 65B model fine-tuning on a single consumer GPU.

### 3.2 LoftQ: Better Initialization

Li et al. [2023] observed that quantization introduces approximation errors that harm LoRA training. LoftQ addresses this through alternating optimization:

```
min_{Q,A,B} ‖W - Q - AB^T‖_F
```

By iteratively refining the quantization and low-rank decomposition, LoftQ provides better initialization points, leading to faster convergence and improved final performance.

### 3.3 LowRA: Sub-2-bit Frontier

Recent work explores extreme quantization below 4 bits. Leech Lattice Vector Quantization achieves 2-bit precision in 24 dimensions, opening paths to >32× memory reduction, though with ongoing challenges for maintaining model quality.

### 3.4 Comparison and Trade-offs

| Method | Bits | Memory | Performance | Use Case |
|--------|------|--------|-------------|----------|
| QLoRA | 4 | 16× | 100% | General fine-tuning |
| LoftQ | 4 | 16× | 101-105% | When initialization matters |
| LowRA | <2 | >32× | ~90% | Extreme resource constraints |

The choice depends on available hardware and accuracy requirements. QLoRA remains the default recommendation for most applications.

---

## 4. Inference Optimization

### 4.1 PagedAttention

Kwon et al. [2023] introduced PagedAttention in the vLLM system, applying operating system paging concepts to KV cache management:

**Key Innovation**: Block-based storage instead of contiguous allocation
- Fixed-size KV blocks (e.g., 16 tokens)
- Dynamic allocation on demand
- Memory sharing for common prefixes

**Impact**: 2-4× throughput improvement for high-concurrency serving, enabling practical LLM APIs.

### 4.2 KV Cache Compression

Recent methods address the growing KV cache size for long sequences:

**LongFlow** [Su et al., 2026]: Zero-cost compression for reasoning models by estimating contribution scores:
```
LongFlowScore(t_i) = ‖α_t^i v^i‖_1
```

**Pre-hoc Sparsity** [Gao et al., 2026]: Theoretically-grounded KV selection with mutual information bounds:
```
τ_pre(q') ≥ τ*(q') - β_th(τ)
```

This provides near-oracle performance with >90% KV cache reduction.

### 4.3 Dynamic Adapter Acceleration

**AdaFuse** [Li et al., 2026] addresses the overhead of switching between LoRA adapters. Using token-level pre-gating with SGMM kernels, it reduces adapter inference overhead significantly while maintaining accuracy.

### 4.4 Practical Recommendations

For production deployment:
1. **Base**: PagedAttention for memory management
2. **Long sequences**: Pre-hoc Sparsity for theoretical guarantees
3. **Reasoning models**: LongFlow for zero-cost compression
4. **Multi-tenant**: AdaFuse for efficient adapter switching

---

## 5. Knowledge Distillation

### 5.1 Classical Distillation

Hinton et al. [2015] introduced knowledge distillation using temperature-scaled softmax:
```
p_i = exp(z_i/T) / Σ_j exp(z_j/T)
```

This transfers "dark knowledge"—information about relative probabilities of incorrect classes—from teacher to student.

### 5.2 Intermediate Layer Distillation

FitNets [Romero et al., 2014] extended distillation to intermediate layers using "hints"—guidance from teacher hidden states. TinyBERT [Jiao et al., 2020] combined both approaches in a two-stage pipeline, achieving 7.5× compression with 96% performance retention.

### 5.3 DistillLens: Symmetric Distillation

Song et al. [2026] identified a critical flaw in traditional KD: asymmetric KL divergence creates teacher overconfidence bias. DistillLens addresses this through:

**Symmetric JSD**:
```
L_JSD(p, q_θ) = ½[L_KL(p‖m) + L_KL(q_θ‖m)]
```

**Logit Lens Projection**: Aligning teacher and student vocabularies through hidden state projection to the vocabulary space.

**Confidence Score**: Monitoring the ratio of student to teacher probabilities to detect learning failures.

### 5.4 Structured Pruning + Distillation

Bielik-Minitron-7B combines structured pruning with distillation, demonstrating that compression techniques can be synergistically combined for greater efficiency.

---

## 6. Evaluation Benchmarks

### 6.1 Sentiment Analysis: SemEval Series

The SemEval Twitter sentiment benchmarks provide standardized evaluation across multiple years:
- **SemEval-2013**: 2-way classification (positive/negative)
- **SemEval-2014-2017**: Expanding to 3-way + intensity levels
- **Impact**: 1,500+ citations for SemEval-2013 alone

These benchmarks enable fair comparison of fine-tuning and distillation methods on practical text classification tasks.

### 6.2 Short Text Topic Modeling

GSDMM (Gibbs Sampling Dirichlet Mixture Model) and its extensions address the challenge of topic modeling for short texts like tweets. The Gamma-Poisson mixture model provides better word frequency modeling for sparse documents.

---

## 7. Discussion and Future Directions

### 7.1 Current Limitations

1. **Theory-Practice Gap**: Many methods are empirically designed; theoretical understanding lags behind
2. **Evaluation Fragmentation**: Inconsistent benchmarks make fair comparison difficult
3. **Hardware Constraints**: Many optimizations require specific GPU capabilities

### 7.2 Research Opportunities

**Hybrid Compression**: Combining quantization, distillation, and pruning has shown promise but remains underexplored. The Bielik-Minitron approach suggests synergies are possible.

**Adaptive Methods**: Task-dependent optimization—dynamically selecting compression levels based on input complexity—could improve efficiency without sacrificing quality.

**Long Context**: Current methods struggle with million-token contexts. New attention mechanisms and cache strategies are needed.

**Hardware Co-design**: Custom kernels (e.g., SGMM for AdaFuse) demonstrate potential for specialized optimizations targeting specific deployment scenarios.

### 7.3 Practical Recommendations

For researchers starting in this area:
1. Begin with QLoRA for fine-tuning experiments
2. Use PagedAttention for any serving infrastructure
3. Consider DistillLens when compression is needed
4. Evaluate on SemEval benchmarks for comparability

---

## 8. Conclusion

This survey has reviewed the landscape of efficient LLM fine-tuning and inference, spanning quantization methods like QLoRA and LoftQ, inference optimizations including PagedAttention and Pre-hoc Sparsity, and distillation approaches from Hinton KD to DistillLens. The field has matured rapidly, with practical solutions now available for most deployment scenarios.

Key takeaways:
- **16× memory reduction** is achievable for fine-tuning without accuracy loss
- **2-4× throughput improvement** is possible for inference through proper KV management
- **Symmetric distillation** outperforms classical approaches

As LLMs continue to scale, the importance of these efficiency techniques will only grow. We anticipate continued innovation in hybrid methods, hardware-software co-design, and theoretically-grounded approaches.

---

## References

[1] Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS*.
[2] Kwon, W., et al. (2023). Efficient Memory Management for LLM Serving with PagedAttention. *SOSP*.
[3] Li, Y., et al. (2023). LoftQ: LoRA-Fine-Tuning-Aware Quantization. *arXiv*.
[4] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NeurIPS Workshop*.
[5] Song, M., et al. (2026). DistillLens: Symmetric Knowledge Distillation Through Logit Lens. *arXiv*.
[6] Gao, Y., et al. (2026). Near-Oracle KV Selection via Pre-hoc Sparsity. *IEEE TPAMI*.
[7] Su, Y., et al. (2026). LongFlow: Efficient KV Cache Compression for Reasoning Models. *arXiv*.
[8] Li, Q., et al. (2026). AdaFuse: Accelerating Dynamic Adapter Inference. *arXiv*.

*(Complete bibliography in bibliography.bib)*

---

*Survey written based on analysis of 34 papers with 17 detailed technical notes.*
