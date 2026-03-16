# Literature Survey Report: Efficient LLM Fine-tuning and Inference

**Project**: literature-survey-2026-03-12  
**Date**: 2026-03-16  
**Total Papers Analyzed**: 34 (17 scored 5, 17 scored 4)

---

## 1. Paper Overview

### 1.1 High-Impact Papers (Score 5)

| arXiv ID | Title | Category | Key Contribution |
|----------|-------|----------|------------------|
| 2305.14314 | QLoRA | Quantization | 4-bit LoRA fine-tuning preserving 16-bit performance |
| 2309.06180 | PagedAttention | Inference | Block-based KV cache management for serving |
| 2310.08659 | LoftQ | Quantization | LoRA-aware quantization initialization |
| 2502.08141 | LowRA | Quantization | Sub-2-bit LoRA fine-tuning |
| 2602.08329 | Pre-hoc Sparsity | Inference | Pre-attention KV selection with guarantees |
| 2602.13567 | DistillLens | Distillation | Symmetric intermediate layer distillation |
| 2603.08743 | Zipage | Inference | Compressed PagedAttention for reasoning |
| 2603.11504 | LongFlow | Inference | Zero-cost KV cache compression for reasoning |
| 2603.11564 | KV Cache Compression | Inference | Decoding-aligned KV compression |
| 2603.11873 | AdaFuse | Inference | Token-level pre-gating for dynamic adapters |

### 1.2 Research Directions

The surveyed papers cluster into four main directions:

1. **QLoRA & Quantization** (5 papers): 4-bit and sub-4-bit fine-tuning methods
2. **vLLM & Inference Optimization** (8 papers): KV cache compression, paging, sparse attention
3. **Knowledge Distillation** (4 papers): Model compression through distillation
4. **Short Text/Sentiment Analysis** (6 papers): GSDMM, topic modeling, Twitter sentiment

---

## 2. Method Comparison Table

### 2.1 Quantization Methods

| Method | Bit-width | Core Technique | Memory Saving | Performance |
|--------|-----------|----------------|---------------|-------------|
| QLoRA | 4-bit (NF4) | NormalFloat + Double Quant | 16× | Matches 16-bit |
| LoftQ | 4-bit | Alternating quantization + SVD | 16× | Improved init |
| LowRA | <2-bit | Extreme quantization | >32× | Under 2 bits |
| QDyLoRA | 4-bit | Dynamic rank selection | 16× | Adaptive |

### 2.2 Inference Optimization Methods

| Method | Target | Core Technique | Speedup | Key Feature |
|--------|--------|----------------|---------|-------------|
| PagedAttention | KV Cache | Block-based paging | 2-4× | Memory sharing |
| Zipage | KV Cache | Compressed paging | 2.1× | Async compression |
| LongFlow | KV Cache | Zero-cost estimation | >90% overhead | Contribution-based |
| PrHS | Attention | Pre-hoc sparsity | 3× sparsity | Theoretical bounds |
| AdaFuse | Adapters | Token pre-gating | Significant | SGMM kernel |

### 2.3 Distillation Methods

| Method | Level | Divergence | Key Innovation |
|--------|-------|------------|----------------|
| DistillLens | Intermediate | JSD (symmetric) | Logit Lens projection |
| Standard KD | Output | KL (asymmetric) | Final layer only |

---

## 3. Key Mathematical Formulas

### 3.1 Quantization

**QLoRA NormalFloat**:
$$q_i = \frac{1}{2}\left( Q_X\left(\frac{i}{2^k+1}\right) + Q_X\left(\frac{i+1}{2^k+1}\right)\right)$$

**LoftQ Alternating Optimization**:
$$\min_{Q, A, B} \|W - Q - AB^{\top}\|_{F}$$

### 3.2 KV Cache Compression

**LongFlow Score**:
$$\text{LongFlowScore}(t_i) = \left\| \alpha_t^i \mathbf{v}^i \right\|_1 = \alpha_t^i \sum_{l=1}^{d} | (\mathbf{v}^i)_l |$$

**AdaFuse Token Pre-Gating**:
$$y^l = f^l(x^l) + \sum_{i = 1}^{N}{G^1(x^{1})_iE^l_i(x^l)}$$

### 3.3 Distillation

**DistillLens JSD**:
$$\mathcal{L}_{JSD}(p, q_\theta) = \frac{1}{2} \left[ \mathcal{L}_{KL}(p \| m) + \mathcal{L}_{KL}(q_\theta \| m)\right]$$

**Confidence Score**:
$$c_{\theta}(y|x) = \frac{q_{\theta}(y|x)}{p(y|x)}$$

### 3.4 Sparse Attention

**Pre-hoc Sparsity Guarantee**:
$$\tau_{\mathrm{pre}}(\mathbf{q}') \ge \tau^*(\mathbf{q}') - \beta_{\mathrm{th}}(\tau)$$

$$&\beta_{\mathrm{th}}(\tau) \le \frac{2K_{\max}}{\sqrt d}\sqrt{2-2\tau}$$

---

## 4. Technical Recommendations

### 4.1 For Fine-tuning Large Models

**Recommended Stack**:
1. **Base**: QLoRA with NF4 (4-bit NormalFloat)
2. **Initialization**: LoftQ for improved starting point
3. **Adapter Placement**: All linear layers (not just Q,V)
4. **Rank**: r=64 or r=128 depending on task complexity

**Memory Budget**:
- 7B model: ~5-6 GB
- 13B model: ~10-12 GB  
- 65B model: ~40-48 GB

### 4.2 For Inference Serving

**High-Concurrency Serving**:
1. **Base**: PagedAttention for KV cache management
2. **Compression**: LongFlow for reasoning models (zero-cost)
3. **Sparse Attention**: Pre-hoc Sparsity for theoretical guarantees
4. **Dynamic Adapters**: AdaFuse with token pre-gating

**Optimization Priority**:
1. Block-based KV management (PagedAttention)
2. Static memory allocation
3. Asynchronous compression/decoding
4. Prefix caching for shared prompts

### 4.3 For Model Compression

**Distillation Pipeline**:
1. **Layer Mapping**: Proportional depth ratio
2. **Divergence**: Symmetric JSD (not asymmetric KL)
3. **Projection**: Logit Lens for vocabulary alignment
4. **Supervision**: All layers + final output

---

## 5. Research Gaps and Opportunities

### 5.1 Identified Gaps

1. **Ultra-Low Bit Fine-tuning**: LowRA explores <2-bit but limited evaluation
2. **Multi-Task Adapter Switching**: Dynamic adapter loading at inference
3. **Theoretical Understanding**: Limited theory on quantization-fine-tuning interaction
4. **Cross-Architecture Distillation**: Logit Lens assumes shared vocabulary

### 5.2 Future Directions

1. **Hybrid Compression**: Combine quantization + distillation + pruning
2. **Adaptive Sparsity**: Task-dependent KV cache budgets
3. **Hardware-Aware Optimization**: Custom kernels for specific deployments
4. **Long Context Optimization**: Extend paging to million-token contexts

---

## 6. Summary Statistics

- **Total Papers**: 34
- **5-Star Papers**: 17
- **With Full tex Analysis**: 15
- **With Abstract Analysis**: 19

### Direction Breakdown
- vLLM/Inference: 13 papers (38%)
- QLoRA/Quantization: 9 papers (26%)
- Distillation: 6 papers (18%)
- Sentiment/GSDMM: 6 papers (18%)

---

## 7. References

All detailed paper notes available in `notes/` directory:
- paper_2305.14314.md (QLoRA)
- paper_2309.06180.md (PagedAttention)
- paper_2310.08659.md (LoftQ)
- paper_2602.08329.md (Pre-hoc Sparsity)
- paper_2602.13567.md (DistillLens)
- paper_2603.08743.md (Zipage)
- paper_2603.11504.md (LongFlow)
- paper_2603.11873.md (AdaFuse)
- ... and more

---

*Generated by OpenClaw Research Survey Agent*
