# Method Comparison - Efficient LLM Techniques

## 1. Quantization Methods for Fine-tuning

| Method | Year | Bit-width | Core Technique | Memory Saving | Performance | Venue |
|--------|------|-----------|----------------|---------------|-------------|-------|
| QLoRA | 2023 | 4-bit (NF4) | NormalFloat + Double Quantization | 16× | Matches 16-bit | NeurIPS |
| LoftQ | 2023 | 4-bit | Alternating Quantization + SVD | 16× | Improved init | arXiv |
| LowRA | 2025 | <2-bit | Extreme Quantization | >32× | Under 2 bits | arXiv |

**Key Insight**: QLoRA's NF4 data type is information-theoretically optimal for normally distributed weights.

## 2. Inference Optimization Methods

| Method | Year | Target | Core Technique | Speedup | Key Feature | Venue |
|--------|------|--------|----------------|---------|-------------|-------|
| PagedAttention | 2023 | KV Cache | Block-based paging | 2-4× throughput | Copy-on-Write | SOSP |
| Zipage | 2026 | KV Cache | Compressed paging | 2.1× | Async compression | arXiv |
| LongFlow | 2026 | KV Cache | Zero-cost estimation | >90% overhead reduction | Contribution-based | arXiv |
| Pre-hoc Sparsity | 2026 | Attention | Pre-attention KV selection | 3× sparsity | Theoretical bounds | TPAMI |
| AdaFuse | 2026 | Adapters | Token pre-gating | Significant | SGMM kernel | arXiv |

**Key Insight**: PagedAttention's block-based management enables memory sharing; Pre-hoc Sparsity provides theoretical guarantees.

## 3. Knowledge Distillation Methods

| Method | Year | Level | Divergence | Key Innovation | Venue |
|--------|------|-------|------------|----------------|-------|
| Hinton KD | 2015 | Output | KL (asymmetric) | Temperature softmax | NeurIPS-W |
| FitNets | 2014 | Intermediate | MSE | Hint-based training | ICLR |
| TinyBERT | 2020 | Both | MSE + CE | Two-stage distillation | EMNLP |
| DistillLens | 2026 | Intermediate | JSD (symmetric) | Logit Lens projection | arXiv |
| Bielik-Minitron | 2026 | Both | KL + Pruning | Structured pruning + KD | arXiv |

**Key Insight**: DistillLens's symmetric JSD addresses teacher overconfidence; Logit Lens aligns vocabularies.

## 4. Topic Modeling for Short Text

| Method | Year | Core Technique | Application | Venue |
|--------|------|----------------|-------------|-------|
| GSDMM | 2012 | Gibbs Sampling | Movie Group Process | - |
| Gamma-Poisson | 2020 | Poisson word frequency | Short text topic | - |
| Enhanced Model | 2025 | LLM-enhanced clustering | Short text | arXiv |

## 5. Sentiment Analysis Benchmarks

| Dataset | Year | Task | Size | Venue |
|---------|------|------|------|-------|
| SemEval-2013 | 2013 | Twitter Sentiment | Multi-domain | NAACL |
| SemEval-2014 | 2014 | Twitter Sentiment | Expanded | SemEval |
| SemEval-2015 | 2015 | Multi-dimension | 3 subtasks | NAACL |
| SemEval-2016 | 2016 | Sentiment Intensity | 5-point scale | SemEval |
| SemEval-2017 | 2017 | Fine-grained | 3-point + intensity | SemEval |

---

## Performance Summary

### Fine-tuning Efficiency
- **QLoRA**: 16× memory reduction, comparable accuracy
- **LoftQ**: Better initialization, faster convergence
- **LowRA**: >32× reduction, <2-bit precision

### Inference Speedup
- **PagedAttention**: 2-4× throughput improvement
- **Pre-hoc Sparsity**: 90%+ KV cache reduction with guarantees
- **LongFlow**: Zero-cost compression for reasoning models

### Model Compression
- **DistillLens**: Symmetric divergence reduces teacher bias
- **TinyBERT**: 7.5× smaller, retains 96% performance

