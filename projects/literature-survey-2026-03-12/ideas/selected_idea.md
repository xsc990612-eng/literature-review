# Selected Idea: SparsePagedAttention (SPA)

## Overview
Integrate Pre-hoc Sparsity's KV selection into PagedAttention's block-based memory management for simultaneous compute and memory reduction.

## Detailed Architecture

### Data Structure: SparseKVBlock
```python
@dataclass
class SparseKVBlock:
    # Variable-length storage
    k_vectors: Tensor[dynamic_n, head_dim]  # only selected tokens
    v_vectors: Tensor[dynamic_n, head_dim]
    
    # Metadata for reconstruction
    bitmap: Bitmask[block_size]  # which positions are kept
    original_positions: Int[dynamic_n]  # original token positions
    selection_scores: Float[dynamic_n]  # Pre-hoc scores for debugging
```

### Pre-hoc Selection Integration
```python
def select_kv_tokens(query, kv_block, threshold):
    """
    From Pre-hoc Sparsity (2602.08329)
    τ_pre(q') ≥ τ*(q') - β_th(τ)
    """
    # Compute attention scores for all tokens in block
    scores = query @ kv_block.k_vectors.T
    
    # Apply theoretical bound
    selection_mask = scores > compute_threshold(query, kv_block)
    
    return selection_mask
```

### Modified Attention Kernel
```python
def sparse_paged_attention(query, kv_cache_blocks):
    output = zeros_like(query)
    
    for block in kv_cache_blocks:
        # Load only selected KV pairs
        selected_k = block.k_vectors  # already filtered
        selected_v = block.v_vectors
        
        # Compute attention
        attn_scores = query @ selected_k.T / sqrt(dim)
        attn_weights = softmax(attn_scores)
        
        # Weighted sum
        block_output = attn_weights @ selected_v
        
        # Scatter to output positions
        output[block.original_positions] += block_output
    
    return output
```

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Block size | 16-128 | Tradeoff: smaller = finer sparsity, larger = less overhead |
| Sparsity target | 80-95% | From Pre-hoc paper: >90% with guarantees |
| Threshold τ | 0.9 | Balances quality vs compression |

## Training/Calibration

SPA requires no training - it's a drop-in replacement for PagedAttention:
1. Compute Pre-hoc thresholds once per model
2. Replace KV cache storage
3. Use during inference

## Memory Analysis

### Standard PagedAttention
```
Memory = batch_size × seq_len × num_heads × head_dim × 2 (K+V) × 2 bytes (fp16)
```

### SparsePagedAttention
```
Memory = batch_size × seq_len × sparsity_ratio × num_heads × head_dim × 2 × 2
       + bitmap_overhead

With 90% sparsity: 10× memory reduction
```

### Bitmap Overhead
- Block size 128: 128 bits = 16 bytes per block
- Sequence 16K: 128 blocks → 2KB overhead (negligible)

## Implementation Roadmap

### Phase 1: CPU Prototype (2 weeks)
- Implement SparseKVBlock data structure
- Modify attention computation
- Validate correctness on synthetic data

### Phase 2: GPU Kernel (4 weeks)
- CUDA kernel for sparse attention
- Optimize memory coalescing
- Benchmark vs vLLM baseline

### Phase 3: Integration (2 weeks)
- Integrate with vLLM serving engine
- Add configuration options
- Production testing

## Evaluation Plan

### Metrics
1. **Throughput**: tokens/second at various batch sizes
2. **Memory**: Peak KV cache usage
3. **Quality**: Perplexity on long documents (up to 128K)
4. **Latency**: Time to first token (TTFT)

### Baselines
- vLLM PagedAttention (full KV)
- vLLM + H2O (heuristic compression)
- vLLM + StreamingLLM (windowed)

### Datasets
- LongChat (32K context)
- L-Eval (document QA)
- InfiniteBench (100K+ context)

## Expected Results

| Metric | PagedAttention | SPA | Improvement |
|--------|----------------|-----|-------------|
| KV Memory | 100% | 10% | **10×** |
| Throughput | 100 tok/s | 180 tok/s | **1.8×** |
| Perplexity | 12.5 | 12.7 | **+1.6%** |
| Max Context | 32K | 128K | **4×** |

## Citations

- **Pre-hoc Sparsity** (2602.08329): Selection algorithm with theoretical guarantees
  ```
  Gao et al., "Near-Oracle KV Selection via Pre-hoc Sparsity", IEEE TPAMI 2026
  ```

- **PagedAttention** (2309.06180): Block-based memory management
  ```
  Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention", SOSP 2023
  ```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Bitmap overhead too high | Use larger blocks (256-512) |
| Variable-length tensor inefficiency | Pad to 8-token boundaries |
| Selection quality drops | Dynamic threshold adjustment |
| Kernel complexity | Start with Triton, optimize to CUDA |

## Future Extensions

1. **Dynamic block size**: Larger blocks for early layers, smaller for later
2. **Hierarchical sparsity**: Block-level + token-level selection
3. **Quantized SPA**: Combine with 4-bit KV quantization

