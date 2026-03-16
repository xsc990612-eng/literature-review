# Idea 2: SparsePagedAttention (SPA)

## Strategy
**Simplification** - Combine Pre-hoc Sparsity with PagedAttention to reduce both compute and memory

## Core Concept
Integrate Pre-hoc Sparsity's KV selection directly into PagedAttention's block-based memory management. Only store and attend to KV pairs that pass the pre-hoc selection threshold.

## Mathematical Formulation

### Pre-hoc Selection (from 2602.08329)
```
τ_pre(q') ≥ τ*(q') - β_th(τ)
where β_th(τ) ≤ (2K_max/√d)√(2-2τ)
```

### PagedAttention Blocks (from 2309.06180)
```
KV_cache = [Block_1, Block_2, ..., Block_n]
each Block contains K tokens
```

### SPA Integration
Instead of storing all K tokens per block, store only tokens where:
```
SelectionScore(token_i) > threshold
```

Block structure becomes:
```
SparseBlock {
  tokens: [token_id],  // variable length
  kv_vectors: [K_i, V_i],  // only selected tokens
  bitmap: [0,1,0,1,1,...]  // which positions kept
}
```

## Key Innovation
Pre-hoc Sparsity currently reduces compute but still stores full KV cache. SPA reduces both compute AND memory by not storing dropped KV pairs.

## Expected Benefits
1. **Memory**: 90%+ KV cache reduction (Pre-hoc) × block sharing (Paged)
2. **Compute**: Only attend to selected tokens
3. **Theory**: Preserves Pre-hoc's mutual information guarantees
4. **Serving**: Enables longer context windows

## Implementation
```python
# Modified PagedAttention kernel
for block in kv_cache:
    # Load only selected KV pairs
    selected_k, selected_v = block.sparse_load()
    
    # Compute attention only on selected
    scores = q @ selected_k.T
    out = softmax(scores) @ selected_v
    
    # Scatter back to full sequence
    output.scatter(block.bitmap, out)
```

## Validation Plan
- Baseline: vLLM PagedAttention
- Metrics: Throughput (tok/s), memory usage, perplexity on long documents
- Context lengths: 4K, 16K, 64K, 128K

## Citations
- Pre-hoc Sparsity (2602.08329): Selection algorithm with theory
- PagedAttention (2309.06180): Block-based memory management

## Risk
Bitmap overhead for sparse blocks; need to optimize block size vs sparsity.

