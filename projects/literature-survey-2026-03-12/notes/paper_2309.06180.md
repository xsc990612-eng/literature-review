# Paper Analysis: PagedAttention (2309.06180)

## Metadata
- **arXiv ID**: 2309.06180
- **Title**: Efficient Memory Management for Large Language Model Serving with PagedAttention
- **Authors**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **Score**: 5/5
- **Category**: vLLM / Memory Management
- **Published**: 2023-09

## Abstract
High throughput serving of large language models requires batching many requests simultaneously. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows/shrinks dynamically.

## Core Method

### 1. PagedAttention Overview

**Problem**: KV cache memory management in LLM serving
- Large memory footprint per request
- Dynamic growth and shrinkage
- Internal and external fragmentation
- Inability to share KV cache across requests

**Solution**: Inspired by virtual memory paging in OS

### 2. Key Ideas

**A. Block-based KV Cache Management**
- KV cache divided into fixed-size blocks
- Each block contains key/value vectors for fixed number of tokens
- Non-contiguous physical storage

**B. Dynamic Block Allocation**
- Blocks allocated on-demand during generation
- Eliminates pre-allocation of max sequence length
- Reduces memory waste

**C. Block Table**
- Maps logical token positions to physical blocks
- Enables non-contiguous storage
- Per-request block table

**D. Memory Sharing**
- Copy-on-write for shared sequences
- Enables efficient parallel sampling and beam search
- Shared prompt KV cache across requests

### 3. Key Formulations

**Memory Usage**:
$$M_{\text{paged}} = \sum_{i=1}^{B} \lceil \frac{L_i}{B_{\text{size}}} \rceil \times M_{\text{block}}$$

Where:
- $B$: batch size
- $L_i$: sequence length of request $i$
- $B_{\text{size}}$: block size
- $M_{\text{block}}$: memory per block

**Vs Contiguous Allocation**:
$$M_{\text{contiguous}} = B \times L_{\text{max}} \times M_{\text{token}}$$

### 4. Key Operations

**Attention with Paged KV Cache**:
- Gather key/value vectors from non-contiguous blocks
- Efficient CUDA kernel for block gathering
- Memory access pattern optimized for GPU

**Block Allocation Policies**:
- First-fit, best-fit allocation strategies
- Block eviction for memory pressure
- Preemptive scheduling support

### 5. System Integration (vLLM)

**Continuous Batching**:
- Dynamic batch size adjustment
- Request arrival and completion handling
- Fair scheduling across requests

**Parallel Sampling**:
- Single prompt, multiple outputs
- Shared prompt KV cache
- Divergent generation paths

**Beam Search**:
- Shared beam history
- Copy-on-write for diverging beams
- Efficient memory usage

## Technical Contributions

1. **PagedAttention Algorithm**: Block-based KV cache management
2. **Memory Efficiency**: Eliminates fragmentation, enables sharing
3. **System Implementation**: vLLM inference engine
4. **Throughput Gains**: 2-4× throughput improvement over baseline
5. **Production Ready**: Widely adopted in production systems

## Evaluation Results
- 2-4× higher throughput than Orca
- Supports higher batch sizes
- Efficient memory utilization
- Validated on various model sizes

## Relationship to Our Research

### Relevance
This is the **foundational paper** for efficient LLM serving, establishing the paging paradigm for KV cache management.

### Key Insights
1. **OS paging analogy**: Virtual memory concepts apply to KV cache
2. **On-demand allocation**: Eliminates pre-allocation waste
3. **Block tables enable sharing**: Critical for parallel sampling
4. **System-level optimization**: Algorithm + implementation matter

### Potential Applications
- **Extended paging**: Larger blocks for longer contexts
- **Hierarchical paging**: Multi-level block hierarchy
- **Compression + Paging**: Integrate KV compression with paging

### Limitations to Consider
- Block size is fixed hyperparameter
- Memory bandwidth bottleneck for block gathering
- No built-in support for KV cache compression

## References
- Original PagedAttention paper
- Foundation of vLLM inference engine
