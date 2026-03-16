# Paper Analysis: Zipage (2603.08743)

## Metadata
- **arXiv ID**: 2603.08743
- **Title**: Zipage: Maintain High Request Concurrency for LLM Reasoning through Compressed PagedAttention
- **Authors**: Mengqi Liao, Lu Wang, Chaoyun Zhang, Bo Qiao, Si Qin, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Huaiyu Wan
- **Score**: 5/5
- **Category**: vLLM / Inference Optimization
- **Published**: 2026-03

## Abstract
With reasoning becoming the generative paradigm for large language models (LLMs), the memory bottleneck caused by KV cache during the decoding phase has become a critical factor limiting high-concurrency service. This paper introduces Compressed PagedAttention, combining token-wise KV cache eviction with PagedAttention, with comprehensive scheduling and support for prefix caching and asynchronous compression.

## Core Method

### 1. Compressed PagedAttention Overview

**Key Features**:
- Each request's block count capped at $N_{\max}$ (except during prefilling)
- Compression triggered when: $N \ge N_{\max}$ AND last block fully occupied
- Less important KV entries evicted, retained ones relocated to first $N_{\max} - 1$ blocks
- $N_{\max}$-th block reserved for subsequent decoding

**Memory Model**:

Pre-allocated GPU memory: $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{L \times N_{\text{total}} \times b \times h_{\text{kv}}\times d}$

Where:
- $L$: number of layers
- $N_{\text{total}}$: total number of blocks
- $b$: block size
- $h_{\text{kv}}$: number of attention heads
- $d$: attention dimension

### 2. Maximum Concurrency Formulation

Linear programming constraints for maximum concurrency $M$:

$$\left\{
\begin{aligned}
    &m_{\text{kv}} \times N_{\text{total}} + M \times m_{\text{q}} \leq m_{\text{available}}, \\
    &M \leq \frac{N_{\text{total}}}{N_{\max}}, \\
    &M > 0, \quad N_{\text{total}} > 0,
\end{aligned}
\right.$$

Where:
- $m_{\text{available}}$: total available memory
- $m_{\text{kv}}$: memory per KV cache block
- $m_{\text{q}}$: memory for caching query states

**Optimal Solution**:
$$M = \left\lfloor \frac{m_{\text{available}}}{m_{\text{kv}}\times N_{\max}+m_{\text{q}}}\right\rfloor$$

$$N_{\text{total}} = \left\lfloor\frac{m_{\text{available}}}{m_{\text{kv}}+{m_{\text{q}}}/{N_{\max}}} \right\rfloor$$

### 3. Compression Scoring Function

**Scoring Function**: $\phi(\mathbf Q,\mathbf K, \mathcal I)$

Where:
- $\mathbf{Q} \in \mathbb{R}^{L \times M \times w \times h_{q} \times d}$: query states from observation window (last $w$ tokens)
- $\mathcal{I}$: block tables and query slot indexes

**Integrated Methods**:
1. **Attention Score**: Basic attention between query states and key states
2. **R-KV Redundancy Score**: Evaluates redundancy of KV cache
3. **G-KV Global Score**: Aggregates historical attention scores

**Lightning Redundancy Score** (novel contribution):
- Reduced complexity: $\mathcal{O}(N \times b^2)$ (vs $\mathcal{O}(N^2 \times b^2)$ for raw R-KV)
- Better performance than raw redundancy score
- Significantly accelerates compression

### 4. Scheduling Strategies

**Constrained Scheduling**:
- Concurrency limited to $M$
- No preemption needed
- Simple but may underutilize blocks

**Hybrid Scheduling** (proposed):
- Only first $M$ requests eligible for query slots
- Requests with $< N_{\max}$ blocks or $< b-w$ tokens in last block can decode without query slots
- Preemption triggered only when necessary
- Maximizes block utilization

### 5. Prefix Caching Support

**Challenge**: Compression disrupts shared prefix structure

**Solution**:
- If $N_{\text{prefix}} \ge N_{\max} - 1$: allocate $N_{\max} - 1$ new blocks
- If $N_{\text{prefix}} < N_{\max} - 1$: allocate $N_{\text{prefix}}$ new + reuse existing blocks
- Shared prefixes preserved after compression

### 6. Asynchronous Decoding and Compression

**Observation**:
- Decoding dominates time (~90%)
- Compression ~10% of total time
- Only $<1\%$ of requests need compression per step

**Solution**:
- Compression and decoding executed asynchronously
- Requests ready for decoding don't wait for compression
- Compressed requests rejoin subsequent decoding steps
- Eliminates GPU idle time

## Technical Contributions

1. **Token-wise + PagedAttention**: First practical combination supporting continuous batching
2. **Hybrid Scheduling**: Maximizes concurrency while maintaining simplicity
3. **Prefix Cache Compatibility**: Preserves shared prefix structure through compression
4. **Lightning Redundancy Score**: $\mathcal{O}(N \times b^2)$ complexity with better performance
5. **Asynchronous Execution**: Hides compression latency behind decoding

## Evaluation Results
- **2.1× speedup** over Full KV inference engines
- **95% performance retention** at 2K KV cache budget
- Outperforms vLLM and Nano-vLLM in TPS
- Effective on reasoning tasks (AMC 23, AIME 24, LiveCodeBench)

## Relationship to Our Research

### Relevance
This paper provides a **production-ready system** combining KV cache compression with industrial inference engine features (continuous batching, prefix caching).

### Key Insights
1. **Token-wise eviction** is essential for maintaining accuracy vs page-wise
2. **Scheduling matters**: Hybrid scheduling significantly improves concurrency
3. **Asynchronous execution** crucial for hiding compression overhead
4. **Complexity reduction** (Lightning score) enables practical deployment
5. **Prefix caching** requires special handling during compression

### Potential Applications
- **vLLM enhancement**: Integrate Compressed PagedAttention into existing serving stack
- **Reasoning model serving**: Optimize for long-CoT models like o1/R1
- **Multi-tenant deployment**: Hybrid scheduling for diverse workload mixes

### Limitations to Consider
- Block size $b$ and window $w$ are fixed hyperparameters
- Memory model assumes uniform access patterns
- Requires custom kernel implementation for full benefits

## References
- Based on acl_latex.tex from the paper source
