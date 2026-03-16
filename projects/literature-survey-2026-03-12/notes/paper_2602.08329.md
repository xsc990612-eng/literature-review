# Paper Analysis: Pre-hoc Sparsity (PrHS) (2602.08329)

## Metadata
- **arXiv ID**: 2602.08329
- **Title**: Near-Oracle KV Selection via Pre-hoc Sparsity for Long-Context Inference
- **Authors**: Yifei Gao, Lei Wang, Rong-Cheng Tu, Qixin Zhang, Jun Cheng, Dacheng Tao
- **Score**: 5/5
- **Category**: vLLM / Sparse Attention
- **Published**: 2026-02

## Abstract
A core bottleneck in large language model (LLM) inference is the cost of attending over the ever-growing key-value (KV) cache. Pre-hoc Sparsity (PrHS) selects KV entries before attention scoring and provides explicit accuracy control, reducing retrieval overhead by >90% and achieving 3× higher retrieval sparsity than HShare.

## Core Method

### 1. Overview: Pre-hoc Sparsity (PrHS)

**Key Difference from Prior Work**:
- **Post-hoc methods**: Select KV after computing attention scores (expensive)
- **PrHS (Pre-hoc)**: Selects KV entries BEFORE attention scoring
- Provides explicit accuracy control through theoretical guarantees

### 2. Three Core Components

#### A. Clustered Indices Sharing (CIS)

**Purpose**: Head-level KV sharing across adjacent queries

**Selection Criteria**:
Three token groups with budgets:
- $C_{\text{sink}}$: Initial sink tokens
- $C_{\text{local}}$: Local tokens
- $k$: Salient middle tokens (top-$k$ selection)

**Per-head Critical Index Set**:
$$\mathcal{C}_t = \{1,\ldots,C_{\text{sink}}\} \cup S_t^* \cup \{t-C_{\text{local}}+1,\ldots,t\}$$

Total size: $C = C_{\text{sink}} + k + C_{\text{local}}$

**Similarity Criteria**:
$$\mathsf{sim}(i,j) = \frac{\mathbf{q}_{i}^{\top} \mathbf{q}_{j}}{\|\mathbf{q}_{i}\|\,\|\mathbf{q}_{j}\|}$$

Sharing triggered when $\mathsf{sim}(i,j) > \tau$

**Neighboring Dilation**:
$$\hat S_t = S_t^* \cup \bigcup_{i=1}^{m}\big\{p_{t,i}+j \;|\;-r\le j\le r\,\big\}$$

Top-$m$ indices dilated with $\pm r$ neighbors

**Theorem (CIS Retained-Mass and MI Guarantee)**:
$$\tau_{\mathrm{pre}}(\mathbf{q}') \ge \tau^*(\mathbf{q}') - \beta_{\mathrm{th}}(\tau)$$

Where:
$$\beta_{\mathrm{th}}(\tau) \le 2\Delta_{\mathrm{att}}(\tau) \le \frac{2K_{\max}}{\sqrt d}\sqrt{2-2\tau}$$

#### B. Progressive Sliding Attention Window (PSAW)

**Purpose**: Progressive narrowing of attention window

$$P_\ell(t) =
\begin{cases}
0, &\ell < \ell_s
\\
\big\lfloor (1 - \phi^{\alpha\cdot\frac{\ell-\ell_s}{N-\ell_s}}) \, t \big\rfloor, &\ell \ge \ell_s
\end{cases}$$

Where:
- $\ell_s$: Layer depth where pruning starts
- $\phi \in (0,1)$: Controls truncation strength
- $\alpha \ge 0$: Decay schedule parameter
- $N$: Total number of layers

#### C. Early Token Freezing (ETF)

**Purpose**: Freeze expanding prefix of early tokens in deeper layers

$$E_\ell(t) =
\begin{cases}
0, &\ell < \ell_s
\\
\big\lfloor (1 - \psi^{\gamma \cdot\frac{\ell-\ell_s}{N-\ell_s}}) \, t \big\rfloor, &\ell \ge \ell_s
\end{cases}$$

Where:
- $\psi \in (0,1)$: Controls final unfrozen fraction
- $\gamma > 0$: Modulates schedule nonlinearity

### 3. Key Theoretical Results

**Attention Change Bound**:
$$\Delta_{\mathrm{att}}(\tau) := \|A(\mathbf{q}') - A(\mathbf{q})\|_1 \le \frac{2K_{\max}}{\sqrt d}\sqrt{2-2\tau}$$

**Pre-hoc Accuracy Guarantee**:
The method provides explicit accuracy control through:
- Lipschitz continuity of attention distributions
- Mean-shift bounds for clustered indices
- Mutual information gap bounds

## Technical Contributions

1. **Pre-hoc Selection**: First to select KV before attention (vs post-hoc heuristics)
2. **CIS with Dilation**: Neighboring dilation increases true-positive overlap
3. **Theoretical Guarantees**: Explicit accuracy control via $\beta_{\mathrm{th}}$ bounds
4. **3× Sparsity**: Higher retrieval sparsity than HShare at same/better accuracy
5. **>90% Overhead Reduction**: Significant retrieval overhead reduction

## Evaluation Results
- >90% retrieval overhead reduction
- 3× higher retrieval sparsity than HShare
- Validated on LLaMA and Mistral families
- Tested on GSM8K and CoQA benchmarks

## Relationship to Our Research

### Relevance
This paper provides **theoretically-grounded pre-hoc sparsity** for KV cache compression, complementing post-hoc approaches like HShare.

### Key Insights
1. **Pre-hoc vs Post-hoc**: Pre-selection enables explicit accuracy guarantees
2. **Query similarity**: Cosine similarity threshold $\tau$ controls sharing quality
3. **Dilation essential**: Top-$m$ dilation captures centroid drift
4. **Layer-wise decay**: Exponential decay schedule matches attention locality

### Potential Applications
- **Hybrid sparsity**: Combine PrHS with post-hoc methods
- **Adaptive thresholds**: Learn $\tau$ based on input characteristics
- **Hardware-aware dilation**: Optimize $r$ based on memory hierarchy

### Limitations to Consider
- Requires query similarity computation overhead
- Block-based sharing may miss cross-block similarities
- Multiple hyperparameters ($\tau$, $m$, $r$, $\phi$, $\alpha$) need tuning

## References
- Based on latex/5-method.tex from the paper source
