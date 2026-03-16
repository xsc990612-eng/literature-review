# Paper Analysis: DistillLens (2602.13567)

## Metadata
- **arXiv ID**: 2602.13567
- **Title**: DistillLens: Symmetric Knowledge Distillation Through Logit Lens
- **Authors**: Manish Dhakal, Uthman Jinadu, Anjila Budathoki, Rajshekhar Sunderraman, Yi Ding
- **Score**: 5/5
- **Category**: Knowledge Distillation
- **Published**: 2026-02

## Abstract
Standard Knowledge Distillation (KD) compresses Large Language Models (LLMs) by optimizing final outputs, yet it typically treats the teacher's intermediate layer's thought process as a black box. DistillLens introduces symmetric alignment of evolving thought processes using Logit Lens projection and symmetric divergence objectives.

## Core Method

### 1. Key Innovation: Intermediate Thought Process Alignment

**Problem with Standard KD**:
- Only optimizes final output logits
- Treats teacher's intermediate reasoning as black box
- Missing rich uncertainty profiles from hidden layers

**DistillLens Solution**:
- Projects intermediate hidden states into vocabulary space via Logit Lens
- Enforces structural alignment using symmetric divergence
- Aligns student's internal trajectory with teacher's thought process

### 2. Theoretical Framework

**Confidence Score Definition**:

$$c_{\theta}(y|x) = \frac{q_{\theta}(y|x)}{p(y|x)}$$

Where $c_{\theta}$ measures student probability relative to teacher.

**Jensen-Shannon Divergence (JSD)**:

$$\mathcal{L}_{JSD}(p, q_\theta) = \frac{1}{2} \left[ \mathcal{L}_{KL}(p \| m) + \mathcal{L}_{KL}(q_\theta \| m)\right]$$

Where $m(y|x) = \frac{1}{2}(p(y|x) + q_\theta (y|x))$ is the mixed distribution.

**Key Theoretical Result - Dual-sided Alignment**:

$$\mathcal{L}_{JSD}(p,q_\theta) = \frac{1}{2} \mathbb{E}_{p} \Big[ \underbrace{c_\theta \log c_\theta - (1+c_\theta) \log \frac{1+c_\theta}{2}}_{g(c_\theta)} \Big]$$

**Three Cases of Optimization Landscape**:

1. **Overconfidence** ($c_{\theta} \to \infty$):
$$\lim_{c_{\theta} \to \infty} g(c_\theta) \approx c_\theta \log 2 \quad \text{(Linear Hallucination Penalty)}$$

2. **Underconfidence** ($c_{\theta} \to 0$):
$$\lim_{c_{\theta} \to 0} g(c_\theta) = \log 2 \quad \text{(Bounded Missed Recalls Penalty)}$$

3. **Perfect Alignment** ($c_{\theta} = 1$):
$$g(1) = 0$$

### 3. Layer Mapping Strategy

Uniform proportional mapping between student layer $l$ and teacher layer $l'$:

$$l' = \text{Round}\left( l \times \frac{L_T}{L_S} \right)$$

Where $L_T$ and $L_S$ are teacher and student layer counts.

**Example Mappings**:
- GPT-2 (120M): layers {2,4,6,8,10} → {8,16,24,32,40}
- GPT-2 (340M): layers {4,8,12,16,20} → {8,16,24,32,40}
- TinyLlama: layers {4,7,11,15,18} → {5,10,16,21,26}

### 4. Training Objective

**Intermediate Loss**:
$$\mathcal{L}_{inter} = \frac{1}{|\mathcal{M}|} \sum_{(l, l') \in \mathcal{M}} \mathcal{L}_{JSD}\left(p^{(l')}(y|x), q_\theta^{(l)}(y|x)\right)$$

**Total Loss**:
$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \cdot \mathcal{L}_{inter}$$

Where:
- $\mathcal{L}_{task}$: Standard task loss (e.g., Reverse KL for instruction following)
- $\lambda = 1.0$: Scaling hyperparameter
- $|\mathcal{M}|$: Number of mapped layer pairs

### 5. Algorithm Overview

```
Input: Dataset D, teacher p, student q_θ, layer mapping M, λ=1.0
Output: Trained student parameters θ_N

For each training step:
  1. Sample batch x from D
  2. Get intermediate states {h_p}, {h_q}
  3. Project via LogitLens: q_θ^(l) = LogitLens(h_q^(l))
  4. Project via LogitLens: p^(l') = LogitLens(h_p^(l'))
  5. Compute L_inter = (1/|M|) Σ JSD(p^(l'), q_θ^(l))
  6. Compute L_task (e.g., standard KD)
  7. L_total = L_task + λ · L_inter
  8. Update θ by descending ∇_θ L_total
```

## Technical Contributions

1. **Logit Lens for Training**: First to use Logit Lens for active supervision (not just interpretability)
2. **Symmetric Divergence**: JSD provides dual-sided penalty (vs asymmetric KL)
3. **Modular Design**: Can be integrated with any distillation approach (off-policy or on-policy)
4. **Theoretical Justification**: Proves convex, dual-sided optimization path

## Evaluation Results
- GPT-2-340M: 23.72 Rouge-L (surpasses teacher 23.52)
- GPT-2-120M: 21.12 Rouge-L (vs 17.74 for standard KD)
- TinyLlama-1.1B: 25.48 Rouge-L (vs 23.82 for standard KD)
- Consistently improves all baseline approaches

## Relationship to Our Research

### Relevance
This paper provides a **principled approach to intermediate layer distillation** with strong theoretical foundations, highly relevant for model compression and knowledge transfer.

### Key Insights
1. **Hidden states as belief states**: Intermediate layers contain rich uncertainty information
2. **Symmetric divergence essential**: Prevents both overconfidence and underconfidence
3. **Logit Lens projection**: Enables semantic comparison across different model depths
4. **Proportional layer mapping**: Uniform stride captures thought trajectory evolution

### Potential Applications
- **Multi-teacher distillation**: Apply symmetric alignment across multiple teachers
- **Cross-architecture transfer**: Logit Lens enables comparison across different architectures
- **Dynamic depth distillation**: Vary layer mapping based on task complexity

### Limitations to Consider
- Requires access to teacher's hidden states (white-box distillation)
- Layer mapping is fixed during training (not adaptive)
- Logit Lens assumes shared vocabulary space
- Computational overhead of intermediate projections

## References
- Based on preprint.tex from the paper source
