# Idea 3: Task-Adaptive KV Compression (TAKC)

## Strategy
**Generalization** - Extend LongFlow to task-specific KV retention policies

## Core Concept
Instead of a universal compression strategy, TAKC learns task-specific KV importance functions. Classification tasks keep different tokens than generation tasks.

## Mathematical Formulation

### LongFlow Score (from 2603.11504)
```
LongFlowScore(t_i) = ||α_t^i v^i||_1
```

### Task-Adaptive Extension
```
TAKC_Score(t_i, task) = ||α_t^i v^i||_1 * w_task(token_type, position)
```

Where `w_task` is learned per-task weighting function:
- **Classification**: Emphasize [CLS] token attention, early layers
- **Generation**: Emphasize recent tokens, position bias
- **Reasoning**: Emphasize reasoning tokens (LongFlow pattern)

### Learning w_task
```
min_w E[Loss(model_with_compression(task), target)]
```

During a short calibration phase on task examples, learn which tokens can be dropped.

## Key Innovation
LongFlow assumes all reasoning tasks have same KV importance. TAKC customizes for specific task patterns.

## Task-Specific Policies

| Task Type | Keep Policy | Compression Ratio |
|-----------|-------------|-------------------|
| Sentiment (cls) | CLS attention tokens | 95% |
| Summarization | Salient content tokens | 80% |
| Code gen | Syntax structure tokens | 85% |
| Math reasoning | Step markers + operands | 75% |

## Implementation
```python
class TaskAdaptiveKVCache:
    def __init__(self, task_type):
        self.importance_fn = load_task_weights(task_type)
    
    def compress(self, kv_cache, attention_weights):
        scores = self.importance_fn(attention_weights)
        keep_mask = top_k(scores, k=target_budget)
        return kv_cache[keep_mask]
```

## Validation Plan
- Tasks: Sentiment (SemEval), Summarization (CNN/DM), Code (HumanEval), Math (GSM8K)
- Baseline: LongFlow (uniform), full KV cache
- Metrics: Task accuracy vs compression ratio tradeoff curve

## Citations
- LongFlow (2603.11504): Contribution-based scoring
- Pre-hoc Sparsity (2602.08329): Selection guarantees
- SemEval tasks (1912.02387, 1912.02990, 1912.06806): Evaluation benchmarks

## Risk
Overfitting to specific tasks; need to verify generalization.

