# Idea Scoring Summary

| Idea | Name | Novelty | Feasibility | Impact | Total | Strategy |
|------|------|---------|-------------|--------|-------|----------|
| 1 | Quantized Distillation (QuDist) | 4/5 | 4/5 | 5/5 | **13/15** | Combination |
| 2 | SparsePagedAttention (SPA) | 5/5 | 5/5 | 5/5 | **15/15** | Simplification |
| 3 | Task-Adaptive KV (TAKC) | 4/5 | 4/5 | 4/5 | **12/15** | Generalization |
| 4 | Token-LoRA Fusion (ToFu) | 5/5 | 3/5 | 4/5 | **12/15** | Architecture |
| 5 | Cross-Vocabulary Distillation (CVD) | 5/5 | 3/5 | 5/5 | **13/15** | Constraint Relaxation |

## Selected Idea: **SparsePagedAttention (SPA)**

**Why selected:**
- Highest total score (15/15)
- Combines two strong existing methods (Pre-hoc + Paged)
- Clear implementation path
- Theoretical guarantees from Pre-hoc Sparsity
- Immediate practical impact for serving
- Novelty in integration, not just new technique

**Runner-up:** Cross-Vocabulary Distillation (13/15) - high novelty but harder to implement

