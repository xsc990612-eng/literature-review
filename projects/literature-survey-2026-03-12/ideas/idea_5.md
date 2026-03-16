# Idea 5: Cross-Vocabulary Distillation (CVD)

## Strategy
**Constraint Relaxation** - Remove DistillLens's shared vocabulary assumption

## Core Concept
Enable distillation between teacher and student with different vocabularies by learning a differentiable vocabulary alignment module.

## Problem Statement
DistillLens (2602.13567) uses Logit Lens projection:
```
L_JSD = JSD(TeacherLogits || StudentLogits)
```

This requires `vocab_teacher == vocab_student`.

## Mathematical Formulation

### Vocabulary Alignment Matrix
Learn alignment matrix A where:
```
A[i,j] = similarity(token_i_teacher, token_j_student)
```

### Alignment Strategies

**Strategy 1: Embedding-based**
```
A[i,j] = cos(E_teacher[i], E_student[j])
```

**Strategy 2: Semantic (using LM)**
```
A[i,j] = BERT_similarity(definition_i, definition_j)
```

**Strategy 3: Learned**
```
A = softmax(MLP([E_teacher; E_student]))
```

### Projected Distillation Loss
```
p_teacher_aligned = A^T @ p_teacher  # project to student vocab
L_JSD = JSD(p_teacher_aligned || p_student)
```

### Alignment Loss (to prevent collapse)
```
L_align = -Σ_i max_j A[i,j]  # encourage sharp alignments
```

## Key Innovation
First distillation method that works across arbitrary vocabularies, enabling:
- Cross-lingual distillation (English teacher → Chinese student)
- Cross-tokenizer (BPE → SentencePiece)
- Cross-model-family (GPT → Llama)

## Use Cases

| Scenario | Teacher | Student | A Matrix Size |
|----------|---------|---------|---------------|
| Cross-lingual | Llama-2-7B-en | Qwen-7B-zh | 32K × 32K |
| Cross-tokenizer | GPT-4 (cl100k) | Llama (32K) | 100K × 32K |
| Compression | LLaMA-65B | TinyLlama-1.1B | 32K × 32K |

## Implementation
```python
class CrossVocabularyDistiller:
    def __init__(self, teacher_vocab, student_vocab):
        self.alignment = learn_alignment_matrix(teacher_vocab, student_vocab)
    
    def forward(self, teacher_logits, student_logits):
        # Project teacher to student vocabulary
        aligned_teacher = self.alignment.T @ softmax(teacher_logits)
        
        # Standard JSD distillation
        loss = jsd_loss(aligned_teacher, student_logits)
        return loss
```

## Validation Plan
- Teacher: LLaMA-2-7B (32K vocab)
- Student: Qwen-1.8B (specialized Chinese vocab)
- Baseline: No distillation, random initialization
- Metrics: Chinese downstream tasks (C-Eval, CMMLU)

## Citations
- DistillLens (2602.13567): Symmetric JSD formulation
- TinyBERT (1912.01973): Layer mapping strategies
- MiniLLM: White-box distillation (cited in survey)

## Risk
Alignment quality bottleneck; may need iterative refinement.

## Extension
Combine with Idea 1 (QuDist) for 4-bit cross-vocabulary students.

