# Research Gaps Identified

## Gap 1: Ultra-Low Bit Quantization Theory
- **Description**: LowRA explores sub-2-bit quantization empirically, but lacks theoretical understanding of information loss bounds and convergence guarantees
- **Mentioned in**: LowRA (2502.08141), QLoRA (2305.14314)
- **Why important**: Without theory, practitioners cannot predict when ultra-low bit methods will fail; no principled way to select bit-width for target accuracy

## Gap 2: Quantization-Distillation Synergy
- **Description**: QLoRA/LoftQ optimize for fine-tuning; DistillLens optimizes for compression. No work combines quantized fine-tuning with symmetric distillation
- **Mentioned in**: QLoRA (2305.14314), DistillLens (2602.13567)
- **Why important**: Could enable training 4-bit student models from 16-bit teachers with theoretical guarantees on both quantization error and distillation loss

## Gap 3: Dynamic Multi-Adapter KV Cache Management
- **Description**: AdaFuse optimizes adapter computation; PagedAttention optimizes KV cache. No method optimizes both simultaneously for multi-tenant serving
- **Mentioned in**: AdaFuse (2603.11873), PagedAttention (2309.06180)
- **Why important**: Real-world serving requires handling multiple LoRA adapters while managing KV cache efficiently; current solutions address only one side

## Gap 4: Task-Adaptive KV Cache Compression
- **Description**: LongFlow and Pre-hoc Sparsity use fixed compression strategies. No method adapts KV retention based on downstream task requirements
- **Mentioned in**: LongFlow (2603.11504), Pre-hoc Sparsity (2602.08329)
- **Why important**: Different tasks (classification vs generation vs reasoning) have different attention patterns; universal compression is suboptimal

## Gap 5: Cross-Vocabulary Distillation
- **Description**: DistillLens requires shared vocabulary for Logit Lens projection. No solution for distilling between models with different tokenizers
- **Mentioned in**: DistillLens (2602.13567), TinyBERT (1912.01973)
- **Why important**: Prevents distillation between different model families (e.g., Llama → GPT-style, or across languages)

## Gap 6: Sparsity-Aware Quantization
- **Description**: Pre-hoc Sparsity selects KV pairs; QLoRA quantizes weights. No method quantizes only the non-sparse components for additional savings
- **Mentioned in**: Pre-hoc Sparsity (2602.08329), QLoRA (2305.14314)
- **Why important**: Sparse attention already reduces compute; quantizing remaining dense operations could compound savings

