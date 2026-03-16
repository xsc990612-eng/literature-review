# Paper Analysis: QDyLoRA (2402.10462)

## Metadata
- **arXiv ID**: 2402.10462
- **Title**: QDyLoRA: Quantized Dynamic Low-Rank Adaptation for Efficient Large Language Model Tuning
- **Score**: 4/5
- **Category**: QLoRA / Dynamic Rank

## Core Method
- Dynamic LoRA rank selection during fine-tuning
- Quantized backbone with adaptive low-rank adaptation
- Addresses efficient rank selection challenge

## Key Formula
Adaptive rank selection: $r_t = f(t, \text{gradient\_magnitude})$

## Relationship to Our Research
- Dynamic rank adaptation for efficiency
- Automated architecture search for LoRA
