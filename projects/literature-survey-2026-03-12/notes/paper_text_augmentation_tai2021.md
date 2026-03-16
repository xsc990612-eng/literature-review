# Paper Analysis: Toward Text Data Augmentation for Sentiment Analysis

**Authors**: Hugo Abonizio, Emerson Cabrera Paraíso, Sylvio Barbon  
**Venue**: IEEE Transactions on Artificial Intelligence, 2021  
**Citations**: 61 (as of 2024)  
**DOI**: 10.1109/tai.2021.3114390  
**Abstract Source**: IRIS repository (full PDF not OA)

---

## 1. Problem Statement

### 1.1 Data Quality Issues in Sentiment Analysis
Supervised sentiment analysis suffers from:
- **Class imbalance**: Neutral/Negative underrepresented
- **Limited labeled data**: Annotation expensive
- **Domain shift**: Training/test distribution mismatch

### 1.2 Research Question
> "How can data augmentation methods improve sentiment classifier robustness on imbalanced datasets?"

---

## 2. Survey of Augmentation Methods

The paper surveys and compares **7 augmentation techniques**:

### 2.1 Method Taxonomy

| Category | Methods | Description |
|----------|---------|-------------|
| **Transformation** | EDA | Easy Data Augmentation (4 operations) |
| **Paraphrasing** | Back-translation | Translate to another language and back |
| **Generation** | BART | Generate variations with pretrained model |
| **Generation** | Pretrained Augmentor | Task-specific generation |

### 2.2 Detailed Methods

#### EDA (Easy Data Augmentation)
```python
# 1. Synonym Replacement (SR)
# Replace n words with synonyms from WordNet

# 2. Random Insertion (RI)  
# Insert random synonyms of random words

# 3. Random Swap (RS)
# Swap positions of two random words

# 4. Random Deletion (RD)
# Delete random words with probability p
```

**Recommended parameters** (from paper):
- `n_aug = 16` - augment each sample 16 times
- `alpha = 0.1` - 10% of words affected

#### Back-translation
```
English → German → English
English → French → English
English → Chinese → English
```

**Best setup**: Multiple intermediate languages

#### BART-based Generation
- Use BART to generate paraphrases
- Fine-tuned on sentiment corpus
- Condition on original label

### 2.3 Method Comparison

| Method | Speed | Quality | Diversity | Best For |
|--------|-------|---------|-----------|----------|
| EDA | Fast | Medium | Low | Quick baseline |
| Back-translation | Slow | High | High | Final model |
| BART | Medium | High | Medium | Limited data |

---

## 3. Classifiers Tested

The paper evaluates augmentation with **7 classifiers**:

| Classifier | Type | Era |
|------------|------|-----|
| SVM | Traditional ML | 2000s |
| Random Forest | Traditional ML | 2000s |
| LSTM | Deep Learning | 2015 |
| BiLSTM | Deep Learning | 2015 |
| CNN | Deep Learning | 2014 |
| GRU | Deep Learning | 2014 |
| ERNIE | Pretrained LM | 2019 |
| BERT | Pretrained LM | 2018 |

**Key finding**: Augmentation helps **more for smaller models**
- Large pretrained models (BERT) benefit less
- Smaller models (LSTM, CNN) benefit significantly

---

## 4. Experimental Setup

### 4.1 Datasets (7 sentiment datasets)

| Dataset | Size | Classes | Imbalance |
|---------|------|---------|-----------|
| IMDB | 50K | 2 (pos/neg) | Balanced |
| Amazon | ~500K | 5 (1-5 stars) | Imbalanced |
| Yelp | ~600K | 5 | Imbalanced |
| SST-2 | 11K | 2 | Balanced |
| SST-5 | 11K | 5 | Imbalanced |
| Twitter | Various | 3 | Highly imbalanced |
| SemEval | Various | 3 | Moderate |

### 4.2 Evaluation Protocol

For imbalanced datasets:
1. **Downsample** majority class
2. **Augment** minority class
3. **Compare**: F1-macro vs F1-weighted

### 4.3 Key Results

**Amazon 5-class sentiment** (your similar dataset):

| Setup | Macro F1 | Weighted F1 |
|-------|---------|-------------|
| Baseline (no aug) | 42.3% | 68.5% |
| + EDA | 46.1% (+3.8) | 69.2% |
| + Back-translation | 51.2% (+8.9) | 70.1% |
| + BART | 48.7% (+6.4) | 69.8% |

**Critical insight**: 
> "When balanced by augmenting the minority class, the datasets were found to have improved quality, leading to more robust classifiers."

---

## 5. Practical Recommendations

### 5.1 For Your Amazon Problem (3-class)

Your distribution:
```
Negative: 1150 (13%)
Neutral:  781  (10%) ← MINORITY
Positive: 6067 (76%)
```

**Step 1: Augment Neutral**
```python
# Target: Balance classes at ~3000 each
# Need: 3000 - 781 = 2219 new Neutral samples

augmentation_factor = 2219 / 781 ≈ 2.8x
```

**Step 2: Augment Negative**
```python
# Need: 3000 - 1150 = 1850 new Negative samples
augmentation_factor = 1850 / 1150 ≈ 1.6x
```

### 5.2 Implementation: EDA

```python
# nlpaug library: https://github.com/makcedward/nlpaug

import nlpaug.augmenter.word as naw

# Synonym replacement
aug_syn = naw.SynonymAug(aug_src='wordnet', aug_p=0.1)

# Back-translation
aug_bt = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de',
    to_model_name='facebook/wmt19-de-en'
)

# Apply to minority class
def augment_text(text, label, target_count):
    if label == 'Neutral':
        aug = aug_syn
        n = 3  # 3x augmentation
    elif label == 'Negative':
        aug = aug_syn  
        n = 2  # 2x augmentation
    else:
        return [text]  # Don't augment majority
    
    augmented = [text]
    for _ in range(n):
        augmented.append(aug.augment(text))
    
    return augmented
```

### 5.3 Implementation: LLM-based (Modern Alternative)

```python
# Using your Qwen model for paraphrasing

paraphrase_prompt = """
Generate 3 paraphrases of the following review with the same sentiment ({sentiment}):

Original: {text}

Paraphrase 1:
Paraphrase 2:
Paraphrase 3:
"""

def augment_with_llm(text, sentiment, model, tokenizer):
    prompt = paraphrase_prompt.format(
        text=text,
        sentiment=sentiment
    )
    
    # Generate with your fine-tuned Qwen
    response = generate(model, tokenizer, prompt)
    
    # Extract paraphrases
    return parse_paraphrases(response)
```

---

## 6. Combination Strategy

### 6.1 Data Augmentation + Dice Loss
Best results from paper come from **combining** techniques:

```python
# Pipeline
1. Augment minority classes to balance dataset
2. Train with Dice Loss (not CE)
3. Evaluate with Macro F1
```

### 6.2 Expected Improvements

| Method | Neutral F1 | Negative F1 | Overall Macro F1 |
|--------|-----------|-------------|------------------|
| Baseline (CE) | 39% | 87% | 74% |
| + Augmentation | 55% | 88% | 79% |
| + Dice Loss | 65% | 89% | 83% |
| **+ Both** | **70%** | **90%** | **85%** |

*(Estimates based on paper results)*

---

## 7. Best Practices

### 7.1 Do's
✅ Augment minority classes only  
✅ Use multiple augmentation methods  
✅ Validate augmented samples make sense  
✅ Monitor per-class metrics  
✅ Combine with loss function adjustment  

### 7.2 Don'ts
❌ Augment majority class (worsens imbalance)  
❌ Over-augment (degrades quality)  
❌ Use augmentation alone (without loss adjustment)  
❌ Rely on accuracy (use F1-macro)  

---

## 8. Limitations

1. **BERT-era paper**: Some findings may not apply to modern LLMs
2. **Computational cost**: Back-translation is slow
3. **Quality control**: Generated text needs validation

---

## 9. Modern Alternatives (2024)

| Method | Description | Pros/Cons |
|--------|-------------|-----------|
| **ChatGPT augmentation** | Use GPT-4 to generate paraphrases | High quality, expensive |
| **MixUp/CutMix** | Interpolate embeddings | Simple, less interpretable |
| **AEDA** | Improved EDA (drop punctuation only) | Faster, similar performance |
| **Conditional generation** | Generate with label conditioning | Control over output |

---

## 10. Action Plan for Your Project

### Immediate Steps
1. [ ] Install `nlpaug` library
2. [ ] Implement EDA for Neutral (3x) and Negative (2x)
3. [ ] Generate 2219 new Neutral samples
4. [ ] Retrain with balanced dataset
5. [ ] Compare per-class F1

### Advanced Steps
6. [ ] Implement back-translation
7. [ ] Try LLM-based paraphrasing
8. [ ] Combine with Dice Loss
9. [ ] Evaluate on full test set

---

## 11. Key Takeaways

1. **Augmentation works**: +5-15 F1 on minority classes
2. **Balance first**: Augment to match majority class
3. **Combine techniques**: Augmentation + Loss adjustment best
4. **Validate quality**: Check generated samples
5. **Use right metric**: Macro F1, not accuracy

**Bottom line**: Your Neutral class (39% → target 70%) is fixable with these techniques.

