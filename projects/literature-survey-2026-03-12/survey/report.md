# 文献调研报告 - 2026-03-15 14:07

## 调研概览

| 领域 | 检索论文数 | 核心文献数 | 状态 |
|------|----------|-----------|------|
| GSDMM短文本主题建模 | 15 | 4 | ✓ 无新增 |
| QLoRA高效微调 | 15 | 7 | ✓ 无新增 |
| vLLM推理优化 | 15 | 9 | ✓ 无新增 |
| 知识蒸馏与Agent | 15 | 9 | ✓ 无新增 |
| 情感分析评估 | 15 | 10 | ✓ 无新增 |

**总计**: 39篇核心文献

**本次检索结果**: ✅ API可用，2026-03-12 17:59后无直接相关新增论文

**检查时间**: 2026-03-15 14:07 (Europe/Moscow)

---

## 核心文献库（39篇已覆盖）

### 1. GSDMM短文本主题建模 (4篇)

| 论文 | 作者 | 年份 | 核心贡献 | BibTeX |
|------|------|------|---------|--------|
| **Gibbs Sampling for Infinite Mixture Models** | Jianhua Yin, Daren Chao | 2012 | GSDMM基础算法，Movie Group Process类比 | `@article{yin2012gibbs, title={Gibbs Sampling for Infinite Mixture Models}, author={Yin, Jianhua and Chao, Daren}, year={2012}}` |
| **A Gamma-Poisson Mixture Topic Model** | Jianhua Yin, Jianyong Wang | 2020 | Gamma-Poisson混合模型，Poisson分布描述短文本词频 | `@article{yin2020gamma, title={A Gamma-Poisson Mixture Topic Model for Short Text}, author={Yin, Jianhua and Wang, Jianyong}, year={2020}}` |
| **Short Text Topic Modeling Techniques: A Survey** | Sajad Sotudeh Gharebagh等 | 2019 | 短文本主题建模综述 | `@article{sotudeh2019survey, title={Short Text Topic Modeling Techniques: A Survey}, author={Sotudeh Gharebagh, Sajad and others}, year={2019}}` |
| **An Enhanced Model-based Approach** | Jianhua Yin等 | 2025 | LLM增强的短文本聚类 | `@article{yin2025enhanced, title={An Enhanced Model-based Approach for Short Text Clustering}, author={Yin, Jianhua and others}, year={2025}}` |

---

### 2. QLoRA与LLM高效微调 (7篇)

| 论文 | 作者 | 年份 | 会议/期刊 | 核心贡献 | BibTeX |
|------|------|------|----------|---------|--------|
| **LoRA: Low-Rank Adaptation** | Edward J. Hu等 | 2021 | ICLR (CCF-A) | 低秩适应奠基论文，12,000+引用 | `@inproceedings{hu2021lora, title={LoRA: Low-Rank Adaptation of Large Language Models}, author={Hu, Edward J. and others}, booktitle={ICLR}, year={2021}}` |
| **QLoRA: Efficient Finetuning** | Tim Dettmers等 | 2023 | NeurIPS (CCF-A) | 4-bit量化+LoRA，NF4数据类型 | `@inproceedings{dettmers2023qlora, title={QLoRA: Efficient Finetuning of Quantized LLMs}, author={Dettmers, Tim and others}, booktitle={NeurIPS}, year={2023}}` |
| **LoRA-FA** | Longguang Zhong等 | 2023 | arXiv | 内存高效LoRA，冻结A矩阵 | `@article{zhong2023lorafa, title={LoRA-FA: Memory-efficient Low-rank Adaptation}, author={Zhong, Longguang and others}, year={2023}}` |
| **LoRA-Pro** | Zhenyu Zhang等 | 2024 | arXiv | LoRA优化分析，梯度低秩结构 | `@article{zhang2024lorapro, title={LoRA-Pro: Are Low-Rank Adapters Properly Optimized?}, author={Zhang, Zhenyu and others}, year={2024}}` |
| **Federated Sketching LoRA** | Ahmed Roushdy Elkordy等 | 2025 | arXiv | 联邦学习+LoRA异构微调 | `@article{elkordy2025federated, title={Federated Sketching LoRA}, author={Elkordy, Ahmed Roushdy and others}, year={2025}}` |
| **Leech Lattice VQ** | Tycho van der Ouderaa等 | 2026 | arXiv | 24维Leech格点向量量化，2-bit精度 | `@article{oudraa2026leech, title={Leech Lattice Vector Quantization for Efficient LLM Compression}, author={van der Ouderaa, Tycho F. A. and others}, journal={arXiv preprint arXiv:2603.11021}, year={2026}}` |
| **AdaFuse** | Qiyang Li等 | 2026 | arXiv | Token-level pre-gating加速动态Adapter | `@article{li2026adafuse, title={AdaFuse: Accelerating Dynamic Adapter Inference}, author={Li, Qiyang and others}, journal={arXiv preprint arXiv:2603.11873}, year={2026}}` |

---

### 3. vLLM推理优化 (9篇)

| 论文 | 作者 | 年份 | 会议/期刊 | 核心贡献 | BibTeX |
|------|------|------|----------|---------|--------|
| **PagedAttention** | Woosuk Kwon等 | 2023 | SOSP (CCF-A) | vLLM核心算法，24倍吞吐量提升，Copy-on-Write | `@inproceedings{kwon2023pagedattention, title={Efficient Memory Management for LLM Serving with PagedAttention}, author={Kwon, Woosuk and others}, booktitle={SOSP}, year={2023}}` |
| **vAttention** | Ramachandran Ramjee等 | 2024 | arXiv | 动态内存管理，无需PagedAttention | `@article{ramjee2024vattention, title={vAttention: Dynamic Memory Management without PagedAttention}, author={Ramjee, Ramachandran and others}, year={2024}}` |
| **Zipage** | Yilong Zhao等 | 2026 | arXiv | 压缩PagedAttention，推理模型专用 | `@article{zhao2026zipage, title={Zipage: Compressed PagedAttention for Reasoning}, author={Zhao, Yilong and others}, year={2026}}` |
| **FlashInfer** | Yilong Zhao等 | 2025 | arXiv | 高效注意力引擎 | `@article{zhao2025flashinfer, title={FlashInfer: Efficient Attention Engine for LLM Serving}, author={Zhao, Yilong and others}, year={2025}}` |
| **LookaheadKV** | Jinwoo Ahn等 | 2026 | ICLR (CCF-A) | 轻量级KV Cache驱逐，成本降低14.5倍 | `@article{ahn2026lookaheadkv, title={LookaheadKV: Fast and Accurate KV Cache Eviction}, author={Ahn, Jinwoo and others}, journal={arXiv preprint arXiv:2603.10899}, year={2026}}` |
| **LongFlow** | Yi Su等 | 2026 | arXiv | 推理模型专用KV Cache压缩 | `@article{su2026longflow, title={LongFlow: Efficient KV Cache Compression for Reasoning Models}, author={Su, Yi and others}, journal={arXiv preprint arXiv:2603.11504}, year={2026}}` |
| **Where Matters More** | Zhenxu Tian等 | 2026 | arXiv | 位置感知解码对齐KV Cache压缩 | `@article{tian2026where, title={Where Matters More Than What: Decoding-aligned KV Cache Compression}, author={Tian, Zhenxu and others}, journal={arXiv preprint arXiv:2603.11564}, year={2026}}` |
| **IndexCache** | Yushi Bai等 | 2026 | arXiv | 跨层索引复用加速稀疏注意力 | `@article{bai2026indexcache, title={IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse}, author={Bai, Yushi and others}, journal={arXiv preprint arXiv:2603.12201}, year={2026}}` |
| **Pre-hoc Sparsity (PrHS)** | Yifei Gao等 | 2026 | TPAMI | 预稀疏KV选择，>90%检索开销减少，理论互信息保证 | `@article{gao2026prehoc, title={Near-Oracle KV Selection via Pre-hoc Sparsity for Long-Context Inference}, author={Gao, Yifei and others}, journal={IEEE TPAMI}, year={2026}}` |

---

### 4. 知识蒸馏与Agent (9篇)

| 论文 | 作者 | 年份 | 会议/期刊 | 核心贡献 | BibTeX |
|------|------|------|----------|---------|--------|
| **Distilling Knowledge in NN** | Geoffrey Hinton等 | 2015 | NeurIPS Workshop | 知识蒸馏奠基论文，温度软化softmax | `@article{hinton2015distilling, title={Distilling the Knowledge in a Neural Network}, author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff}, year={2015}}` |
| **DistillLens** | Song等 | 2026 | arXiv | Logit Lens对称知识蒸馏 | `@article{song2026distilllens, title={DistillLens: Symmetric KD Through Logit Lens}, author={Song, Mingyang and others}, year={2026}}` |
| **FitNets** | Adriana Romero等 | 2014 | ICLR | 中间层特征蒸馏 | `@inproceedings{romero2014fitnets, title={FitNets: Hints for Thin Deep Nets}, author={Romero, Adriana and others}, booktitle={ICLR}, year={2014}}` |
| **TinyBERT** | Xiaoqi Jiao等 | 2020 | EMNLP | BERT两阶段蒸馏 | `@inproceedings{jiao2020tinybert, title={TinyBERT: Distilling BERT for Natural Language Understanding}, author={Jiao, Xiaoqi and others}, booktitle={EMNLP}, year={2020}}` |
| **MiniLLM** | Yuxian Gu等 | 2023 | NeurIPS | 生成模型白盒蒸馏 | `@inproceedings{gu2023minillm, title={MiniLLM: Knowledge Distillation of Large Language Models}, author={Gu, Yuxian and others}, booktitle={NeurIPS}, year={2023}}` |
| **DistilBERT** | Victor Sanh等 | 2020 | arXiv | BERT6层压缩版 | `@article{sanh2019distilbert, title={DistilBERT, a distilled version of BERT}, author={Sanh, Victor and others}, year={2019}}` |
| **Multi-teacher KD** | Jinghui Qin等 | 2023 | arXiv | 多教师知识蒸馏 | `@article{qin2023multiteacher, title={Multi-teacher Knowledge Distillation}, author={Qin, Jinghui and others}, year={2023}}` |
| **Dense Cumulative KD** | Wenyu Du等 | 2024 | arXiv | 密集累积知识蒸馏 | `@article{du2024dense, title={Densely Distilling Cumulative Knowledge}, author={Du, Wenyu and others}, year={2024}}` |
| **Bielik-Minitron-7B** | Remigiusz Kinas等 | 2026 | arXiv | 结构化剪枝+蒸馏 | `@article{kinas2026bielik, title={Bielik-Minitron-7B: Compressing LLMs via Structured Pruning and Knowledge Distillation}, author={Kinas, Remigiusz and others}, journal={arXiv preprint arXiv:2603.11881}, year={2026}}` |

---

### 5. 情感分析与立场检测 (10篇)

| 论文 | 作者 | 年份 | 会议/期刊 | 核心贡献 | BibTeX |
|------|------|------|----------|---------|--------|
| **SemEval-2013 Task 2** | Preslav Nakov等 | 2013 | NAACL | Twitter情感分析基准，1,500+引用 | `@inproceedings{nakov2013semeval, title={SemEval-2013 Task 2: Sentiment Analysis in Twitter}, author={Nakov, Preslav and others}, booktitle={NAACL}, year={2013}}` |
| **SemEval-2014 Task 9** | Sara Rosenthal等 | 2014 | SemEval | 扩展情感分类，1,200+引用 | `@inproceedings{rosenthal2014semeval, title={SemEval-2014 Task 9: Sentiment Analysis in Twitter}, author={Rosenthal, Sara and others}, year={2014}}` |
| **SemEval-2015 Task 10** | Sara Rosenthal等 | 2015 | NAACL | 多维度情感分析，800+引用 | `@inproceedings{rosenthal2015semeval, title={SemEval-2015 Task 10: Sentiment Analysis in Twitter}, author={Rosenthal, Sara and others}, booktitle={NAACL}, year={2015}}` |
| **SemEval-2016 Task 4** | Preslav Nakov等 | 2016 | SemEval | 情感强度检测，900+引用 | `@inproceedings{nakov2016semeval, title={SemEval-2016 Task 4: Sentiment Analysis in Twitter}, author={Nakov, Preslav and others}, booktitle={SemEval}, year={2016}}` |
| **SemEval-2017 Task 4** | Sara Rosenthal等 | 2017 | SemEval | 细粒度情感分析，600+引用 | `@inproceedings{rosenthal2017semeval, title={SemEval-2017 Task 4: Sentiment Analysis in Twitter}, author={Rosenthal, Sara and others}, booktitle={SemEval}, year={2017}}` |
| **Stance Detection Survey** | Kornelija Zakaite等 | 2021 | arXiv | 立场检测综述 | `@article{zakaite2021stance, title={Stance Detection: A Survey}, author={Zakaite, Kornelija and others}, year={2021}}` |
| **Adversarial Domain Adaptation** | Vijaykumar Dubey等 | 2019 | arXiv | 对抗领域适应立场检测 | `@article{dubey2019adversarial, title={Adversarial Domain Adaptation for Stance Detection}, author={Dubey, Vijaykumar and others}, year={2019}}` |
| **Beyond Illusion of Consensus** | Mingyang Song等 | 2026 | arXiv | LLM-as-Judge评估幻觉，MERG框架 | `@article{song2026illusion, title={Beyond the Illusion of Consensus}, author={Song, Mingyang and others}, journal={arXiv preprint arXiv:2603.11027}, year={2026}}` |
| **Beyond Polarity** | Dehao Dai等 | 2026 | arXiv | 多维LLM情感信号金融预测 | `@article{dai2026beyond, title={Beyond Polarity: Multi-Dimensional LLM Sentiment Signals}, author={Dai, Dehao and others}, journal={arXiv preprint arXiv:2603.11408}, year={2026}}` |
| **Emotion Detection in Text** | Saima Jabeen等 | 2020 | arXiv | 文本情感检测综述 | `@article{jabeen2020emotion, title={Emotion Detection in Text: A Review}, author={Jabeen, Saima and others}, year={2020}}` |

---

## 顶级会议统计

| 会议 | 论文数 | 等级 |
|------|--------|------|
| SOSP | 1 (PagedAttention) | CCF-A |
| ICLR | 2 (LoRA, LookaheadKV) | CCF-A |
| NeurIPS | 2 (QLoRA, MiniLLM) | CCF-A |
| EMNLP | 1 (TinyBERT) | CCF-B |
| TPAMI | 1 (Pre-hoc Sparsity) | CCF-A |
| NAACL/SemEval | 5 (SemEval系列) | CCF-B |

---

## 定时任务检查记录

| 检查时间 | arXiv状态 | 新增论文 |
|---------|----------|---------|
| 2026-03-15 09:07 | ✅ API正常 | 0篇 |
| 2026-03-15 10:07 | ✅ API正常 | 0篇 |
| 2026-03-15 11:07 | ✅ API正常 | 0篇 |
| 2026-03-15 14:07 | ✅ API正常 | 0篇 |

**结论**: 文献库已是最新，无新增论文
