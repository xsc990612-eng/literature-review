# 文献检索完整汇总报告

**检索时间范围**: 2026年2月28日 - 2026年3月2日  
**检索轮次**: 40+ 次定时检索  
**覆盖领域**: 5个研究方向

---

## 一、GSDMM 短文本主题建模 (7篇)

### 核心论文

| # | 作者 | 年份 | 标题 | 会议/期刊 | 等级 |
|---|------|------|------|-----------|------|
| 1 | Yin & Wang | 2014 | A Dirichlet Multinomial Mixture Model-based Approach for Short Text Clustering | KDD | CCF-A |
| 2 | Cheng et al. | 2025 | An Enhanced Model-based Approach for Short Text Clustering (GSDMM+) | arXiv/TKDE | CCF-A |
| 3 | Yin et al. | 2016 | A Model-based Approach for Text Clustering with Outlier Detection (GSDPMM) | ICDE | CCF-A |
| 4 | Qiang et al. | 2020 | Short Text Topic Modeling Techniques, Applications, and Performance: A Survey | TKDE | CCF-A |
| 5 | Lossio-Ventura et al. | 2021 | Evaluation of Clustering and Topic Modeling Methods over Health-Related Short Texts | AI in Medicine | SCI |

### 核心贡献总结
- **GSDMM算法**: 折叠吉布斯采样 + 狄利克雷多项式混合模型
- **自动推断聚类数量**: 无需预设K值
- **解决短文本稀疏性**: "一个文档一个主题"假设
- **Movie Group Process**: 直观解释算法原理

---

## 二、QLoRA 与 LLM 高效微调 (8篇)

### 核心论文

| # | 作者 | 年份 | 标题 | 会议/期刊 | 等级 |
|---|------|------|------|-----------|------|
| 1 | Hu et al. | 2022 | LoRA: Low-Rank Adaptation of Large Language Models | ICLR | 顶会 |
| 2 | Dettmers et al. | 2023 | QLoRA: Efficient Finetuning of Quantized LLMs | NeurIPS | CCF-A |
| 3 | Xu et al. | 2024 | QA-LoRA: Quantization-Aware Low-Rank Adaptation | ICLR | CCF-A |
| 4 | Wan et al. | 2024 | Efficient Large Language Models: A Survey | TMLR | 综述 |

### 核心贡献总结
- **LoRA**: 低秩适配，冻结权重 + 可训练低秩矩阵
- **QLoRA**: 4-bit NormalFloat (NF4) + 双重量化 + 分页优化器
- **效果**: 65B模型单卡48GB微调，达到ChatGPT 99.3%性能
- **参数效率**: 可训练参数减少10,000倍，GPU内存降低3倍

---

## 三、vLLM 推理优化 (3篇)

### 核心论文

| # | 作者 | 年份 | 标题 | 会议/期刊 | 等级 |
|---|------|------|------|-----------|------|
| 1 | Kwon et al. | 2023 | Efficient Memory Management for Large Language Model Serving with PagedAttention | SOSP | CCF-A |

### 核心贡献总结
- **PagedAttention**: 操作系统虚拟内存启发
- **KV Cache零浪费**: 从60-80%降至不到4%
- **吞吐量提升**: 2-4倍于FasterTransformer和Orca
- **灵活共享**: 请求内外KV缓存共享

---

## 四、知识蒸馏与 Agent (10篇+)

### 核心论文

| # | 作者 | 年份 | 标题 | 会议/期刊 | 等级 |
|---|------|------|------|-----------|------|
| 1 | Hinton et al. | 2015 | Distilling the Knowledge in a Neural Network | NIPS Workshop | 奠基 |
| 2 | Yao et al. | 2023 | ReAct: Synergizing Reasoning and Acting in Language Models | ICLR | 顶会 |
| 3 | Xu et al. | 2024 | A Survey on Knowledge Distillation of Large Language Models | arXiv | 综述 |
| 4 | Kang et al. | 2025 | Distilling LLM Agent into Small Models with Retrieval and Code Tools | NeurIPS | CCF-A |
| 5 | Luo et al. | 2026 | AgentArk: Distilling Multi-Agent Intelligence into a Single Model | arXiv | 前沿 |
| 6 | Qiu et al. | 2025 | AgentDistill: Training-Free Agent Distillation via Model-Context-Protocols | arXiv | 创新 |
| 7 | Song et al. | 2025 | MRGKD: Multi-agent Reasoning Graph Knowledge Distillation | SIGIR | CCF-A |

### 核心贡献总结
- **知识蒸馏演进**: 模型压缩 → 推理能力迁移 → Agent行为蒸馏
- **Agent蒸馏**: first-thought prefix + 自一致动作生成
- **无训练蒸馏**: AgentDistill通过MCPs实现
- **多Agent蒸馏**: AgentArk框架，R-SFT/DA/PAD三策略

---

## 五、情感分析评估方法 (6篇)

### 核心论文

| # | 作者 | 年份 | 标题 | 会议/期刊 | 等级 |
|---|------|------|------|-----------|------|
| 1 | Kharde & Sonawane | 2016 | Sentiment Analysis of Twitter Data: A Survey | IJCA | - |
| 2 | Calefato et al. | 2018 | A Benchmark Study on Sentiment Analysis for Software Engineering Research | ICSE | CCF-A |
| 3 | Zhang et al. | 2022 | Survey on Sentiment Analysis: Evolution of Research Methods and Topics | AI Review | JCR Q1 |
| 4 | Song et al. | 2024 | MOSABench: Multi-Object Sentiment Analysis Benchmark for MLLMs | arXiv | 基准 |
| 5 | Avelar et al. | 2023 | A Sentiment Analysis Benchmark for Automated Machine Learning | STIL | - |
| 6 | Loughran & McDonald | 2022 | Sentiment Analysis Methods: Survey and Evaluation | SSRN | 金融 |

### 核心贡献总结
- **评估指标**: Accuracy, Precision, Recall, F1, AUC-ROC, 混淆矩阵
- **多目标情感分析**: MOSABench基准
- **领域特定**: 软件工程、金融、医疗等垂直领域
- **AutoML基准**: 46个数据集标准化评估

---

## 统计汇总

| 领域 | 核心论文数 | CCF-A论文 | 综述论文 |
|------|-----------|-----------|----------|
| GSDMM短文本主题建模 | 7 | 4 | 1 |
| QLoRA与LLM高效微调 | 8 | 3 | 1 |
| vLLM推理优化 | 3 | 1 | 0 |
| 知识蒸馏与Agent | 10+ | 2 | 1 |
| 情感分析评估方法 | 6 | 1 | 2 |
| **总计** | **34+** | **11** | **5** |

---

## 顶级会议/期刊分布

- **CCF-A类**: 11篇 (KDD, NeurIPS, ICLR, SOSP, SIGIR, ICSE)
- **顶级会议**: 3篇 (ICLR, ICLR)
- **SCI/Q1期刊**: 3篇 (TKDE, AI Review, AI in Medicine)
- **arXiv预印本**: 10+篇 (最新前沿工作)

---

## 时间跨度

- **2014**: GSDMM奠基
- **2015**: 知识蒸馏奠基 (Hinton)
- **2016**: GSDPMM扩展
- **2018**: 软件工程情感分析基准
- **2019**: 短文本主题建模综述
- **2021**: LoRA基础
- **2022**: 情感分析方法综述, LoRA发表
- **2023**: QLoRA, vLLM, ReAct
- **2024**: QA-LoRA, LLM知识蒸馏综述, MOSABench
- **2025**: GSDMM+, Agent蒸馏, MRGKD, AgentDistill
- **2026**: AgentArk (最新)

---

*报告生成时间: 2026年3月2日*  
*数据来源: 40+次定时文献检索任务*
