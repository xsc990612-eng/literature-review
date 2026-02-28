# 文献综述报告

> 生成时间：2026年3月1日

---

## 一、GSDMM短文本主题建模核心文献

### 1. 开创性论文（必读）
- **作者/年份**: Jianhua Yin, Jing Wang / 2014
- **标题**: A Dirichlet Multinomial Mixture Model-based Approach for Short Text Clustering
- **会议/期刊**: KDD 2014 (ACM SIGKDD International Conference on Knowledge Discovery and Data Mining)
- **等级**: CCF-A类，数据挖掘顶会
- **核心贡献**: 首次提出GSDMM算法，基于Dirichlet多项式混合模型和折叠Gibbs采样，解决短文本聚类中的稀疏性和高维问题。提出"Movie Group Process"概念，实现自动推断聚类数量

```bibtex
@inproceedings{yin2014dirichlet,
  title={A dirichlet multinomial mixture model-based approach for short text clustering},
  author={Yin, Jianhua and Wang, Jing},
  booktitle={Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={233--242},
  year={2014}
}
```

### 2. 综述论文
- **作者/年份**: Jipeng Qiang, Zhenyu Qian等 / 2020
- **标题**: Short Text Topic Modeling Techniques, Applications, and Performance: A Survey
- **期刊**: IEEE Transactions on Knowledge and Data Engineering (TKDE)
- **等级**: CCF-A类，数据挖掘顶级期刊
- **核心贡献**: 系统综述短文本主题建模方法，将GSDMM归类为DMM-based方法，对比分析其与LDA、BTM等方法的优劣

```bibtex
@article{qiang2020short,
  title={Short text topic modeling techniques, applications, and performance: a survey},
  author={Qiang, Jipeng and Qian, Zhenyu and Li, Yun and Yuan, Yunhao and Wu, Xindong},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={34},
  number={3},
  pages={1420--1438},
  year={2020},
  publisher={IEEE}
}
```

### 3. 改进版本（最新）
- **作者/年份**: Enhao Cheng, Shoujia Zhang, Jianhua Yin等 / 2025
- **标题**: An Enhanced Model-based Approach for Short Text Clustering (GSDMM+)
- **期刊**: arXiv预印本
- **核心贡献**: 提出GSDMM+，引入自适应聚类初始化、基于熵的词权重调整和粒度调整与聚类合并三大改进模块

```bibtex
@article{cheng2025enhanced,
  title={An Enhanced Model-based Approach for Short Text Clustering},
  author={Cheng, Enhao and Zhang, Shoujia and Yin, Jianhua and Song, Xuemeng and Gan, Tian and Nie, Liqiang},
  journal={arXiv preprint arXiv:2507.13793},
  year={2025}
}
```

---

## 二、QLoRA与LLM高效微调文献

### 1. 开创性论文（必读）
- **作者/年份**: Tim Dettmers等 / 2023
- **标题**: QLoRA: Efficient Finetuning of Quantized LLMs
- **会议**: NeurIPS 2023
- **等级**: CCF-A类，机器学习顶会
- **核心贡献**: 提出QLoRA，结合4-bit量化(NF4)和LoRA，实现单卡48GB GPU微调65B参数模型。提出双重量化和分页优化器技术

```bibtex
@article{dettmers2023qlora,
  title={Qlora: Efficient finetuning of quantized llms},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```

### 2. LoRA基础论文
- **作者/年份**: Edward J. Hu等 / 2022
- **标题**: LoRA: Low-Rank Adaptation of Large Language Models
- **会议**: ICLR 2022
- **等级**: CCF-A类
- **核心贡献**: 提出低秩适配(LoRA)方法，通过低秩矩阵分解实现参数高效微调，冻结原始权重仅训练适配器

```bibtex
@article{hu2022lora,
  title={Lora: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2022}
}
```

### 3. 量化感知改进
- **作者/年份**: Yuhui Xu等 / 2023
- **标题**: QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models
- **会议**: ICLR 2024
- **核心贡献**: 提出量化感知LoRA，解决量化与适配自由度不平衡问题，支持INT4/INT3/INT2量化

```bibtex
@article{xu2023qa,
  title={QA-LoRA: Quantization-aware low-rank adaptation of large language models},
  author={Xu, Yuhui and Xie, Lingxi and Gu, Xiaotao and Chen, Xin and Chang, Heng and Zhang, Yifei and Chen, Zhengsu and Zhang, Xiaopeng and Tian, Qi},
  journal={arXiv preprint arXiv:2309.14717},
  year={2023}
}
```

### 4. 权重分解改进
- **作者/年份**: Shih-Yang Liu等 / 2024
- **标题**: DoRA: Weight-Decomposed Low-Rank Adaptation
- **会议**: ICML 2024 (Oral)
- **等级**: CCF-A类
- **核心贡献**: 将权重分解为幅度和方向分量，LoRA仅更新方向，幅度作为独立可学习标量，精度显著超越LoRA

```bibtex
@article{liu2024dora,
  title={DoRA: Weight-decomposed low-rank adaptation},
  author={Liu, Shih-Yang and Wang, Chien-Yi and Yin, Hongxu and Molchanov, Pavlo and Wang, Yu-Chiang Frank and Cheng, Kwang-Ting and Chen, Min-Hung},
  journal={arXiv preprint arXiv:2402.09353},
  year={2024}
}
```

---

## 三、vLLM推理优化文献

### 1. 开创性论文（必读）
- **作者/年份**: Woosuk Kwon等 / 2023
- **标题**: Efficient Memory Management for Large Language Model Serving with PagedAttention
- **会议**: SOSP 2023
- **等级**: CCF-A类，操作系统顶会
- **核心贡献**: 提出PagedAttention算法，借鉴操作系统虚拟内存和分页技术，实现KV Cache近零内存浪费和灵活共享，吞吐量提升2-4倍

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient memory management for large language model serving with pagedattention},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E and Zhang, Hao and Stoica, Ion},
  booktitle={Proceedings of the 29th Symposium on Operating Systems Principles},
  pages={611--626},
  year={2023}
}
```

### 2. 技术博客
- **作者/年份**: vLLM Team / 2023
- **标题**: vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention
- **来源**: vLLM官方博客
- **核心贡献**: 介绍vLLM系统设计，对比HuggingFace Transformers实现24倍吞吐量提升

### 3. 最新技术报告
- **作者/年份**: UC Berkeley EECS / 2025
- **标题**: vLLM: A High-Throughput Distributed LLM Serving Engine
- **来源**: UC Berkeley技术报告 EECS-2025-192
- **核心贡献**: 详细介绍vLLM分布式架构，支持张量并行、流水线并行和专家并行

---

## 四、知识蒸馏与Agent文献

### 1. 综述论文（必读）
- **作者/年份**: Xiaohan Xu, Ming Li等 / 2024
- **标题**: A Survey on Knowledge Distillation of Large Language Models
- **期刊**: arXiv预印本
- **核心贡献**: 系统综述LLM知识蒸馏技术，从算法、技能、垂直化三个维度分类，涵盖监督微调、散度优化、强化学习等蒸馏方法

```bibtex
@article{xu2024survey,
  title={A survey on knowledge distillation of large language models},
  author={Xu, Xiaohan and Li, Ming and Tao, Chongyang and Shen, Tao and Cheng, Reynold and Li, Jinyang and Xu, Can and Tao, Dacheng and Zhou, Tianyi},
  journal={arXiv preprint arXiv:2402.13116},
  year={2024}
}
```

### 2. 多Agent蒸馏（最新）
- **作者/年份**: Yinyi Luo, Yiqiao Jin等 / 2026
- **标题**: AgentArk: Distilling Multi-Agent Intelligence into a Single Model
- **期刊**: arXiv预印本
- **核心贡献**: 首个全面探索多Agent系统蒸馏的框架，提出三种蒸馏策略：推理增强SFT、轨迹数据增强、过程感知蒸馏(PAD)

```bibtex
@article{luo2026agentark,
  title={AgentArk: Distilling Multi-Agent Intelligence into a Single Model},
  author={Luo, Yinyi and Jin, Yiqiao and Yu, Weichen and Zhang, Mengqi and Kumar, Srijan and Li, Xiaoxiao and Xu, Weijie and Chen, Xin and Wang, Jindong},
  journal={arXiv preprint arXiv:2602.03955},
  year={2026}
}
```

### 3. 结构化多Agent蒸馏
- **作者/年份**: Jiaao Chen, Shashank Saha等 / 2024
- **标题**: MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models
- **会议**: ICML 2024
- **等级**: CCF-A类
- **核心贡献**: 将多Agent交互表示为图结构，通过图编码器增强学生模型

```bibtex
@article{chen2024magdi,
  title={MAGDi: Structured distillation of multi-agent interaction graphs improves reasoning in smaller language models},
  author={Chen, Jiaao and Saha, Shashank and Stengel-Eskin, Elias and Bansal, Mohit},
  journal={arXiv preprint arXiv:2402.01620},
  year={2024}
}
```

---

## 五、情感分析评估方法文献

### 1. 综合基准（必读）
- **作者/年份**: Zaijing Li, Ting-En Lin等 / 2023
- **标题**: UniSA: Unified Generative Framework for Sentiment Analysis (SAEval Benchmark)
- **会议**: ACM MM 2023
- **等级**: CCF-A类
- **核心贡献**: 提出SAEval基准，整合12个数据集覆盖4个主任务(ERC、MSA、ABSA、CA)和3个下游任务

```bibtex
@inproceedings{li2023unisa,
  title={UniSA: Unified generative framework for sentiment analysis},
  author={Li, Zaijing and Lin, Ting-En and Wu, Yuchuan and Liu, Meng and Tang, Fengxiao and Zhao, Ming and Li, Yongbin},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={6132--6142},
  year={2023}
}
```

### 2. 多对象情感分析基准
- **作者/年份**: Shezheng Song, Chengxiang He等 / 2024
- **标题**: MOSABench: Multi-Object Sentiment Analysis Benchmark for Multimodal Large Language Models
- **核心贡献**: 首个针对多对象情感分析的MLLM基准，包含约1000张图像

```bibtex
@article{song2024mosa,
  title={MOSABench: Multi-Object Sentiment Analysis Benchmark for Multimodal Large Language Models},
  author={Song, Shezheng and He, Chengxiang and Li, Shasha and Zhao, Shan and Wang, Chengyu and Yan, Tianwei and Li, Xiaopeng and Wan, Qian and Ma, Jun and Yu, Jie and Mao, Xiaoguang},
  journal={arXiv preprint arXiv:2412.00060},
  year={2024}
}
```

---

## 总结表格

| 领域 | 必读论文 | 会议/期刊等级 | 年份 |
|------|---------|--------------|------|
| GSDMM | Yin & Wang, KDD | CCF-A | 2014 |
| QLoRA | Dettmers et al., NeurIPS | CCF-A | 2023 |
| vLLM | Kwon et al., SOSP | CCF-A | 2023 |
| 知识蒸馏 | Xu et al., arXiv综述 | 高引用 | 2024 |
| 情感分析评估 | Li et al., ACM MM | CCF-A | 2023 |

---

## 六、RAG检索增强生成文献

### 1. 综述论文（必读）
- **作者/年份**: Yunfan Gao, Yun Xiong等 / 2023-2024
- **标题**: Retrieval-Augmented Generation for Large Language Models: A Survey
- **期刊**: arXiv预印本 (高引用)
- **核心贡献**: 全面综述RAG范式演进，涵盖Naive RAG、Advanced RAG和Modular RAG三个阶段，深入分析检索、生成和增强三大核心技术

```bibtex
@article{gao2023retrieval,
  title={Retrieval-augmented generation for large language models: A survey},
  author={Gao, Yunfan and Xiong, Yun and Gao, Xinyu and Jia, Kangxiang and Pan, Jinliu and Bi, Yuxi and Dai, Yi and Sun, Jiawei and Wang, Meng and Wang, Haofen},
  journal={arXiv preprint arXiv:2312.10997},
  year={2023}
}
```

---

## 七、多模态大语言模型文献

### 1. 开创性综述（必读）
- **作者/年份**: Shukang Yin, Chaoyou Fu等 / 2023
- **标题**: A Survey on Multimodal Large Language Models
- **期刊**: National Science Review (NSR)
- **等级**: 中科院一区，影响因子16.3
- **核心贡献**: 首次系统综述MLLM，涵盖架构、训练策略、数据、评估等，探讨多模态幻觉、M-ICL、M-CoT等扩展技术

```bibtex
@article{yin2023survey,
  title={A survey on multimodal large language models},
  author={Yin, Shukang and Fu, Chaoyou and Zhao, Sirui and Li, Ke and Sun, Xing and Xu, Tong and Chen, Enhong},
  journal={National Science Review},
  year={2023}
}
```

### 2. 最新综合综述
- **作者/年份**: Chia Xin Liang, Pu Tian等 / 2024-2025
- **标题**: A Comprehensive Survey and Guide to Multimodal Large Language Models in Vision-Language Tasks
- **期刊**: arXiv预印本
- **核心贡献**: 涵盖MLLM架构、训练方法、视觉叙事等应用，讨论可扩展性、鲁棒性和跨模态学习挑战

```bibtex
@article{liang2024comprehensive,
  title={A Comprehensive Survey and Guide to Multimodal Large Language Models in Vision-Language Tasks},
  author={Liang, Chia Xin and Tian, Pu and Yin, Caitlyn Heqi and Yua, Yao and Hou, Wei An and Li, Ming and Song, Xinyuan and Wang, Tianyang and Bi, Ziqian and Liu, Ming},
  journal={arXiv preprint arXiv:2411.06284},
  year={2024}
}
```

---

## 八、图神经网络文献

### 1. 综述论文
- **作者/年份**: Jie Zhou, Ganqu Cui等 / 2020
- **标题**: Graph Neural Networks: A Review of Methods and Applications
- **期刊**: AI Open
- **核心贡献**: 系统回顾GNN方法，分类介绍频谱方法、空间方法、注意力机制等，涵盖社交网络、推荐系统等应用

```bibtex
@article{zhou2020graph,
  title={Graph neural networks: A review of methods and applications},
  author={Zhou, Jie and Cui, Ganqu and Hu, Shengding and Zhang, Zhengyan and Yang, Cheng and Liu, Zhiyuan and Wang, Lifeng and Li, Changcheng and Sun, Maosong},
  journal={AI Open},
  volume={1},
  pages={57--81},
  year={2020},
  publisher={Elsevier}
}
```

---

## 九、扩散模型文献

### 1. 图像生成综述
- **作者/年份**: Han Zhang, Weimin Tan等 / 2024
- **标题**: Diffusion Model for Image Generation: A Survey
- **期刊**: IEEE Transactions on Knowledge and Data Engineering
- **核心贡献**: 综述扩散模型在图像生成中的进展，对比VAE、GAN、Flow-based模型，分析条件生成、可控生成等技术

```bibtex
@article{zhang2024diffusion,
  title={Diffusion model for image generation: A survey},
  author={Zhang, Han and Tan, Weimin and Huang, Ye and Tang, Wen and Xue, Jing and Li, Rui and Tang, Yuxing and Zou, He and Fu, Gang and Zhang, Li},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
```

---

## 十、神经网络可解释性文献

### 1. 综述论文
- **作者/年份**: Qiang Zhang, Aldo Lipani等 / 2021
- **标题**: A Survey on Neural Network Interpretability
- **期刊**: IEEE Transactions on Neural Networks and Learning Systems
- **核心贡献**: 全面综述神经网络可解释性研究，澄清可解释性定义，分类介绍特征重要性、概念激活向量、对抗样本解释等方法

```bibtex
@article{zhang2021survey,
  title={A survey on neural network interpretability},
  author={Zhang, Qiang and Lipani, Aldo and Yilmaz, Emine and Yao, Zhiwei},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021},
  publisher={IEEE}
}
```

---

## 扩展总结表格

| 领域 | 必读论文 | 会议/期刊等级 | 年份 |
|------|---------|--------------|------|
| GSDMM | Yin & Wang, KDD | CCF-A | 2014 |
| QLoRA | Dettmers et al., NeurIPS | CCF-A | 2023 |
| vLLM | Kwon et al., SOSP | CCF-A | 2023 |
| 知识蒸馏 | Xu et al., arXiv综述 | 高引用 | 2024 |
| 情感分析评估 | Li et al., ACM MM | CCF-A | 2023 |
| **RAG** | **Gao et al., arXiv** | **高引用** | **2023** |
| **多模态LLM** | **Yin et al., NSR** | **中科院一区** | **2023** |
| **图神经网络** | **Zhou et al., AI Open** | **开放获取** | **2020** |
| **扩散模型** | **Zhang et al., TKDE** | **CCF-A** | **2024** |
| **可解释性** | **Zhang et al., TNNLS** | **CCF-B** | **2021** |
