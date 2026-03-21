# 启智社区 + 昇腾NPU + Qwen3:8B 微调完整指南

## 一、方案选择对比

| 方案 | 难度 | 特点 | 推荐指数 |
|------|------|------|---------|
| **LLaMA Factory** | ⭐⭐ 简单 | 零代码、WebUI、支持昇腾NPU官方适配 | ⭐⭐⭐⭐⭐ **推荐** |
| **MS-Swift** | ⭐⭐⭐ 中等 | 阿里出品、多模态支持好、命令行友好 | ⭐⭐⭐⭐ |
| **MindSpeed LLM** | ⭐⭐⭐⭐ 较难 | 华为官方、性能最优、适合大规模分布式 | ⭐⭐⭐ |

**新手建议：直接使用 LLaMA Factory**，无需写代码，Web界面配置即可。

---

## 二、启智平台环境准备

### 1. 创建云脑调试任务

登录 [启智社区](https://openi.org.cn) → 进入项目 → **云脑** → **新建调试任务**

**关键配置：**

| 配置项 | 推荐选择 | 说明 |
|--------|---------|------|
| 算力集群 | `智算网络集群(Beta)` | 昇腾910B资源 |
| 计算资源 | `昇腾NPU` | 选择NPU类型 |
| 资源规格 | `NPU: 1*Ascend-D910B, CPU: 20, 显存: 32GB` | Qwen3-8B最低配置 |
| 镜像 | `mindtorch0.3_mindspore2.3_torchnpu2.3.1_cann8.0` | 预装CANN和torch-npu |
| 访问Internet | `是` | 需要下载模型和数据集 |

> 💡 **提示**：Qwen3-8B微调至少需要 **32GB NPU显存**，选择单卡910B(32GB)刚好够用，如果Batch Size设大了会OOM。

---

## 三、方案一：LLaMA Factory 微调（推荐）

### Step 1: 环境初始化

进入启智平台的Jupyter Notebook或Terminal：

```bash
# 1. 激活昇腾环境变量（每次开机都要执行）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 安装昇腾算子依赖（启智平台通常已预装，如未安装执行）
pip uninstall te topi hccl -y
pip install sympy
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl

# 3. 克隆并安装 LLaMA Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch-npu,metrics]"

# 4. 设置模型下载源（国内用ModelScope）
export USE_MODELSCOPE_HUB=1

# 5. 验证安装
llamafactory-cli env
```

**预期输出**：
```
- `llamafactory` version: 0.9.x
- PyTorch version: 2.3.0 (NPU)
- CANN version: 8.0.x
- NPU type: Ascend910B
```

### Step 2: 准备数据集

将数据集放在 `LLaMA-Factory/data` 目录下，格式如下：

```json
[
  {
    "instruction": "请分析这条电商评论的情感倾向",
    "input": "这个产品质量很差，完全不值这个价",
    "output": "负面"
  },
  {
    "instruction": "请分析这条电商评论的情感倾向",
    "input": "非常满意，物流很快，客服态度也很好",
    "output": "正面"
  }
]
```

然后在 `dataset_info.json` 中注册：

```json
{
  "sentiment_analysis": {
    "file_name": "sentiment_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

### Step 3: 启动 WebUI 进行微调

```bash
# 在启智平台的Terminal中执行
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/LLaMA-Factory
export USE_MODELSCOPE_HUB=1

# 启动Web界面
llamafactory-cli webui
```

**启智平台特殊处理**：
- 点击右侧 **"查看 notebook"** 旁边的端口转发按钮
- 或使用 **SSH端口转发**：`ssh -p <端口> -L 7860:localhost:7860 user@<启智IP>`
- 然后在本地浏览器访问 `http://localhost:7860`

### Step 4: WebUI 配置参数

| 配置项 | 推荐设置 | 说明 |
|--------|---------|------|
| 模型名称 | `Qwen3-8B` | 选择Qwen3-8B或Qwen3-8B-Thinking |
| 模型路径 | `Qwen/Qwen3-8B` | ModelScope会自动下载 |
| 训练方式 | `LoRA` | 参数高效微调 |
| 数据路径 | `sentiment_analysis` | 刚才注册的数据集名 |
| 学习率 | `5e-5` | LoRA推荐学习率 |
| 训练轮数 | `3` | 根据数据量调整 |
| Batch Size | `1` | 910B 32GB建议设为1 |
| 梯度累积 | `8` | 等效Batch Size=8 |
| LoRA Rank | `8` | 一般8-64 |
| LoRA Alpha | `32` | 通常设为rank的2-4倍 |
| 量化 | `不开启` | 如需省显存可开INT8/BF16 |

**关键配置截图参考**：
- 模型选择: `qwen3-8b` (对话模板选 `qwen3`)
- 训练类型: `lora`
- 数据类型: `bf16` (昇腾910B支持BF16)

### Step 5: 开始训练

点击 **"开始训练"** 按钮，观察日志：

```bash
# 正常训练日志示例
[INFO|trainer.py:xxx] ***** Running training *****
[INFO|trainer.py:xxx]   Num examples = 1000
[INFO|trainer.py:xxx]   Num Epochs = 3
[INFO|trainer.py:xxx]   Batch size per device = 1
...
# loss 应该逐渐下降
{'loss': 2.1345, 'learning_rate': 4.8e-05, 'epoch': 0.1}
{'loss': 1.8234, 'learning_rate': 4.5e-05, 'epoch': 0.2}
```

**显存监控**（另一个Terminal）：
```bash
npu-smi info
# 查看显存占用，正常应在 28-32GB 之间
```

### Step 6: 导出与推理

训练完成后，导出合并模型：

```bash
# 合并LoRA权重到原模型
ASCEND_RT_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path Qwen/Qwen3-8B \
    --adapter_name_or_path ./saves/Qwen3-8B/lora/<训练时间戳>/ \
    --template qwen3 \
    --finetuning_type lora \
    --export_dir ./qwen3_8b_finetuned \
    --export_size 5 \
    --export_device auto
```

推理测试：
```bash
ASCEND_RT_VISIBLE_DEVICES=0 llamafactory-cli chat \
    --model_name_or_path ./qwen3_8b_finetuned \
    --template qwen3
```

---

## 四、方案二：MS-Swift 微调（命令行偏好）

```bash
# 安装swift
pip install ms-swift -U
pip install torch-npu==2.3.1 decorator

# 准备数据集（JSON格式）
# 然后执行训练
ASCEND_RT_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type lora \
    --dataset <数据集路径> \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --max_length 2048 \
    --output_dir ./output
```

---

## 五、常见问题解决

### 1. OOM (显存溢出)

**症状**：`RuntimeError: NPU out of memory`

**解决**：
```bash
# 减小 batch size
--per_device_train_batch_size 1

# 增加梯度累积（保持等效batch size）
--gradient_accumulation_steps 16

# 开启梯度检查点
--gradient_checkpointing true

# 使用QLoRA（4-bit量化）
--quantization_bit 4
```

### 2. CANN环境未找到

**症状**：`ASCEND_HOME_PATH not set` 或找不到CANN

**解决**：
```bash
# 确认环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证CANN安装
cat /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info
```

### 3. 模型下载慢/失败

**解决**：
```bash
# 使用ModelScope镜像（国内快）
export USE_MODELSCOPE_HUB=1

# 或在WebUI中设置模型路径为本地路径
--model_name_or_path /home/ma-user/work/models/Qwen3-8B
```

### 4. 启智平台端口转发问题

**症状**：WebUI启动后本地无法访问

**解决**：
```bash
# 方法1: 使用启智平台自带的端口转发功能
# 在"调试任务"页面点击"端口转发"

# 方法2: SSH隧道（本地终端执行）
ssh -N -f -L 7860:localhost:7860 <用户名>@<启智节点IP> -p <SSH端口>
# 然后本地浏览器访问 http://localhost:7860
```

---

## 六、性能参考

| 配置 | 训练速度 | 显存占用 |
|------|---------|---------|
| Qwen3-8B + LoRA(r=8) + BF16 + Batch=1 | ~2-3s/iter | ~28GB |
| Qwen3-8B + QLoRA(4bit) + Batch=1 | ~1.5s/iter | ~18GB |
| Qwen3-8B + LoRA + Batch=2 | OOM | >32GB |

---

## 七、进阶建议

1. **多卡训练**：如果申请到多卡资源，LLaMA Factory自动支持DeepSpeed ZeRO-2/3
2. **混合精度**：昇腾910B原生支持BF16，比FP16更稳定
3. **数据预处理**：大量数据建议使用 `datasets` 库的流式加载
4. **监控**：使用 `npu-smi info` 和 `htop` 实时监控资源

---

## 参考链接

- [LLaMA Factory 昇腾NPU官方文档](https://llamafactory.readthedocs.io/zh-cn/latest/advanced/npu.html)
- [启智社区昇腾910B使用指南](https://openi.org.cn/docs)
- [华为昇腾模型迁移教程](https://ascend.github.io/docs/sources/llamafactory/example.html)

---

**有具体问题随时问我，我可以帮你排查报错或优化配置！**
