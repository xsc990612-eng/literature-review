# Qwen3-8B 模型下载指南

## 方法一：ModelScope（国内最快）

```bash
# 安装modelscope
pip install modelscope

# 命令行下载
modelscope download --model Qwen/Qwen3-8B --local_dir ./Qwen3-8B

# 或使用Python下载
from modelscope import snapshot_download

model_dir = snapshot_download(
    model_id="Qwen/Qwen3-8B",
    local_dir="./Qwen3-8B",
    local_dir_use_symlinks=False
)
print(f"模型下载到: {model_dir}")
```

**启智平台/昇腾环境专用**：
```bash
# LLaMA Factory会自动通过ModelScope下载
export USE_MODELSCOPE_HUB=1

# 然后在WebUI或命令行指定模型名即可
--model_name_or_path Qwen/Qwen3-8B
```

---

## 方法二：HuggingFace（需要科学上网）

```bash
# 安装huggingface-cli
pip install huggingface-hub

# 下载（需登录）
huggingface-cli download Qwen/Qwen3-8B --local-dir ./Qwen3-8B

# 或使用git（需安装git-lfs）
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B
```

---

## 方法三：阿里云盘（国内备用）

官方提供的镜像：
- 链接：https://www.alipan.com/s/xxxxxx （搜索"Qwen3官方"）
- 或直接在ModelScope下载，速度一样快

---

## 方法四：启智社区模型库

1. 登录 [启智社区](https://openi.org.cn)
2. 进入 **模型** → 搜索 "Qwen3-8B"
3. 点击 **下载** 或使用API导入

---

## 模型文件说明

下载完成后目录结构：
```
Qwen3-8B/
├── config.json              # 模型配置
├── generation_config.json   # 生成配置
├── model.safetensors        # 模型权重（约16GB）
├── tokenizer.json           # 分词器
├── tokenizer_config.json    # 分词器配置
└── vocab.json               # 词表
```

**磁盘空间要求**：
- BF16模型：约 16GB
- INT4量化版：约 5GB（推理用）
- LoRA微调后：额外 +几十MB

---

## 启智平台实测下载代码

在启智的Notebook中直接运行：

```python
# 一键下载到启智工作目录
!pip install modelscope -q

from modelscope import snapshot_download
import os

# 下载到启智的持久化目录（避免重启丢失）
output_dir = "/home/ma-user/work/models/Qwen3-8B"
os.makedirs(output_dir, exist_ok=True)

model_dir = snapshot_download(
    model_id="Qwen/Qwen3-8B",
    local_dir=output_dir,
    local_dir_use_symlinks=False
)
print(f"✅ 模型已下载到: {model_dir}")
```

**下载耗时**：ModelScope在国内约 5-10分钟（取决于网速）

---

## 快速验证

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/home/ma-user/work/models/Qwen3-8B"  # 或 "Qwen/Qwen3-8B"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 测试生成
inputs = tokenizer("你好，请介绍一下自己", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 推荐选择

| 场景 | 推荐渠道 | 原因 |
|------|---------|------|
| 启智平台微调 | **ModelScope** | 国内速度快、LLaMA Factory原生支持 |
| 本地开发 | ModelScope / HF | 按需选择 |
| 急需离线使用 | 阿里云盘 | 可提前下载打包 |

**一句话总结**：启智平台直接设 `USE_MODELSCOPE_HUB=1`，LLaMA Factory会自动下载，不用手动操心。
