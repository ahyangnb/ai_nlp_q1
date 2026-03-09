使用市面上的ai大模型进行微调，尝试各种微调方法和算法。

# 介绍
使用阿里千问3.5-4B AI模型在常规日常使用设备【苹果Macbook M2 16G】，完整实现模型微调【LoRA方法】，现完整开源，后续将持续开源各种模型微调方法以及推理相关，帮助大家持续加强 NLP【自然语言处理】和VLP【视觉语言处理】，模型支持ollama/vLLM/TensorRT-LLM等推理引擎。


| 左对齐 | 居中对齐 | 右对齐 |
|:-------|:--------:|--------:|
| ![alt text](image-1.png) | ![alt text](image.png) | 右对齐内容 |
| 文本 | 文本 | 文本 |

想了解向量数据库，RAG，AI知识库等相关内容可以看：https://github.com/fluttercandies/pool_ai_knowledge

# 环境
使用python 3.9, 新版有太多不支持，我使用的是apple macbook pro M2运行。

# 使用
如果使用venv【项目虚拟空间管理】
```
python3 -m venv venv
source venv/bin/activate
```

然后
```
pip3 install -r requirements.txt
# 或直接安装 unsloth（会自动安装依赖）
pip3 install unsloth
```

# 基本用法（训练100步）
python fine_tune_qwen35.py --dataset ./my_dataset.jsonl

# 指定输出目录和训练步数
python fine_tune_qwen35.py --dataset ./my_dataset.jsonl --output_dir ./my_model --steps 200

# 训练后上传到 Huggingface
python fine_tune_qwen35.py --dataset ./my_dataset.jsonl --hf_repo yourusername/qwen35-finetuned --hf_token YOUR_TOKEN

# 训练后同时导出 GGUF 格式（适合 llama.cpp 部署）
python fine_tune_qwen35.py --dataset ./my_dataset.jsonl --export_gguf

# 📤 上传到 Huggingface 的完整流程
```
# 1. 安装并登录 huggingface-cli
pip install huggingface-hub
huggingface-cli login
# 按提示输入您的 token

# 2. 创建模型仓库（可选，会自动创建）
huggingface-cli repo create your-model-name

# 3. 运行脚本并上传
python fine_tune_qwen35.py --dataset ./data.jsonl --hf_repo yourusername/your-model-name
```
上传后，您的模型将出现在 https://huggingface.co/yourusername/your-model-name，其他人可以直接使用或下载。
