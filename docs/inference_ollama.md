# Ollama 推理教程

Ollama 适合把微调后的模型变成本地服务，之后可以用 `ollama run`、HTTP API 或其他客户端调用。

推荐流程是：

```text
训练完成 -> merged_16bit 合并模型 -> 转 GGUF -> 编写 Modelfile -> ollama create -> ollama run
```

## 1. 准备训练产物

先确保训练已经完成，并且存在合并模型目录：

```bash
ls ./qwen35_finetuned/merged_16bit
```

如果你训练时指定了输出目录，例如：

```bash
python fine_tune_qwen35.py \
  --dataset ./my_dataset.jsonl \
  --output_dir ./my_model \
  --steps 200
```

那么合并模型目录就是：

```text
./my_model/merged_16bit
```

## 2. 安装 Ollama

macOS 可以直接下载安装：

```bash
brew install --cask ollama
```

或者到官网下载安装包：

```text
https://ollama.com/download
```

启动 Ollama：

```bash
ollama serve
```

如果你已经通过桌面应用启动了 Ollama，这一步可以跳过。

## 3. 安装 llama.cpp 转换工具

Ollama 更适合加载 GGUF 文件，所以需要先把 Hugging Face 格式的合并模型转成 GGUF。

建议把 llama.cpp 放在项目同级目录：

```bash
cd ..
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
python3 -m pip install -r requirements.txt
cmake -B build
cmake --build build --config Release -j
cd ../ai_nlp_q1
```

## 4. 转换为 GGUF

先转换成 F16 GGUF：

```bash
python ../llama.cpp/convert_hf_to_gguf.py \
  ./qwen35_finetuned/merged_16bit \
  --outfile ./qwen35-finetuned-f16.gguf \
  --outtype f16
```

F16 文件比较大。为了更适合本地长期运行，建议再量化成 Q4_K_M：

```bash
../llama.cpp/build/bin/llama-quantize \
  ./qwen35-finetuned-f16.gguf \
  ./qwen35-finetuned-q4_k_m.gguf \
  Q4_K_M
```

如果你的 llama.cpp 构建产物路径不同，可以查找一下：

```bash
find ../llama.cpp/build -name "llama-quantize" -type f
```

## 5. 编写 Modelfile

在项目根目录创建 `Modelfile.qwen35-finetuned`：

```text
FROM ./qwen35-finetuned-q4_k_m.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """你是一个经过本地数据微调的中文 AI 助手。回答要准确、简洁。"""
```

如果你的训练数据用了不同的 prompt 模板，需要让这里的 `TEMPLATE` 和训练时保持一致。

## 6. 创建 Ollama 模型

```bash
ollama create qwen35-finetuned -f Modelfile.qwen35-finetuned
```

查看模型是否创建成功：

```bash
ollama list
```

## 7. 本地运行

```bash
ollama run qwen35-finetuned
```

也可以直接传入问题：

```bash
ollama run qwen35-finetuned "解释什么是机器学习"
```

## 8. 通过 HTTP API 调用

Ollama 默认服务地址是 `http://localhost:11434`。

```bash
curl http://localhost:11434/api/generate \
  -d '{
    "model": "qwen35-finetuned",
    "prompt": "总结一下 LoRA 微调的优点",
    "stream": false
  }'
```

## 9. 常见问题

### Ollama 不能直接加载 `merged_16bit`

`merged_16bit` 是 Hugging Face Transformers 格式。Ollama 本地运行通常使用 GGUF，所以需要先转换。

### 量化后效果变差

可以尝试更高精度的量化：

```bash
../llama.cpp/build/bin/llama-quantize \
  ./qwen35-finetuned-f16.gguf \
  ./qwen35-finetuned-q5_k_m.gguf \
  Q5_K_M
```

如果你更看重效果而不是体积，也可以直接在 Modelfile 里使用 F16 GGUF：

```text
FROM ./qwen35-finetuned-f16.gguf
```

### 回答格式不稳定

优先检查 `Modelfile` 的 `TEMPLATE` 是否和训练脚本中的格式一致。本项目训练时使用的是：

```text
<|im_start|>user
用户内容<|im_end|>
<|im_start|>assistant
助手内容<|im_end|>
```

### Mac 上速度慢

可以优先使用 `Q4_K_M` 量化版本。M2 Max 32GB 通常可以运行 4B 级别模型，但 F16 会更吃内存。
