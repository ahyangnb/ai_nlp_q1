# Python 推理教程

训练完成后，`fine_tune_qwen35.py` 默认会保存两份模型产物：

```text
qwen35_finetuned/
  lora/          # LoRA 适配器，体积小，需要配合原始基座模型使用
  merged_16bit/  # 合并后的完整模型，最适合直接推理和后续转 GGUF
```

Python 推理适合快速验证微调效果，也方便继续接入自己的业务代码。

## 1. 准备环境

```bash
source .venv/bin/activate
pip install -r requirements_mac.txt
```

如果你还没有按推荐方式创建虚拟环境，建议使用 Python 3.11 或 3.12：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements_mac.txt
```

## 2. 使用合并模型推理

这是最简单的方式，直接加载 `merged_16bit`：

```bash
python infer_qwen35.py \
  --model_path ./qwen35_finetuned/merged_16bit \
  --prompt "解释什么是机器学习"
```

如果你的训练输出目录不是默认值，把路径换成自己的目录：

```bash
python infer_qwen35.py \
  --model_path ./my_model/merged_16bit \
  --prompt "将下面这句话翻译成英文：今天天气很好"
```

## 3. 使用 LoRA 适配器推理

如果只想加载 LoRA 适配器，需要同时指定原始基座模型：

```bash
python infer_qwen35.py \
  --model_path ./qwen35_finetuned/lora \
  --base_model Qwen/Qwen3.5-4B \
  --prompt "解释什么是机器学习"
```

这种方式首次运行时会下载基座模型。已经把基座模型下载到本地的话，也可以把 `--base_model` 改成本地路径。

## 4. 常用参数

```bash
python infer_qwen35.py \
  --model_path ./qwen35_finetuned/merged_16bit \
  --prompt "总结一下 LoRA 微调的优点" \
  --max_new_tokens 512 \
  --temperature 0 \
  --top_p 0.9 \
  --device auto
```

参数说明：

- `--max_new_tokens`: 最多生成多少个新 token
- `--temperature`: 随机性，越低越稳定；默认 `0` 使用贪心解码，MPS 上更稳
- `--top_p`: 采样范围，常用 `0.8` 到 `0.95`
- `--device`: 默认 `auto`，会优先使用 `cuda`，其次 `mps`，最后 `cpu`
- `--dtype`: 默认 `auto`，CUDA 使用 `float16`，MPS 使用更稳定的 `float32`

## 5. Mac Studio M2 Max 推荐命令

```bash
python infer_qwen35.py \
  --model_path ./qwen35_finetuned/merged_16bit \
  --prompt "解析下面句子的意思：年轻人说 HHHACCC 是什么意思" \
  --device mps \
  --dtype fp32 \
  --temperature 0 \
  --max_new_tokens 256
```
python infer_qwen35.py \
  --model_path ./qwen35_finetuned/merged_16bit \
  --prompt $'解析下面句子的意思\n\n年轻人说 HHHACCC 是什么意思' \
  --device mps \
  --dtype fp32 \
  --temperature 0 \
  --max_new_tokens 64


如果想要更随机的输出，可以把 `--temperature` 设为 `0.7`。如果 MPS 上出现 `probability tensor contains either inf, nan`，先恢复为 `--temperature 0 --dtype fp32`。

## 6. 常见问题

### 加载很慢

第一次运行可能需要下载模型，之后会走本地缓存。`merged_16bit` 体积也比较大，加载需要一点时间。

### MPS 报错或显存不够

可以先切到 CPU 验证流程：

```bash
python infer_qwen35.py \
  --model_path ./qwen35_finetuned/merged_16bit \
  --prompt "解释什么是机器学习" \
  --device cpu \
  --dtype fp32
```

如果 CPU 能跑、MPS 不能跑，通常是 PyTorch/MPS 兼容或内存峰值问题。

### 输出不像训练数据风格

优先检查三点：

- 训练步数是否太少
- 训练数据是否太少或格式不稳定
- 推理时是否加载了正确的 `merged_16bit` 或 `lora` 目录

### 输出一直是感叹号

如果输出变成连续的 `!!!!!!!!`，通常不是推理参数问题，而是训练时权重已经数值发散。可以先检查训练日志里是否出现：

```text
loss: 0.0
grad_norm: NaN
entropy: NaN
```

这时不要继续使用该模型，建议删除坏的输出目录后重新训练：

```bash
rm -rf ./qwen35_finetuned

python fine_tune_qwen35.py \
  --dataset ./my_dataset.jsonl \
  --steps 20 \
  --dtype fp32 \
  --learning_rate 5e-5 \
  --target_modules attention
```

如果你的数据只有几条样本，不建议一上来训练 100 步或 200 步。先用 10 到 20 步验证模型能正常回答，再逐步增加数据量和训练步数。
