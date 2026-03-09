# fine_tune_qwen35.py
"""
Qwen3.5-4B 微调脚本 (适配 Apple M2 MacBook Pro)
基于 Hugging Face PEFT + SFTTrainer，支持 LoRA 微调
支持 MPS (Metal Performance Shaders) 后端
"""

import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
import argparse

# 配置参数
MODEL_NAME = "Qwen/Qwen3.5-4B"
MAX_SEQ_LENGTH = 2048  # 可根据您的数据集调整
LORA_R = 16  # LoRA 秩
LORA_ALPHA = 16  # LoRA alpha 参数
LORA_DROPOUT = 0  # Dropout 率

def setup_environment():
    """设置训练环境"""
    print("🔧 设置训练环境...")
    
    # 对于 M2 Mac，确认 PyTorch 使用 MPS (Metal Performance Shaders)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ MPS (Metal) 可用，将使用: {device}")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS 不可用，回退到 CPU")
    
    return device

def download_and_prepare_model(load_in_4bit=True):
    """
    下载并准备模型
    在 Apple Silicon 上使用 float16 加载（MPS 不支持 bitsandbytes 量化）
    """
    print(f"📥 正在下载模型: {MODEL_NAME}")
    print(f"📏 最大序列长度: {MAX_SEQ_LENGTH}")
    
    # MPS 不支持 bitsandbytes 4-bit 量化，使用 float16 代替
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        model_max_length=MAX_SEQ_LENGTH,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    
    print(f"✅ 模型加载完成，内存占用: {get_model_size(model):.2f} GB")
    return model, tokenizer

def get_model_size(model):
    """估算模型内存占用"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024**3)  # 转换为 GB

def prepare_lora_model(model):
    """
    配置 LoRA 参数
    使用 Hugging Face PEFT 的 LoraConfig
    """
    print("🔧 配置 LoRA 参数...")
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 启用梯度检查点以节省内存
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print(f"✅ LoRA 配置完成 (r={LORA_R}, alpha={LORA_ALPHA})")
    return model

def load_custom_dataset(dataset_path, format_type="json"):
    """
    加载自定义数据集
    支持 JSON/JSONL 格式，参考指令微调格式 [citation:8]
    
    数据集格式示例 (JSONL):
    {"instruction": "任务指令", "input": "可选输入", "output": "期望输出"}
    {"instruction": "把下面内容总结为三点", "input": "长文本内容...", "output": "- 要点1\n- 要点2\n- 要点3"}
    """
    print(f"📊 正在加载数据集: {dataset_path}")
    
    try:
        if format_type == "json":
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        elif format_type == "csv":
            dataset = load_dataset("csv", data_files=dataset_path, split="train")
        else:
            raise ValueError(f"不支持的数据格式: {format_type}")
        
        print(f"✅ 数据集加载成功，共 {len(dataset)} 条样本")
        print(f"📋 数据集列名: {dataset.column_names}")
        
        # 显示前2条样本预览
        print("\n📝 数据预览:")
        for i in range(min(2, len(dataset))):
            print(f"  样本 {i+1}: {dataset[i]}")
        
        return dataset
    
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        print("请确保数据集格式正确，推荐使用 JSONL 格式")
        return None

def format_instruction_dataset(examples, tokenizer):
    """
    将指令数据集格式化为模型训练所需的格式
    参考 QLoRA 微调的数据格式 [citation:8]
    """
    texts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples.get("input", [""])[i]
        output = examples["output"][i]
        
        # 构建对话模板
        if input_text and input_text.strip():
            # 有输入的情况
            prompt = f"""<|im_start|>user
{instruction}\n\n{input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
        else:
            # 只有指令的情况
            prompt = f"""<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
        
        texts.append(prompt)
    
    return {"text": texts}

def setup_trainer(model, tokenizer, dataset, output_dir="./qwen35_finetuned"):
    """
    配置 SFTTrainer 训练器
    参考 Unsloth 的训练配置 [citation:1][citation:3]
    """
    print("⚙️ 配置训练器...")
    
    # 训练参数配置
    training_args = SFTConfig(
        # 输出设置
        output_dir=output_dir,
        # 训练批次
        per_device_train_batch_size=1,  # M2 16GB 内存下使用 batch size=1
        gradient_accumulation_steps=4,   # 梯度累积，模拟更大的 batch
        # 优化器设置
        optim="adamw_torch",             # 标准 AdamW（MPS 不支持 8-bit 优化器）
        learning_rate=2e-4,
        warmup_steps=10,
        # 训练步数
        max_steps=100,                    # 根据数据集大小调整
        # 日志和保存
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,               # 只保留最后2个检查点
        # 精度设置 - MPS 上使用 float32 避免 GradScaler 兼容性问题
        bf16=False,
        fp16=False,
        # 其他
        seed=3407,
        dataset_num_proc=1,
        report_to="none",                  # 不报告到外部服务
        # 必须的 SFTConfig 参数
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",         # 指定文本字段
    )
    
    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    
    print("✅ 训练器配置完成")
    return trainer

def save_and_export(model, tokenizer, output_dir, export_gguf=False):
    """
    保存模型并导出
    保存 LoRA 权重和合并后的完整模型
    """
    print(f"💾 正在保存模型到 {output_dir}")
    
    # 1. 保存 LoRA 权重（默认）
    lora_path = os.path.join(output_dir, "lora")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"✅ LoRA 权重已保存到 {lora_path}")
    
    # 2. 保存合并后的完整模型
    merged_path = os.path.join(output_dir, "merged_16bit")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"✅ 合并后的模型已保存到 {merged_path}")
    
    return lora_path

def push_to_huggingface(model, tokenizer, repo_id, token=None):
    """
    上传模型到 Huggingface Hub
    """
    print(f"☁️ 正在上传模型到 Huggingface: {repo_id}")
    
    if token:
        login(token=token)
    
    # 上传 LoRA 权重
    model.push_to_hub(repo_id, token=token)
    tokenizer.push_to_hub(repo_id, token=token)
    
    print(f"✅ 模型已上传到 https://huggingface.co/{repo_id}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qwen3.5-4B 微调脚本")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="数据集文件路径 (JSON/JSONL)")
    parser.add_argument("--output_dir", type=str, default="./qwen35_finetuned",
                        help="模型输出目录")
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="Huggingface 仓库ID (例如: username/qwen35-ft)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Huggingface 访问令牌")
    parser.add_argument("--steps", type=int, default=100,
                        help="训练步数")
    
    args = parser.parse_args()
    
    # 1. 设置环境
    device = setup_environment()
    
    # 2. 下载并准备模型
    model, tokenizer = download_and_prepare_model(load_in_4bit=True)
    
    # 3. 配置 LoRA
    model = prepare_lora_model(model)
    
    # 4. 加载数据集
    dataset = load_custom_dataset(args.dataset)
    if dataset is None:
        return
    
    # 5. 格式化数据集（如果需要）
    if "text" not in dataset.column_names:
        print("🔄 正在格式化数据集为指令微调格式...")
        dataset = dataset.map(
            lambda x: format_instruction_dataset(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
    
    # 6. 配置训练器
    trainer = setup_trainer(model, tokenizer, dataset, args.output_dir)
    
    # 7. 开始训练
    print("🚀 开始训练...")
    trainer.train()
    print("✅ 训练完成！")
    
    # 8. 保存模型
    save_and_export(model, tokenizer, args.output_dir)
    
    # 9. 上传到 Huggingface（如果指定了 repo）
    if args.hf_repo:
        push_to_huggingface(model, tokenizer, args.hf_repo, args.hf_token)
    
    print(f"\n🎉 所有步骤完成！模型保存在: {args.output_dir}")

if __name__ == "__main__":
    main()