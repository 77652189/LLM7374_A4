
import json
import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
from huggingface_hub import login

"""
Part 2 - DPO Fine-tuning
使用 LLaMA-3.2 3B Instruct + LoRA 进行 DPO 微调
训练数据: data/preference_dataset.jsonl
输出: PEFT adapter 上传到 HuggingFace Hub
"""

# ── 环境变量 ──────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN 未设置，请检查 .env 文件")
login(token=HF_TOKEN)

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_ID       = "meta-llama/Llama-3.2-3B-Instruct"
DATA_PATH  = Path(__file__).parent.parent / "part1_dataset" / "data" / "preference_dataset.jsonl"
OUTPUT_DIR = Path(__file__).parent / "output" / "dpo_model"
HF_REPO_ID     = "Nayc/llama-3.2-3b-dpo"

# 训练参数
NUM_EPOCHS     = 3
BATCH_SIZE     = 2
GRAD_ACCUM     = 4        # 等效 batch size = 8
LEARNING_RATE  = 5e-5
BETA           = 0.1      # DPO 温度参数
MAX_LENGTH     = 512
MAX_PROMPT_LEN = 256

# LoRA 参数
LORA_RANK      = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LORA_TARGET    = ["q_proj", "v_proj", "k_proj", "o_proj"]

# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_dataset(path: Path) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append({
                "prompt":   r["prompt"],
                "chosen":   r["chosen"],
                "rejected": r["rejected"],
            })
    dataset = Dataset.from_list(records)
    print(f"数据集加载完成: {len(dataset)} 条")
    print(f"示例:")
    print(f"  Prompt:   {dataset[0]['prompt'][:80]}...")
    print(f"  Chosen:   {dataset[0]['chosen'][:80]}...")
    print(f"  Rejected: {dataset[0]['rejected'][:80]}...")
    return dataset


# ── 模型加载 ──────────────────────────────────────────────────────────────────
def load_model(model_id: str):
    print(f"\n加载模型: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO 训练推荐 left padding

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    print("模型加载完成")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return tokenizer, model


# ── LoRA 配置 ─────────────────────────────────────────────────────────────────
def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    dataset = load_dataset(DATA_PATH)

    # 2. 加载模型
    tokenizer, model = load_model(MODEL_ID)

    # 3. 应用 LoRA
    print("\n应用 LoRA...")
    model = apply_lora(model)

    # 4. DPO 训练配置
    print("\n配置 DPO 训练...")
    dpo_config = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        max_length=MAX_LENGTH,
        bf16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
    )

    # 5. 初始化 DPOTrainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 6. 开始训练
    print("\n开始 DPO 训练...")
    print(f"  模型: {MODEL_ID}")
    print(f"  数据量: {len(dataset)} 条")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Beta: {BETA}")
    print(f"  LoRA rank: {LORA_RANK}\n")

    train_result = trainer.train()

    # 7. 打印训练结果
    print("\n训练完成！")
    print(f"  总步数: {train_result.global_step}")
    print(f"  训练损失: {train_result.training_loss:.4f}")

    # 8. 保存模型
    print(f"\n保存 PEFT adapter 到: {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # 9. 上传到 HuggingFace Hub
    print(f"\n上传到 HuggingFace: {HF_REPO_ID}")
    trainer.model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    print(f"✅ 上传完成！")
    print(f"模型链接: https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()