
import json
import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import DPOTrainer, DPOConfig
from huggingface_hub import login

"""
Part 3 - Iterative DPO: Round 2 训练
在第1轮 DPO 模型基础上继续训练
数据: data/preference_dataset_round2.jsonl
输出: PEFT adapter 上传到 HuggingFace Hub
"""


# ── 环境变量 ──────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN 未设置，请检查 .env 文件")
login(token=HF_TOKEN)

# ── 配置 ──────────────────────────────────────────────────────────────────────
BASE_MODEL_ID  = "meta-llama/Llama-3.2-3B-Instruct"
ROUND1_PEFT_ID = "Nayc/llama-3.2-3b-dpo"
DATA_PATH      = Path(__file__).parent / "data" / "preference_dataset_round2.jsonl"
OUTPUT_DIR     = Path(__file__).parent / "output" / "dpo_model_round2"
HF_REPO_ID     = "Nayc/llama-3.2-3b-dpo-round2"

# 训练参数（与第1轮相同）
NUM_EPOCHS    = 3
BATCH_SIZE    = 2
GRAD_ACCUM    = 4
LEARNING_RATE = 5e-5
BETA          = 0.1
MAX_LENGTH    = 512

# LoRA 参数
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGET  = ["q_proj", "v_proj", "k_proj", "o_proj"]


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
    print(f"Round 2 数据集: {len(dataset)} 条")
    print(f"示例:")
    print(f"  Prompt:   {dataset[0]['prompt'][:80]}...")
    print(f"  Chosen:   {dataset[0]['chosen'][:80]}...")
    print(f"  Rejected: {dataset[0]['rejected'][:80]}...")
    return dataset


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    dataset = load_dataset(DATA_PATH)

    # 2. 加载 base 模型
    print(f"\n加载 base 模型: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )

    # 3. 加载第1轮 LoRA adapter
    print(f"\n加载第1轮 DPO adapter: {ROUND1_PEFT_ID}")
    model = PeftModel.from_pretrained(
        base_model,
        ROUND1_PEFT_ID,
        token=HF_TOKEN,
        is_trainable=True,  # 允许继续训练
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    model.print_trainable_parameters()

    # 4. DPO 训练配置
    print("\n配置 Round 2 DPO 训练...")
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
    print("\n开始 Round 2 DPO 训练...")
    print(f"  基础模型: {BASE_MODEL_ID}")
    print(f"  初始adapter: {ROUND1_PEFT_ID}")
    print(f"  数据量: {len(dataset)} 条")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Beta: {BETA}\n")

    train_result = trainer.train()

    # 7. 打印结果
    print("\n训练完成！")
    print(f"  总步数: {train_result.global_step}")
    print(f"  训练损失: {train_result.training_loss:.4f}")

    # 8. 保存并上传
    print(f"\n保存 adapter 到: {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"\n上传到 HuggingFace: {HF_REPO_ID}")
    trainer.model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    print(f"✅ 上传完成！")
    print(f"模型链接: https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()