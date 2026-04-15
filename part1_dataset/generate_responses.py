
import json
import random
import torch
import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv

"""
Part 1 - Step 1: 生成偏好数据集的原始回答
从 Alpaca 数据集采样 500 条 instruction，
用 LLaMA-3.2 1B 以两种不同温度生成回答，制造质量差异。
输出: data/raw_responses.jsonl
"""

# ── 环境变量 ──────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN 未设置，请检查 .env 文件")
login(token=HF_TOKEN)

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
NUM_SAMPLES = 500
SEED = 42
OUTPUT_PATH = Path("data/raw_responses.jsonl")

# 两个回答的温度设置：高温度回答质量较差，低温度回答质量较好
TEMP_HIGH = 1.2   # rejected 候选（更随机、质量偏低）
TEMP_LOW  = 0.3   # chosen 候选（更保守、质量偏高）
MAX_NEW_TOKENS = 256

# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_alpaca_instructions(num_samples: int, seed: int) -> list[dict]:
    """从 Alpaca 数据集采样 instruction，过滤掉有 input 字段的复杂样本"""
    print(f"加载 Alpaca 数据集，采样 {num_samples} 条...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # 只保留没有额外 input 的单轮 instruction（更干净）
    filtered = [
        item for item in dataset
        if not item["input"].strip()
    ]

    random.seed(seed)
    sampled = random.sample(filtered, num_samples)
    print(f"过滤后可用样本: {len(filtered)}，已采样: {num_samples}")
    return sampled


# ── 模型加载 ──────────────────────────────────────────────────────────────────
def load_model(model_id: str):
    """加载 LLaMA-3.2 1B Instruct 模型"""
    print(f"加载模型: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN
    )
    model.eval()
    print("模型加载完成")
    return tokenizer, model


# ── 推理 ──────────────────────────────────────────────────────────────────────
def generate_response(
    instruction: str,
    tokenizer,
    model,
    temperature: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """用指定温度生成一条回答"""
    # 用 chat template 格式化输入
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只取新生成的 token
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. 采样 instruction
    samples = load_alpaca_instructions(NUM_SAMPLES, SEED)

    # 2. 加载模型
    tokenizer, model = load_model(MODEL_ID)

    # 3. 逐条生成两个回答，实时写入 jsonl（防止中途崩溃丢数据）
    print(f"\n开始生成回答，共 {NUM_SAMPLES} 条...\n")
    success_count = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, item in enumerate(samples):
            instruction = item["instruction"]

            try:
                # 低温度回答（质量较好，作为 chosen 候选）
                response_low = generate_response(
                    instruction, tokenizer, model, temperature=TEMP_LOW
                )
                # 高温度回答（质量较差，作为 rejected 候选）
                response_high = generate_response(
                    instruction, tokenizer, model, temperature=TEMP_HIGH
                )

                record = {
                    "instruction": instruction,
                    "response_low_temp": response_low,    # chosen 候选
                    "response_high_temp": response_high,  # rejected 候选
                    "temp_low": TEMP_LOW,
                    "temp_high": TEMP_HIGH,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                success_count += 1

                # 每 50 条打印进度
                if (i + 1) % 50 == 0:
                    print(f"进度: {i + 1}/{NUM_SAMPLES} | 已完成: {success_count}")
                    print(f"  指令: {instruction[:60]}...")
                    print(f"  低温回答: {response_low[:80]}...")
                    print(f"  高温回答: {response_high[:80]}...\n")

            except Exception as e:
                print(f"第 {i+1} 条生成失败: {e}，跳过")
                continue

    print(f"\n✅ 生成完成！成功: {success_count}/{NUM_SAMPLES}")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()