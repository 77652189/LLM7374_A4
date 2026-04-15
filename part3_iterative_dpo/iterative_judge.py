import json
import os
import random
import torch
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

"""
Part 3 - Iterative DPO: Round 2 数据生成
用第1轮 DPO 模型作为 Judge（Self-Rewarding 思路）
生成新的偏好数据集用于第2轮训练
输出: data/preference_dataset_round2.jsonl
"""


# ── 环境变量 ──────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN 未设置，请检查 .env 文件")
login(token=HF_TOKEN)

# ── 配置 ──────────────────────────────────────────────────────────────────────
BASE_MODEL_ID  = "meta-llama/Llama-3.2-3B-Instruct"
PEFT_MODEL_ID  = "Nayc/llama-3.2-3b-dpo"
NUM_SAMPLES    = 100
SEED           = 123   # 不同于第一轮的42，确保采样到新的instruction
TEMP_LOW       = 0.3
TEMP_HIGH      = 1.2
MAX_NEW_TOKENS = 256
OUTPUT_PATH    = Path(__file__).parent / "data" / "preference_dataset_round2.jsonl"

# ── Judge Prompt ──────────────────────────────────────────────────────────────
# Self-Rewarding: 用 DPO 模型自己作为 judge
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Compare the two responses below and determine which one is better.

Evaluate based on:
1. Helpfulness (1-5)
2. Accuracy (1-5)
3. Completeness (1-5)
4. Clarity (1-5)
5. Safety (1-5)

Instruction: {instruction}

Response A: {response_a}

Response B: {response_b}

First provide brief reasoning, then output exactly one of: WINNER: A or WINNER: B"""


# ── 模型加载 ──────────────────────────────────────────────────────────────────
def load_base_model(model_id: str):
    print(f"加载 base 模型: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()
    print("Base 模型加载完成")
    return tokenizer, model


def load_dpo_judge(base_model, peft_model_id: str):
    print(f"\n加载 DPO Judge 模型: {peft_model_id}")
    dpo_model = PeftModel.from_pretrained(
        base_model,
        peft_model_id,
        token=HF_TOKEN,
    )
    dpo_model.eval()
    print("DPO Judge 加载完成")
    return dpo_model


# ── 生成回答 ──────────────────────────────────────────────────────────────────
def generate_response(instruction: str, tokenizer, model, temperature: float) -> str:
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── DPO Judge 评判 ────────────────────────────────────────────────────────────
def judge_with_dpo_model(
    instruction: str,
    response_a: str,
    response_b: str,
    tokenizer,
    judge_model,
) -> dict:
    """用本地 DPO 模型作为 judge 评判两个回答"""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        response_a=response_a[:300],   # 截断避免太长
        response_b=response_b[:300],
    )
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(judge_model.device)

    with torch.no_grad():
        output_ids = judge_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # 解析 winner
    if "WINNER: A" in raw_output.upper():
        winner = "A"
        chosen, rejected = response_a, response_b
    elif "WINNER: B" in raw_output.upper():
        winner = "B"
        chosen, rejected = response_b, response_a
    else:
        # fallback: 低温度回答作为 chosen
        winner = "A"
        chosen, rejected = response_a, response_b

    return {
        "winner": winner,
        "chosen": chosen,
        "rejected": rejected,
        "raw_judge_output": raw_output,
    }


# ── 数据采样 ──────────────────────────────────────────────────────────────────
def sample_new_instructions(num_samples: int, seed: int) -> list[dict]:
    """从 Alpaca 采样新的 instruction，排除第一轮用过的"""
    print(f"\n采样新 instruction: {num_samples} 条 (seed={seed})")

    # 加载第一轮用过的 instruction
    round1_path = Path(__file__).parent.parent / "part1_dataset" / "data" / "preference_dataset.jsonl"
    round1_prompts = set()
    if round1_path.exists():
        with open(round1_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                round1_prompts.add(r["prompt"])
    print(f"排除第一轮已用 instruction: {len(round1_prompts)} 条")

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    filtered = [
        item for item in dataset
        if not item["input"].strip() and item["instruction"] not in round1_prompts
    ]

    random.seed(seed)
    sampled = random.sample(filtered, num_samples)
    print(f"新采样: {num_samples} 条")
    return sampled


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. 采样新 instruction
    samples = sample_new_instructions(NUM_SAMPLES, SEED)

    # 2. 加载 base 模型（用于生成两个回答）
    tokenizer, base_model = load_base_model(BASE_MODEL_ID)

    # 3. 加载 DPO 模型作为 judge
    judge_model = load_dpo_judge(base_model, PEFT_MODEL_ID)

    # 4. 逐条生成回答 + judge 评判
    print(f"\n开始生成偏好数据（Round 2），共 {NUM_SAMPLES} 条...\n")
    success_count = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, item in enumerate(samples):
            instruction = item["instruction"]

            try:
                # 生成两个回答
                response_a = generate_response(
                    instruction, tokenizer, base_model, TEMP_LOW
                )
                response_b = generate_response(
                    instruction, tokenizer, base_model, TEMP_HIGH
                )

                # DPO 模型作为 judge
                result = judge_with_dpo_model(
                    instruction, response_a, response_b,
                    tokenizer, judge_model
                )

                record = {
                    "prompt":           instruction,
                    "chosen":           result["chosen"],
                    "rejected":         result["rejected"],
                    "winner":           result["winner"],
                    "raw_judge_output": result["raw_judge_output"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                success_count += 1

                if (i + 1) % 10 == 0:
                    print(f"进度: {i+1}/{NUM_SAMPLES} | 成功: {success_count}")
                    print(f"  指令: {instruction[:60]}...")
                    print(f"  Winner: {result['winner']}\n")

            except Exception as e:
                print(f"第 {i+1} 条失败: {e}，跳过")
                continue

    print(f"\n✅ Round 2 数据生成完成！成功: {success_count}/{NUM_SAMPLES}")
    print(f"输出: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()