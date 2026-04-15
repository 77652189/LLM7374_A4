
import os
import torch
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

"""
Part 3 - Iterative DPO: 三模型对比
对比 Base Model vs DPO Round1 vs DPO Round2
使用与 Part 2b 相同的10条 instruction
输出: results_iterative.csv
"""

# ── 环境变量 ──────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

# ── 配置 ──────────────────────────────────────────────────────────────────────
BASE_MODEL_ID   = "meta-llama/Llama-3.2-3B-Instruct"
PEFT_ROUND1_ID  = "Nayc/llama-3.2-3b-dpo"
PEFT_ROUND2_ID  = "Nayc/llama-3.2-3b-dpo-round2"
OUTPUT_PATH     = Path(__file__).parent / "results_iterative.csv"
MAX_NEW_TOKENS  = 256
TEMPERATURE     = 0.7

# ── 同 Part 2b 的10条 instruction ─────────────────────────────────────────────
INSTRUCTIONS = [
    "Explain the concept of recursion in programming.",
    "Write a haiku about autumn leaves.",
    "What are the health benefits of drinking green tea?",
    "How do I negotiate a salary raise?",
    "Describe the water cycle in simple terms.",
    "Give me 3 tips for improving my public speaking skills.",
    "What is the difference between a democracy and a republic?",
    "How do I make a basic tomato pasta sauce from scratch?",
    "Explain why the sky is blue.",
    "What are some strategies for managing stress at work?",
]

# ── 推理函数 ──────────────────────────────────────────────────────────────────
def generate(instruction: str, tokenizer, model) -> str:
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_all(instructions: list, tokenizer, model, label: str) -> list:
    print(f"\n生成 {label} 回答...")
    responses = []
    for i, instruction in enumerate(instructions):
        response = generate(instruction, tokenizer, model)
        responses.append(response)
        print(f"  [{i+1}/10] {instruction[:50]}...")
    return responses


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. 加载 tokenizer 和 base 模型
    print(f"加载 tokenizer + base 模型: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    base_model.eval()

    # 2. Base model 生成
    base_responses = generate_all(INSTRUCTIONS, tokenizer, base_model, "Base Model")

    # 3. Round 1 DPO 生成
    print(f"\n加载 DPO Round 1: {PEFT_ROUND1_ID}")
    round1_model = PeftModel.from_pretrained(
        base_model, PEFT_ROUND1_ID, token=HF_TOKEN
    )
    round1_model.eval()
    round1_responses = generate_all(INSTRUCTIONS, tokenizer, round1_model, "DPO Round 1")

    # 4. Round 2 DPO 生成
    print(f"\n加载 DPO Round 2: {PEFT_ROUND2_ID}")
    # 需要从 base 重新加载避免 adapter 叠加
    base_model2 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    round2_model = PeftModel.from_pretrained(
        base_model2, PEFT_ROUND2_ID, token=HF_TOKEN
    )
    round2_model.eval()
    round2_responses = generate_all(INSTRUCTIONS, tokenizer, round2_model, "DPO Round 2")

    # 5. 构造 DataFrame
    df = pd.DataFrame({
        "instruction":      INSTRUCTIONS,
        "base_response":    base_responses,
        "dpo_round1":       round1_responses,
        "dpo_round2":       round2_responses,
    })

    # 6. 打印结果
    print("\n" + "="*80)
    print("三模型对比结果")
    print("="*80)
    for i, row in df.iterrows():
        print(f"\n[{i+1}] Instruction: {row['instruction']}")
        print(f"\n  Base:    {row['base_response'][:150]}...")
        print(f"\n  Round 1: {row['dpo_round1'][:150]}...")
        print(f"\n  Round 2: {row['dpo_round2'][:150]}...")
        print("-"*80)

    # 7. 保存
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\n✅ 结果保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()