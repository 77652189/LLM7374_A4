import os
import torch
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

"""
Part 2b: Comparative Analysis
对比原始 LLaMA-3.2-3B 和 DPO 微调模型在10条新 instruction 上的输出
输出: part2_evaluation/results.csv
"""


# ── 环境变量 ──────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

# ── 配置 ──────────────────────────────────────────────────────────────────────
BASE_MODEL_ID  = "meta-llama/Llama-3.2-3B-Instruct"
PEFT_MODEL_ID  = "Nayc/llama-3.2-3b-dpo"
OUTPUT_PATH    = Path(__file__).parent / "results.csv"
MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.7

# ── 10条新 instruction ────────────────────────────────────────────────────────
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


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. 加载 tokenizer
    print(f"加载 tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载原始模型
    print(f"\n加载原始模型: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    base_model.eval()

    # 3. 生成原始模型回答
    print("\n生成原始模型回答...")
    base_responses = []
    for i, instruction in enumerate(INSTRUCTIONS):
        response = generate(instruction, tokenizer, base_model)
        base_responses.append(response)
        print(f"  [{i+1}/10] {instruction[:50]}...")

    # 4. 加载 DPO 微调模型
    print(f"\n加载 DPO 微调模型: {PEFT_MODEL_ID}")
    dpo_model = PeftModel.from_pretrained(
        base_model,
        PEFT_MODEL_ID,
        token=HF_TOKEN,
    )
    dpo_model.eval()

    # 5. 生成 DPO 模型回答
    print("\n生成 DPO 模型回答...")
    dpo_responses = []
    for i, instruction in enumerate(INSTRUCTIONS):
        response = generate(instruction, tokenizer, dpo_model)
        dpo_responses.append(response)
        print(f"  [{i+1}/10] {instruction[:50]}...")

    # 6. 构造 DataFrame
    df = pd.DataFrame({
        "instruction":    INSTRUCTIONS,
        "base_response":  base_responses,
        "dpo_response":   dpo_responses,
    })

    # 7. 打印结果
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)
    for i, row in df.iterrows():
        print(f"\n[{i+1}] Instruction: {row['instruction']}")
        print(f"\n  Base Model:\n  {row['base_response'][:200]}...")
        print(f"\n  DPO Model:\n  {row['dpo_response'][:200]}...")
        print("-"*80)

    # 8. 保存结果
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\n✅ 结果保存到: {OUTPUT_PATH}")
    print(df[["instruction", "base_response", "dpo_response"]].to_string())


if __name__ == "__main__":
    main()