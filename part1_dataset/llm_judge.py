
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

"""
Part 1 - Step 2: LLM Judge 裁判偏好数据集
读取 data/raw_responses.jsonl，
用 OpenRouter (llama-3.3-70b-instruct:free) 作为 Judge 判断哪个回答更好。
支持断点续跑：已完成的条目会自动跳过。
输出: data/preference_dataset.jsonl
"""

# ── 环境变量 ──────────────────────────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY 未设置，请检查 .env 文件")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ── 配置 ──────────────────────────────────────────────────────────────────────
JUDGE_MODEL   = "nvidia/nemotron-3-super-120b-a12b:free"
INPUT_PATH    = Path("data/raw_responses.jsonl")
OUTPUT_PATH   = Path("data/preference_dataset.jsonl")
EXAMPLES_PATH = Path("data/judge_examples.jsonl")
MAX_SAMPLES   = 100   # 目标总量
RETRY_DELAY   = 10
MAX_RETRIES   = 3

# ── Judge Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI assistant evaluator. Your task is to compare two responses to a given instruction and determine which response is better.

Evaluate each response on the following five dimensions:
1. **Helpfulness** (1-5): Does the response directly and completely address what was asked?
2. **Accuracy** (1-5): Is the information factually correct and reliable?
3. **Completeness** (1-5): Does the response cover all relevant aspects without unnecessary omissions?
4. **Clarity** (1-5): Is the response well-organized, easy to understand, and well-written?
5. **Safety** (1-5): Is the response free from harmful, biased, or inappropriate content?

Scoring Guide:
- 5: Excellent - exceeds expectations on this dimension
- 4: Good - meets expectations well
- 3: Adequate - meets basic expectations
- 2: Poor - partially meets expectations
- 1: Unacceptable - fails to meet expectations

You must follow this exact output format:
<reasoning>
Dimension-by-dimension analysis comparing Response A and Response B.
</reasoning>
<scores>
Response A: Helpfulness=X, Accuracy=X, Completeness=X, Clarity=X, Safety=X, Total=X
Response B: Helpfulness=X, Accuracy=X, Completeness=X, Clarity=X, Safety=X, Total=X
</scores>
<winner>A</winner> or <winner>B</winner>"""

FEW_SHOT_EXAMPLES = """Here are examples of how to evaluate responses:

---
EXAMPLE 1:

Instruction: What is photosynthesis?

Response A: Photosynthesis is when plants use sunlight. They need water too. It makes food for the plant.

Response B: Photosynthesis is the biological process by which plants, algae, and some bacteria convert light energy (typically from the sun) into chemical energy stored as glucose. The process occurs in the chloroplasts and can be summarized as: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2. It consists of two stages: the light-dependent reactions (which capture solar energy and produce ATP) and the Calvin cycle (which uses that energy to fix carbon dioxide into sugar).

<reasoning>
Helpfulness: Response A gives a vague answer that barely addresses what photosynthesis is. Response B directly answers with a complete scientific explanation including the chemical equation and stages. B is far more helpful (A:2, B:5).
Accuracy: Response A is not wrong but omits critical details. Response B provides accurate information including the chemical equation and two-stage description. (A:3, B:5).
Completeness: Response A only covers the very basics. Response B includes the organisms involved, the location (chloroplasts), the chemical equation, and the two main stages. (A:2, B:5).
Clarity: Response A is simple but lacks structure. Response B is well-organized with a logical flow from definition to mechanism. (A:3, B:4).
Safety: Both responses are safe. (A:5, B:5).
</reasoning>
<scores>
Response A: Helpfulness=2, Accuracy=3, Completeness=2, Clarity=3, Safety=5, Total=15
Response B: Helpfulness=5, Accuracy=5, Completeness=5, Clarity=4, Safety=5, Total=24
</scores>
<winner>B</winner>

---
EXAMPLE 2:

Instruction: Give me tips for better sleep.

Response A: Here are evidence-based tips for better sleep: 1) Maintain a consistent sleep schedule, even on weekends, to regulate your circadian rhythm. 2) Keep your bedroom cool (around 65-68F/18-20C), dark, and quiet. 3) Avoid screens 1 hour before bed due to blue light suppressing melatonin. 4) Limit caffeine after 2 PM since it has a 5-6 hour half-life. 5) Try relaxation techniques like deep breathing or progressive muscle relaxation. 6) Avoid large meals within 2-3 hours of bedtime.

Response B: Sleep is important. You should try to sleep at the same time. Don't drink coffee. Make sure your room is dark. Exercise can also be good sometimes if you feel like it maybe.

<reasoning>
Helpfulness: Response A provides actionable, specific tips backed by reasoning. Response B gives vague suggestions without any useful detail. (A:5, B:2).
Accuracy: Response A includes accurate scientific details (melatonin, caffeine half-life, temperature range). Response B is technically not wrong but too vague to be reliable. (A:5, B:2).
Completeness: Response A covers 6 distinct evidence-based areas. Response B only gestures at a few topics without depth. (A:5, B:2).
Clarity: Response A uses a numbered list with clear explanations. Response B is disorganized and uncertain in tone ("maybe", "sometimes"). (A:5, B:2).
Safety: Both are safe. (A:5, B:5).
</reasoning>
<scores>
Response A: Helpfulness=5, Accuracy=5, Completeness=5, Clarity=5, Safety=5, Total=25
Response B: Helpfulness=2, Accuracy=2, Completeness=2, Clarity=2, Safety=5, Total=13
</scores>
<winner>A</winner>

---
Now evaluate the following:"""


# ── Judge 调用 ────────────────────────────────────────────────────────────────
def judge_responses(instruction: str, response_a: str, response_b: str) -> dict:
    """调用 OpenRouter Judge 裁判两个回答"""
    user_message = f"""{FEW_SHOT_EXAMPLES}

Instruction: {instruction}

Response A: {response_a}

Response B: {response_b}

Evaluate both responses following the format above."""

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            raw_output = completion.choices[0].message.content
            return parse_judge_output(raw_output, response_a, response_b)

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  调用失败 (attempt {attempt+1}): {e}，{RETRY_DELAY}s 后重试")
                time.sleep(RETRY_DELAY)
            else:
                raise e


def parse_judge_output(raw: str, response_a: str, response_b: str) -> dict:
    """解析 Judge 的结构化输出"""
    result = {"raw_output": raw, "winner": None, "chosen": None, "rejected": None}

    if "<winner>A</winner>" in raw:
        result["winner"]   = "A"
        result["chosen"]   = response_a
        result["rejected"] = response_b
    elif "<winner>B</winner>" in raw:
        result["winner"]   = "B"
        result["chosen"]   = response_b
        result["rejected"] = response_a
    else:
        result["winner"]       = "A"
        result["chosen"]       = response_a
        result["rejected"]     = response_b
        result["parse_failed"] = True

    return result


# ── 断点续跑 ──────────────────────────────────────────────────────────────────
def load_completed_prompts(output_path: Path) -> set:
    completed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    completed.add(record["prompt"])
                except Exception:
                    continue
        print(f"断点续跑：已完成 {len(completed)} 条，跳过这些条目")
    return completed


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"读取原始数据: {len(records)} 条")

    completed_prompts = load_completed_prompts(OUTPUT_PATH)

    remaining = [r for r in records if r["instruction"] not in completed_prompts]
    remaining = remaining[:MAX_SAMPLES - len(completed_prompts)]
    print(f"待处理: {len(remaining)} 条（目标总量: {MAX_SAMPLES}）\n")

    if not remaining:
        print("✅ 已达到目标数量！")
        return

    examples_saved = 0
    if EXAMPLES_PATH.exists():
        with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
            examples_saved = sum(1 for _ in f)

    success_count = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f, \
         open(EXAMPLES_PATH, "a", encoding="utf-8") as ex_f:

        for i, record in enumerate(remaining):
            instruction = record["instruction"]
            response_a  = record["response_low_temp"]
            response_b  = record["response_high_temp"]

            try:
                result = judge_responses(instruction, response_a, response_b)

                dpo_record = {
                    "prompt":           instruction,
                    "chosen":           result["chosen"],
                    "rejected":         result["rejected"],
                    "winner":           result["winner"],
                    "raw_judge_output": result["raw_output"],
                }
                out_f.write(json.dumps(dpo_record, ensure_ascii=False) + "\n")
                success_count += 1

                if examples_saved < 5:
                    ex_f.write(json.dumps(dpo_record, ensure_ascii=False) + "\n")
                    examples_saved += 1

                if (i + 1) % 10 == 0:
                    total_done = len(completed_prompts) + success_count
                    print(f"进度: {total_done}/{MAX_SAMPLES} | 本次成功: {success_count}")
                    print(f"  指令: {instruction[:60]}...")
                    print(f"  Winner: {result['winner']}\n")

                time.sleep(2)  # 保守一点，避免限速

            except Exception as e:
                print(f"第 {i+1} 条 Judge 失败: {e}，跳过")
                continue

    total_done = len(completed_prompts) + success_count
    print(f"\n✅ 本次完成！新增: {success_count} 条，累计: {total_done}/{MAX_SAMPLES}")
    print(f"偏好数据集: {OUTPUT_PATH}")
    print(f"示例文件:   {EXAMPLES_PATH}")


if __name__ == "__main__":
    main()