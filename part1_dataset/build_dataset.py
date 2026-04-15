
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from huggingface_hub import login

"""
Part 1 - Step 3: 上传偏好数据集到 HuggingFace Hub
读取 data/preference_dataset.jsonl，
转换为 HuggingFace Dataset 格式并上传。
"""

# ── 环境变量 ──────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN 未设置，请检查 .env 文件")
login(token=HF_TOKEN)

# ── 配置 ──────────────────────────────────────────────────────────────────────
INPUT_PATH   = Path("data/preference_dataset.jsonl")
HF_REPO_ID   = "Nayc/dpo-preference-dataset"

# ── 加载数据 ──────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def main():
    print(f"读取数据集: {INPUT_PATH}")
    records = load_jsonl(INPUT_PATH)
    print(f"共 {len(records)} 条记录")

    # 只保留 DPO 训练需要的字段
    dpo_records = [
        {
            "prompt":    r["prompt"],
            "chosen":    r["chosen"],
            "rejected":  r["rejected"],
        }
        for r in records
    ]

    # 转换为 HuggingFace Dataset
    dataset = Dataset.from_list(dpo_records)
    print(f"\n数据集结构:")
    print(dataset)
    print(f"\n示例:")
    for i in range(min(2, len(dataset))):
        print(f"\n[{i+1}] Prompt:   {dataset[i]['prompt'][:80]}...")
        print(f"     Chosen:   {dataset[i]['chosen'][:80]}...")
        print(f"     Rejected: {dataset[i]['rejected'][:80]}...")

    # 上传到 HuggingFace Hub
    print(f"\n上传到 HuggingFace: {HF_REPO_ID}")
    dataset.push_to_hub(
        HF_REPO_ID,
        token=HF_TOKEN,
        private=False,
    )
    print(f"✅ 上传完成！")
    print(f"数据集链接: https://huggingface.co/datasets/{HF_REPO_ID}")

if __name__ == "__main__":
    main()